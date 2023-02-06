using Spglib

struct SymOp{T <: Real}
    # (Uu)(x) = u(W x + w) in real space
    W::Mat3{Int}
    w::Vec3{T}

    # (Uu)(G) = e^{-i G τ} u(S^-1 G) in reciprocal space
    S::Mat3{Int}
    τ::Vec3{T}
end
function SymOp(W, w::AbstractVector{T}) where {T}
    w = mod.(w, 1)
    S = W'
    τ = -W \ w
    SymOp{T}(W, w, S, τ)
end

function symmetry_operations(lattice, positions, atoms)
    atom_indices = map(x -> findfirst(x .== unique(atoms))::Int, atoms)
    cell = Cell(collect(eachcol(lattice)), positions, atom_indices)

    rotations, translations = Spglib.get_symmetry(cell)
    nsymmetry = size(rotations, 3)
    # Note: Transposes are performed to convert between spglib row-major to julia column-major
    Ws = [Mat3{Int}(rotations[:, :, i]') for i in 1:nsymmetry]
    ws = [Vec3{eltype(lattice)}(translations[:, i]) for i in 1:nsymmetry]
    [SymOp(W, w) for (W, w) in zip(Ws, ws)]
end

"""
# Input
- `kpts`: k points in reduced coordinates on the uniform grid
- `ngrid`: size of the k point grid
- `symmetry`: vector of SymOp

# Output
- `ikirr_to_ik`: `ikirr_to_ik[ikirr]` is the index of an i-th irreducible k point in `kpts`
- `ik_to_ikirr`: `ikirr = ik_to_ikirr[ik]` is the index of `ikirr_to_ik` such that the k point `kpts[ikirr_to_ik[ikirr]]` maps to `kpts[ik]`
- `ikirr_isym_to_isk`: `ikirr_isym_to_isk[ikirr][isym]` is the index of `sxk = symmetry[isym].S * kpts[ikirr_to_ik[ikirr]]` in kpts
"""
function compute_irreducible_kpoints(kpts, ngrid, symmetry)
    kpts_rational = [round.(Int, xk .* ngrid) .// ngrid for xk in vec(kpts)]
    nktot = length(kpts_rational)

    ikirr_to_ik = Int[]
    ik_to_ikirr = zeros(Int, nktot)
    for ik in eachindex(kpts_rational)
        # Check if kpts[ik] is equivalent to k points in ikirr_to_ik
        for (i, ikirr) in enumerate(ikirr_to_ik)
            for symop in symmetry
                if all(isinteger, symop.S * kpts_rational[ikirr] - kpts_rational[ik])
                    ik_to_ikirr[ik] = i
                    break
                end
            end
        end
        if ik_to_ikirr[ik] == 0
            # ik is irreducible
            push!(ikirr_to_ik, ik)
            ik_to_ikirr[ik] = length(ikirr_to_ik)
        end
    end

    nsymmetry = length(symmetry)
    nkirr = length(ikirr_to_ik)
    ikirr_isym_to_isk = [zeros(Int, nsymmetry) for _ in 1:nkirr]
    for (ikirr, ik) in enumerate(ikirr_to_ik)
        for (isym, symop) in enumerate(symmetry)
            sxk = symop.S * kpts_rational[ik]
            ikirr_isym_to_isk[ikirr][isym] = findfirst(xk -> all(isinteger, xk - sxk), kpts_rational)
        end
    end

    (; ikirr_to_ik, ik_to_ikirr, ikirr_isym_to_isk, nsymmetry, nkirr)
end


"""
    construct_dmn(lattice, positions, atoms, orbitals, kpts, ngrid, amn, eig)
Compute the symmetry and dmn matrix elements using the atomic structure and the `amn` matrix.
"""
function construct_dmn(lattice, positions, atoms, orbitals, kpts, ngrid, amn, eig)
    # Exclude state from outer window if projection to initial guesses is small
    nband = size(eig, 1)
    projectability = [trues(nband) for _ in eachindex(kpts)]
    projectability_lower_cutoff = 1e-6
    for ik in eachindex(kpts)
        for rng in degenerate_groups(eig[:, ik])
            if any(ib -> norm(amn[ib, :, ik])^2 < projectability_lower_cutoff, rng)
                projectability[ik][rng] .= false
            end
        end
    end

    # Compute symmetry of the lattice
    symmetry = symmetry_operations(lattice, positions, atoms)

    # Find irreducible k points
    irr_k_data = compute_irreducible_kpoints(kpts, ngrid, symmetry)

    # Compute d matrices
    d_matrix_wann = compute_d_matrix_wann(orbitals, kpts, symmetry, irr_k_data, lattice, positions)
    d_matrix_band = compute_d_matrix_band(d_matrix_wann, irr_k_data, amn, eig, projectability)

    (; symmetry, irr_k_data.nsymmetry, irr_k_data.nkirr, irr_k_data.ik_to_ikirr, irr_k_data.ikirr_to_ik, irr_k_data.ikirr_isym_to_isk, d_matrix_wann, d_matrix_band, projectability)
end

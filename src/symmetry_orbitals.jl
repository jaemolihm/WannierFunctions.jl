using DynamicPolynomials

module PolynomialXYZ
    using DynamicPolynomials
    # FIXME: Make const?
    @polyvar x y z
    xyz = (x, y, z)
end

# Compute d_matrix_wann
# d_matrix_wann[iw, jw, isym, ikirr] = <w_{iw, k'}|S|w_{jw, k}>

# TODO: FIXME....
# k' = Sk + G
# W * r_b + w = r_a + R0
# S|w_{b, jw, R=0}> = ∑_{iw} S|w_{a, iw, R0}> c_{iw, jw}
# S|w_{b, jw, R}> = ∑_{iw} S|w_{a, iw, R0 + W*R}> c_{iw, jw}

# |w_{b, jw, k}> = ∑_R |w_{b, jw, R}> exp(i k ⋅ (R - r_b))
# S |w_{b, jw, k}> = ∑_{R, iw} S|w_{a, iw, R0 + W*R}> exp(i k ⋅ (R - r_b)) c_{iw, jw}
#                  = ∑_{R, iw} S|w_{a, iw, R}> exp(i Sk ⋅ (R - R0 - W * r_b)) c_{iw, jw}
#                  = ∑_{iw} |w_{iw, Sk}> exp(i Sk ⋅ (R - r_a)) exp(-i Sk ⋅ (2R0 + w)) c_{iw, jw}
#                  = ∑_R |w_{iw, R}> exp(i Sk ⋅ R) * c_{iw, jw}
#               = |w_{iw, Sk}> * c_{iw, jw}
# d_matrix_wann[iw, jw, isym, ikirr] = c_{iw, jw}

# Sk ⋅ (R - R0 - W * r_b))
# = Sk ⋅ (R - R0 - r_a - R0 - w))
# = Sk ⋅ (R - r_a) * Sk ⋅ (-2R0 - w))

# We need the coefficients c for S|w_{jw, R}> = |w_{iw, SR}> * c_{iw, jw}.

"""
    _rotate_polynomial(p, Wcart)
- `p`: Polynomial of x, y, z to rotate.
- `Wcart`: Rotation matrix in Cartesian basis.
"""
function _rotate_polynomial(p, Wcart)
    (; x, y, z) = PolynomialXYZ
    x_rot, y_rot, z_rot = Wcart * Vec3(x, y, z)
    subs(p, x => x_rot, y => y_rot, z => z_rot)
end

function _get_coefficients_for_basis(p, basis::MonomialVector)
    inds = map(m -> findfirst(m_ -> m_ == m, basis), monomials(p))
    coeff = zeros(ComplexF64, length(basis))
    coeff[inds] .= coefficients(p)
    coeff
end

"""
# Each WF is represented by (iatm, n_radial, angular)
# iatm: atom index
# n_radial: index for the radial part
# angular: polynomial of x, y, and z for the angular part
"""
struct AtomicOrbitalBasis
    iatm::Int
    n_radial::Int
    angular::Polynomial{true, ComplexF64}
    function AtomicOrbitalBasis(iatm, n_radial, angular)
        new(iatm, n_radial, convert(Polynomial{true, ComplexF64}, angular))
    end
end

function get_orbital(orbital; zaxis=Vec3(0, 0, 1), xaxis=Vec3(1, 0, 0))
    (; x, y, z) = PolynomialXYZ
    xaxis /= norm(xaxis)
    zaxis /= norm(zaxis)
    yaxis = cross(zaxis, xaxis)
    x_ = dot(xaxis, (x, y, z))
    y_ = dot(yaxis, (x, y, z))
    z_ = dot(zaxis, (x, y, z))
    if orbital === :s
        Polynomial(1.0 * x^0)
    elseif orbital === :pz || orbital === :p_1
        z_
    elseif orbital === :px || orbital === :p_2
        x_
    elseif orbital === :py || orbital === :p_3
        y_
    elseif orbital === :sp3_1
        1 + x_ + y_ + z_
    elseif orbital === :sp3_2
        1 + x_ - y_ - z_
    elseif orbital === :sp3_3
        1 - x_ + y_ - z_
    elseif orbital === :sp3_4
        1 - x_ - y_ + z_
    else
        error("orbital $orbital not implemented")
    end::Polynomial{true, Float64}
end

function compute_d_matrix_wann(orbitals, kpts, symmetry, irr_k_data, lattice, positions)
    nwannier = length(orbitals)
    (; nkirr, nsymmetry) = irr_k_data
    iatm_list = unique!([wf.iatm for wf in orbitals])
    n_radial_max_list = [maximum(x -> x.iatm == iatm ? x.n_radial : -1, orbitals) for iatm in iatm_list]

    ind_list = [[Int[] for _ in 1:n_radial_max] for n_radial_max in n_radial_max_list]
    angular_list = [[Polynomial{true, ComplexF64}[] for _ in 1:n_radial_max] for n_radial_max in n_radial_max_list]

    for (iw, wf) in enumerate(orbitals)
        (; iatm, n_radial, angular) = wf
        push!(ind_list[iatm][n_radial], iw)
        push!(angular_list[iatm][n_radial], angular)
    end

    # NOTE: In pw2wannier90:
    # tvec = lattice * symop.w
    # sr = Wcart'

    d_matrix_wann = zeros(ComplexF64, nwannier, nwannier, nsymmetry, nkirr);
    for (isym, symop) in enumerate(symmetry)
        Wcart = lattice * symop.W * inv(lattice)
        for iw in 1:nwannier
            # Rotate wf
            (; iatm, n_radial, angular) = orbitals[iw]
            angular_rot = _rotate_polynomial(angular, Wcart)

            # Rotate atom by their reduced coordinates
            pos_rot = symop.W \ (positions[iatm] - symop.w)
            iatm_rot = findfirst(pos -> norm(pos - pos_rot - round.(Int, pos - pos_rot)) < 1e-5, positions)

            # Find coefficients for the linear combination of angular_list[iatm_rot][n_radial] to angular_rot
            polys = angular_list[iatm_rot][n_radial]
            xyz_monomials = monomials(PolynomialXYZ.xyz, minimum(mindegree.(polys)):maximum(maxdegree.(polys)))
            coeff_basis = reduce(hcat, _get_coefficients_for_basis.(polys, Ref(xyz_monomials)))
            coeff = _get_coefficients_for_basis(angular_rot, xyz_monomials)

            d_matrix_wann[ind_list[iatm_rot][n_radial], iw, isym, :] .= coeff_basis \ coeff
        end
    end

    # Multiply the k-dependent factor
    for (isym, symop) in enumerate(symmetry)
        for ikirr in 1:nkirr
            for iatm in iatm_list
                xk = kpts[irr_k_data.ikirr_to_ik[ikirr]]
                sxk = kpts[irr_k_data.ikirr_isym_to_isk[ikirr][isym]]
                pos_rot = symop.W * positions[iatm] - symop.w
                iatm_rot = findfirst(pos -> norm(pos - pos_rot - round.(Int, pos - pos_rot)) < 1e-5, positions)
                x1 = dot(xk, pos_rot - positions[iatm_rot])
                x2 = dot(inv(symop.S) * sxk - xk, symop.w)
                phase = cispi(2 * (x1 + x2))
                d_matrix_wann[vcat(ind_list[iatm]...), vcat(ind_list[iatm_rot]...), isym, ikirr] .*= phase
            end
        end
    end
    d_matrix_wann
end

"""
Compute d_matrix_band from amn, eig, and d_matrix_wann
A[iSk] = D_band[ik] * A[ik] * D_wann[ik]'
For a degenerate group rng,
A[iSk][rng, :] = D_band[ik][rng, rng] * A[ik][rng, :] * D_wann[ik]'
"""
function compute_d_matrix_band(d_matrix_wann, irr_k_data, amn, eig, l_outer)
    nband = size(amn, 1)
    (; nkirr, nsymmetry) = irr_k_data
    d_matrix_band = zeros(ComplexF64, nband, nband, nsymmetry, nkirr)
    for ikirr in 1:nkirr, isym in 1:nsymmetry
        ik = irr_k_data.ikirr_to_ik[ikirr]
        isk = irr_k_data.ikirr_isym_to_isk[ikirr][isym]
        D_wann = d_matrix_wann[:, :, isym, ikirr]
        A_k = amn[:, :, ik]
        A_Sk = amn[:, :, isk]
        for rng in degenerate_groups(eig[:, ik])
            # Skip if the states are not inside the outer window
            any(.!l_outer[ik][rng]) && continue

            # Check states with very low projectability are excluded
            if any(sum(abs.(A_k[rng, :]), dims=2) .< 1e-5)
                @warn "Projectability is too low, ikirr=$ikirr, rng=$rng. These states must be excluded from l_outer."
            end

            D_band_rng = A_Sk[rng, :] / (A_k[rng, :] * D_wann')
            d_matrix_band[rng, rng, isym, ikirr] .= D_band_rng
        end
    end
    d_matrix_band
end

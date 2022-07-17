using LinearAlgebra

export SymmetryConstraintObjective
export AugmentationObjective
export update_multipliers!

"""
Constraints for symmetry-adapted Wannier functions
"""
struct SymmetryConstraintObjective <: AbstractWannierObjective
    nkirr::Int
    nsymmetry::Int
    d_matrix_band::Array{ComplexF64, 4}
    d_matrix_wann::Array{ComplexF64, 4}
    ik_to_ikirr::Vector{Int}
    ikirr_to_ik::Vector{Int}
    ikirr_isym_to_isk::Vector{Vector{Int}}
    constraints_to_use::Vector{NTuple{2, Int}}
end

function Base.show(io::IO, obj::SymmetryConstraintObjective)
    print(io, "SymmetryConstraintFunctional(")
    print(io, "nkirr = ", obj.nkirr)
    print(io, ", nsymmetry = ", obj.nsymmetry)
    print(io, ")")
end

function SymmetryConstraintObjective(seedname, nwannier)
    dmn = read_dmn(seedname, nwannier)

    # Matrices d_matrix_band and d_matrix_wann should be unitary but numerical error can
    # slightly break it. Here we enforce unitarity to avoid further numerial issues.
    for ikirr in 1:dmn.nkirr
        for isym in 1:dmn.nsymmetry
            u, s, v = svd(dmn.d_matrix_band[:, :, isym, ikirr])
            dmn.d_matrix_band[:, :, isym, ikirr] .= u * v'

            u, s, v = svd(dmn.d_matrix_wann[:, :, isym, ikirr])
            dmn.d_matrix_wann[:, :, isym, ikirr] .= u * v'
        end
    end

    constraints_to_use = NTuple{2, Int}[]
    # isk_found = Int[]
    for ikirr in 1:dmn.nkirr
        for isym in 1:dmn.nsymmetry
            push!(constraints_to_use, (ikirr, isym))
            # isym == 1 && continue
            # ik = dmn.ikirr_to_ik[ikirr]
            # isk = dmn.ikirr_isym_to_isk[ikirr][isym]
            # if ik == isk || isk ∉ isk_found
            #     push!(constraints_to_use, (ikirr, isym))
            #     ik != isk && push!(isk_found, isk)
            # end
        end
    end
    # @assert length(isk_found) == nktot - dmn.nkirr

    SymmetryConstraintObjective(dmn.nkirr, dmn.nsymmetry, dmn.d_matrix_band,
        dmn.d_matrix_wann, dmn.ik_to_ikirr, dmn.ikirr_to_ik, dmn.ikirr_isym_to_isk,
        constraints_to_use)
end

function compute_objective_and_gradient!(gradient, U, obj::SymmetryConstraintObjective, factor=1)
    objective = zero(real(eltype(U)))
    @views for (ikirr, isym) in obj.constraints_to_use
        ik = obj.ikirr_to_ik[ikirr]
        isk = obj.ikirr_isym_to_isk[ikirr][isym]
        Uk = U[:, :, ik]
        Usk = U[:, :, isk]
        D_band = obj.d_matrix_band[:, :, isym, ikirr]
        D_wann = obj.d_matrix_wann[:, :, isym, ikirr]
        C = Usk - D_band * Uk * D_wann'
        objective += factor * norm(C)^2
        if gradient !== nothing
            gradient[:, :, isk] .+= factor .* 2 .* C
            gradient[:, :, ik] .+= factor .* -2 .* (D_band' * C * D_wann)
        end
    end
    gradient !== nothing ? (; objective, gradient) : (; objective)
end

"""
Objective for augmented Lagrangian method.
objective(U) = real(sum(conj.(λ) .* (Usk - D_band * Uk * D_wann')))
"""
struct AugmentationObjective <: AbstractWannierObjective
    λ::Array{ComplexF64, 4}
    symobj::SymmetryConstraintObjective
end

function Base.show(io::IO, obj::AugmentationObjective)
    print(io, "AugmentationObjective(")
    print(io, "size(λ) = ", size(obj.λ))
    print(io, ")")
end

function AugmentationObjective(symobj)
    nband = size(symobj.d_matrix_band, 1)
    nwannier = size(symobj.d_matrix_wann, 1)
    λ = zeros(ComplexF64, nband, nwannier, symobj.nsymmetry, symobj.nkirr)
    AugmentationObjective(λ, symobj)
end

function compute_objective_and_gradient!(gradient, U, obj::AugmentationObjective, factor=1)
    (; λ, symobj) = obj
    objective = zero(real(eltype(U)))
    @views for (ikirr, isym) in symobj.constraints_to_use
        ik = symobj.ikirr_to_ik[ikirr]
        isk = symobj.ikirr_isym_to_isk[ikirr][isym]
        Uk = U[:, :, ik]
        Usk = U[:, :, isk]
        D_band = symobj.d_matrix_band[:, :, isym, ikirr]
        D_wann = symobj.d_matrix_wann[:, :, isym, ikirr]
        C = Usk - D_band * Uk * D_wann'
        objective += factor * real(sum(conj.(λ[:, :, isym, ikirr]) .* C))
        if gradient !== nothing
            gradient[:, :, isk] .+= factor .* λ[:, :, isym, ikirr]
            gradient[:, :, ik] .-= factor .* (D_band' * λ[:, :, isym, ikirr] * D_wann)
        end
    end
    gradient !== nothing ? (; objective, gradient) : (; objective)
end

function update_multipliers!(obj::AugmentationObjective, U, μ)
    (; λ, symobj) = obj
    @views for (ikirr, isym) in symobj.constraints_to_use
        ik = symobj.ikirr_to_ik[ikirr]
        isk = symobj.ikirr_isym_to_isk[ikirr][isym]
        Uk = U[:, :, ik]
        Usk = U[:, :, isk]
        D_band = symobj.d_matrix_band[:, :, isym, ikirr]
        D_wann = symobj.d_matrix_wann[:, :, isym, ikirr]
        C = Usk - D_band * Uk * D_wann'
        λ[:, :, isym, ikirr] .+= μ .* C
    end
    obj
end
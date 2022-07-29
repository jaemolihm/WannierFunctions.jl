using StaticArrays

export Vec3
export Mat3

const Vec3{T} = SVector{3, T} where T
const Mat3{T} = SMatrix{3, 3, T, 9} where T

"""
    compute_weights(bvecs_cart; shell_cutoff=1e-5)
Find `wbs` that satisfy ``∑ᵢ wbs[i] * (bvecs_cart[i] * bvecs_cart[i]') = I(3)``.
(See Eq.(B1) of Marzari and Vanderbilt (1997).)

# Inputs
- `bvecs_cart::Vector{Vec3{T}}`: b vectors in Cartesian basis
- `shell_cutoff=1e-5`: cutoff for shell construction. b vectors will be grouped into the
    same shell if `abs(|bᵢ| - |bᵢ|) < shell_cutoff`.
"""
function compute_weights(bvecs_cart; shell_cutoff=1e-5)
    T = eltype(bvecs_cart[1])

    # Setup shells: group of b vectors with same |b|
    bnorms = norm.(bvecs_cart)
    inds = sortperm(bnorms)

    # Compute ∑_{b ∈ shell} b * b' for each shell
    bnorm_prev = bnorms[inds[1]]
    inds_shell = [[inds[1]]]
    b2_shell = [vec(bvecs_cart[inds[1]] * bvecs_cart[inds[1]]')]
    for i in inds[2:end]
        b = bvecs_cart[i]
        if bnorms[i] - bnorm_prev < shell_cutoff
            # Same shell continued
            push!(inds_shell[end], i)
            b2_shell[end] += vec(b * b')
        else
            # New shell
            push!(inds_shell, [i])
            push!(b2_shell, vec(b * b'))
        end
        bnorm_prev = bnorms[i]
    end

    # Compute weights for the shell, assign them to the b vectors
    wbs_shell = reduce(hcat, b2_shell) \ vec(I(3))
    wbs = zeros(T, length(bvecs_cart))
    for (inds, wb) in zip(inds_shell, wbs_shell)
        wbs[inds] .= wb
    end

    # Check the desired equality is satisfied
    @assert sum(wb * b * b' for (b, wb) in zip(bvecs_cart, wbs)) ≈ I(3)

    wbs
end

using Test
using WannierFunctions
using StaticArrays
using LinearAlgebra

@testset "gradient MV1997" begin
    using FiniteDifferences

    # Test correctness of gradient
    BASE_FOLDER = dirname(dirname(pathof(WannierFunctions)))
    seedname = joinpath(BASE_FOLDER, "test", "data", "Cu", "Cu")

    nks = (3, 3, 3)
    nktot = prod(nks)
    kpts = [SVector(x, y, z) ./ nks for z in 0:nks[3]-1, y in 0:nks[2]-1, x in 0:nks[1]-1]

    mmn, bvecs, neighbors = read_mmn(seedname, kpts);
    amn = read_amn(seedname);
    eig = read_eig(seedname);

    nwannier = 7
    nband = 10
    nnb = length(bvecs)
    lattice = SMatrix{3, 3}([-3.411 0.0000 -3.411; 0.0000 3.411 3.411; 3.411 3.411 0.0000]) * 0.52917720859
    recip_lattice = inv(lattice)' * 2π
    wbs = [0.3713799710197723 for _ in 1:8]
    bvecs_cart = Ref(recip_lattice) .* bvecs

    functional = MarzariVanderbiltFunctional(; nband, nwannier, nktot, nnb, neighbors, wbs, bvecs_cart, mmn)

    # Initialize random unitary matrix
    U0 = zeros(ComplexF64, nband, nwannier, nktot)
    for ik in 1:nktot
        u1, _ = qr(rand(ComplexF64, nband, nband))
        u2, _ = qr(rand(ComplexF64, nwannier, nwannier))
        U0[:, :, ik] .= u1 * vcat(I(nwannier), zeros(nband - nwannier, nwannier)) * u2
    end
    gradient = compute_objective_and_gradient!(zero(U0), U0, functional).gradient

    # Apply small perturbation to U
    ΔU = rand(ComplexF64, nband, nwannier, nktot)
    compute_Ω(x) = compute_objective(U0 .+ x .* ΔU, functional).objective

    ΔΩ_finitediff = central_fdm(5, 1)(compute_Ω, 0)
    ΔΩ_grad = real(sum(conj(gradient) .* ΔU))
    @test ΔΩ_finitediff ≈ ΔΩ_grad
end
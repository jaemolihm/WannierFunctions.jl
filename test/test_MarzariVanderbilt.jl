using Test
using LinearAlgebra
using StaticArrays
using WannierFunctions

@testset "MarzariVanderbilt" begin
    BASE_FOLDER = dirname(dirname(pathof(WannierFunctions)))
    seedname = joinpath(BASE_FOLDER, "test", "data", "si", "si")

    win_data = WannierFunctions.read_win(seedname)
    (; nwannier, nband, lattice, ngrid, nktot, kpts) = win_data

    mmn, bvecs, neighbors = read_mmn(seedname, kpts);
    amn = read_amn(seedname);

    nnb = length(bvecs)
    recip_lattice = inv(lattice)' * 2π
    bvecs_cart = Ref(recip_lattice) .* bvecs
    wbs = WannierFunctions.compute_weights(bvecs_cart)

    # Initial Uk from amn
    U_initial = zeros(ComplexF64, nband, nwannier, nktot)
    for ik in 1:nktot
        u, s, v = svd(amn[:, :, ik])
        U_initial[:, :, ik] .= u * v'
    end

    l_frozen = fill(falses(nband), nktot)
    l_not_frozen = [.!x for x in l_frozen]
    p = (; nktot, nnb, nband, nwannier, bvecs_cart, wbs, neighbors, M_bands=mmn, l_frozen, l_not_frozen)
    obj_spread = MarzariVanderbiltObjective(; nband, nwannier, nktot, nnb, neighbors, wbs, bvecs_cart, mmn)

    # Test MV spread for projection-only WFs
    res = compute_objective(U_initial, obj_spread)
    @test res.objective ≈ res.spreads.Ω

    # Compare with reference data obtained by Wannier90
    spreads = res.spreads
    @test spreads.centers[1] ≈ [-0.000000048556263,   0.000000047513253,   0.000000018410154]
    @test spreads.centers[2] ≈ [-1.357773456945650,  -0.000000033901459,   1.357773443853951]
    @test spreads.centers[3] ≈ [-0.000000017549235,   1.357773468871673,   1.357773496612360]
    @test spreads.centers[4] ≈ [-1.357773469657212,   1.357773476796729,   0.000000006038467]
    @test spreads.spreads ≈ [1.934027748817770, 1.934027789252705, 1.934027832285140, 1.934027699757896]
    @test spreads.Ω ≈ 7.736111070113513
    @test spreads.ΩI ≈ 7.101265930521826
    @test spreads.ΩD ≈ 0.0 atol=1e-10
    @test spreads.ΩOD ≈ 0.6348451634749296

    # Test correctness of gradient
    @testset "gradient" begin
        using FiniteDifferences

        # Initialize random unitary matrix
        U0 = zeros(ComplexF64, nband, nwannier, nktot)
        for ik in 1:nktot
            u1, _ = qr(rand(ComplexF64, nband, nband))
            u2, _ = qr(rand(ComplexF64, nwannier, nwannier))
            U0[:, :, ik] .= u1 * vcat(I(nwannier), zeros(nband - nwannier, nwannier)) * u2
        end
        gradient = compute_objective_and_gradient!(zero(U0), U0, obj_spread).gradient

        # Apply small perturbation to U
        ΔU = rand(ComplexF64, nband, nwannier, nktot)
        compute_Ω(x) = compute_objective(U0 .+ x .* ΔU, obj_spread).objective

        ΔΩ_finitediff = central_fdm(5, 1)(compute_Ω, 0)
        ΔΩ_grad = real(sum(conj(gradient) .* ΔU))
        @test ΔΩ_finitediff ≈ ΔΩ_grad
    end
end

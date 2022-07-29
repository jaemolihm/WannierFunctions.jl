using Test
using LinearAlgebra
using StaticArrays
using WannierFunctions

@testset "silicon valence" begin
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

    # Test projection-only WFs
    spreads_initial = compute_objective(U_initial, obj_spread).spreads
    @test spreads_initial.centers[1] ≈ [-0.000000048556263,   0.000000047513253,   0.000000018410154]
    @test spreads_initial.centers[2] ≈ [-1.357773456945650,  -0.000000033901459,   1.357773443853951]
    @test spreads_initial.centers[3] ≈ [-0.000000017549235,   1.357773468871673,   1.357773496612360]
    @test spreads_initial.centers[4] ≈ [-1.357773469657212,   1.357773476796729,   0.000000006038467]
    @test spreads_initial.spreads ≈ [1.934027748817770, 1.934027789252705, 1.934027832285140, 1.934027699757896]
    @test spreads_initial.Ω ≈ 7.736111070113513
    @test spreads_initial.ΩI ≈ 7.101265930521826
    @test spreads_initial.ΩD ≈ 0.0 atol=1e-10
    @test spreads_initial.ΩOD ≈ 0.6348451634749296

    # Test maximally localized WFs
    U_optimized = run_wannier_minimization(p, U_initial, obj_spread; verbose=false)
    spreads_optimized = compute_objective(U_optimized, obj_spread).spreads
    @test spreads_optimized.Ω ≈ 7.659769027810093
    @test spreads_optimized.ΩI ≈ 7.101265930521826
    @test spreads_optimized.ΩD ≈ 0.0 atol=1e-10
    @test spreads_optimized.ΩOD ≈ 0.5585031210877284
end

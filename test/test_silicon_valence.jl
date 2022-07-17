using Test
using LinearAlgebra
using StaticArrays
using WannierFunctions

@testset "silicon valence" begin
    BASE_FOLDER = dirname(dirname(pathof(WannierFunctions)))
    seedname = joinpath(BASE_FOLDER, "test", "data", "si", "si")

    nks = (6, 6, 6)
    nktot = prod(nks)
    kpts = [SVector(x, y, z) ./ nks for z in 0:nks[3]-1, y in 0:nks[2]-1, x in 0:nks[1]-1]

    mmn, bvecs, neighbors = read_mmn(seedname, kpts);
    amn = read_amn(seedname);

    nwannier = 4
    nband = 4
    nnb = length(bvecs)
    lattice = SMatrix{3, 3}([-5.13164 0.00000 -5.13164; 0.00000 5.13164 5.13164; 5.13164 5.13164 0.00000]) * 0.52917720859
    recip_lattice = inv(lattice)' * 2π
    wbs = [3.3622298066797027 for _ in 1:8]
    bvecs_cart = Ref(recip_lattice) .* bvecs

    # Check Eq. (B1) of MV1997
    @test norm(sum(wb * b * b' for (b, wb) in zip(bvecs_cart, wbs)) - I(3)) < 1e-10

    # Initial Uk from amn
    U_initial = zeros(ComplexF64, nband, nwannier, nktot)
    for ik in 1:nktot
        u, s, v = svd(amn[:, :, ik])
        U_initial[:, :, ik] .= u * v'
    end

    l_frozen = fill(falses(nband), nktot)
    l_not_frozen = [.!x for x in l_frozen]
    p = (; nktot, nnb, nband, nwannier, bvecs_cart, wbs, neighbors, M_bands=mmn, l_frozen, l_not_frozen)
    functional = MarzariVanderbiltFunctional(; nband, nwannier, nktot, nnb, neighbors, wbs, bvecs_cart, mmn)

    # Test projection-only WFs
    spreads_initial = compute_objective(U_initial, functional).spreads
    @test spreads_initial.centers[1] ≈ [-0.000000051184847,   0.000000047216987,   0.000000021319296]
    @test spreads_initial.centers[2] ≈ [-1.357773455928676,  -0.000000033936177,   1.357773443148921]
    @test spreads_initial.centers[3] ≈ [-0.000000018920307,   1.357773469930315,   1.357773495357232]
    @test spreads_initial.centers[4] ≈ [-1.357773469725397,   1.357773476549067,   0.000000005835030]
    @test spreads_initial.spreads ≈ [1.934027772294945, 1.934027794355982, 1.934027827106826, 1.934027700254801]
    @test spreads_initial.Ω ≈ 7.736111094012553
    @test spreads_initial.ΩI ≈ 7.101265930521826
    @test spreads_initial.ΩD ≈ 0.0 atol=1e-10
    @test spreads_initial.ΩOD ≈ 0.6348451634749296

    # Test maximally localized WFs
    U_optimized = run_wannier_minimization(p, U_initial, functional; verbose=false)
    spreads_optimized = compute_objective(U_optimized, functional).spreads
    @test spreads_optimized.Ω ≈ 7.659769051625176
    @test spreads_optimized.ΩI ≈ 7.101265930521826
    @test spreads_optimized.ΩD ≈ 0.0 atol=1e-10
    @test spreads_optimized.ΩOD ≈ 0.5585031210877284
end
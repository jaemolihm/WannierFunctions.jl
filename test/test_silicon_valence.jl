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
    U_initial = zeros(ComplexF64, nwannier, nwannier, nktot)
    for ik in 1:nktot
        u, s, v = svd(amn[:, :, ik])
        U_initial[:, :, ik] .= u * v'
    end

    # Compute MV spread
    p = (; nktot, nnb, nband, nwannier, bvecs_cart, wbs, neighbors, M_bands=mmn)

    U_optimized = run_wannier_minimization(p, U_initial; f_tol=1e-20, g_tol=1e-8, verbose=false)

    res_initial = spreads_MV1997(p, U_initial)
    res_optimized = spreads_MV1997(p, U_optimized)

    @test res_initial.spreads.centers[1] ≈ [-0.000000051184847,   0.000000047216987,   0.000000021319296] atol=1e-10
    @test res_initial.spreads.centers[2] ≈ [-1.357773455928676,  -0.000000033936177,   1.357773443148921] atol=1e-10
    @test res_initial.spreads.centers[3] ≈ [-0.000000018920307,   1.357773469930315,   1.357773495357232] atol=1e-10
    @test res_initial.spreads.centers[4] ≈ [-1.357773469725397,   1.357773476549067,   0.000000005835030] atol=1e-10
    @test res_initial.spreads.spreads ≈ [1.934027772294945, 1.934027794355982, 1.934027827106826, 1.934027700254801] atol=1e-10
    @test res_initial.spreads.Ω ≈ 7.736111094012553 atol=1e-10

    @test res_optimized.spreads.Ω ≈ 7.659769051625176 atol=1e-10
end
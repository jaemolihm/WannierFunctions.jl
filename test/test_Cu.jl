using Test
using LinearAlgebra
using StaticArrays
using WannierFunctions

@testset "silicon valence" begin
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

    # Check Eq. (B1) of MV1997
    @test norm(sum(wb * b * b' for (b, wb) in zip(bvecs_cart, wbs)) - I(3)) < 1e-10

    # Initial Uk from amn
    U_initial = zeros(ComplexF64, nband, nwannier, nktot)
    for ik in 1:nktot
        u, s, v = svd(amn[:, :, ik])
        U_initial[:, :, ik] .= u * v'
    end

    # Compute MV spread
    l_frozen = fill(falses(nband), nktot)
    l_not_frozen = [.!x for x in l_frozen]
    p = (; nktot, nnb, nband, nwannier, bvecs_cart, wbs, neighbors, M_bands=mmn, l_frozen, l_not_frozen)
    res_initial = spreads_MV1997(p, U_initial)

    U_opt_no_window = run_wannier_minimization(p, U_initial; f_tol=1e-20, g_tol=1e-8, verbose=false)
    res_opt_no_window = spreads_MV1997(p, U_opt_no_window)

    dis_froz_max = 20.0
    p.l_frozen .= [eig[:, ik] .< dis_froz_max for ik in 1:p.nktot]
    p.l_not_frozen .= [.!x for x in l_frozen]
    U_opt_froz = run_wannier_minimization(p, U_initial; f_tol=1e-20, g_tol=1e-8, verbose=false)
    res_opt_froz = spreads_MV1997(p, U_opt_froz)

    # Test projection-only WFs
    r_proj_only_wannier90 = [
        [ 0.000000000031795, -0.000000000002363,  0.000000000001633],
        [ 0.000000000011655, -0.000000000014041,  0.000000000050861],
        [-0.000000000010668, -0.000000000000238,  0.000000000007438],
        [-0.000000000009023,  0.000000000060557,  0.000000000005202],
        [ 0.000000000020657, -0.000000000010528,  0.000000000015073],
        [-0.902511729222057,  0.902511729256294,  0.902511729251714],
        [ 0.902511729253828, -0.902511729252045, -0.902511729241448]
    ]
    spreads_proj_only_wannier90 = [0.645497433074877, 0.443233870388558, 0.443233870389778,
        0.645419079898551, 0.443235741490892, 1.009068611500336, 1.009068611488011]
    for iw in 1:nwannier
        @test res_initial.spreads.centers[iw] ≈ r_proj_only_wannier90[iw] atol=1e-10
    end
    @test res_initial.spreads.spreads ≈ spreads_proj_only_wannier90
    @test res_initial.spreads.Ω ≈ 4.638757218231004
    @test res_initial.spreads.ΩI ≈ 3.801670014961136
    @test res_initial.spreads.ΩD ≈ 0.005395159613377665
    @test res_initial.spreads.ΩOD ≈ 0.8316920436570605

    @test res_opt_no_window.spreads.Ω ≈ 3.2927830611080546
    @test res_opt_no_window.spreads.ΩI ≈ 3.169871256263533
    @test res_opt_no_window.spreads.ΩD ≈ 0.008861537122230756
    @test res_opt_no_window.spreads.ΩOD ≈ 0.11405026772229071

    @test res_opt_froz.spreads.Ω ≈ 3.795583395565954
    @test res_opt_froz.spreads.ΩI ≈ 3.4917706157204274
    @test res_opt_froz.spreads.ΩD ≈ 0.0015655121204359856
    @test res_opt_froz.spreads.ΩOD ≈ 0.3022472677250906

    # Check frozen window constraint is satisfied
    for ik in 1:p.nktot
        l_frozen = p.l_frozen[ik]
        @test U_opt_froz[:, :, ik]' * U_opt_froz[:, :, ik] ≈ I(p.nwannier)
        @test (U_opt_froz[:, :, ik] * U_opt_froz[:, :, ik]')[l_frozen, l_frozen] ≈ I(count(l_frozen))
    end
end
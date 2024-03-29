using Test
using LinearAlgebra
using StaticArrays
using WannierFunctions

@testset "Cu" begin
    BASE_FOLDER = dirname(dirname(pathof(WannierFunctions)))
    seedname = joinpath(BASE_FOLDER, "test", "data", "Cu", "Cu")

    win_data = WannierFunctions.read_win(seedname)
    (; nwannier, nband, lattice, ngrid, nktot, kpts) = win_data

    mmn, bvecs, neighbors = read_mmn(seedname, kpts);
    amn = read_amn(seedname);
    eig = read_eig(seedname);

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

    obj_spread = MarzariVanderbiltObjective(; nband, nwannier, nktot, nnb, neighbors, wbs, bvecs_cart, mmn)

    # Compute MV spread
    l_frozen = fill(falses(nband), nktot)
    l_not_frozen = [.!x for x in l_frozen]
    l_outer = fill(trues(nband), nktot)
    p = (; nktot, nnb, nband, nwannier, bvecs_cart, wbs, neighbors, M_bands=mmn, l_frozen, l_not_frozen, l_outer)

    # Test projection-only WFs without any window
    spreads_initial = compute_objective(U_initial, obj_spread).spreads
    r_proj_only_wannier90 = [
        [ 0.000000000031795, -0.000000000002363,  0.000000000001633],
        [ 0.000000000011655, -0.000000000014041,  0.000000000050861],
        [-0.000000000010668, -0.000000000000238,  0.000000000007437],
        [-0.000000000009023,  0.000000000060557,  0.000000000005202],
        [ 0.000000000020657, -0.000000000010528,  0.000000000015073],
        [-0.902511733195872,  0.902511733230109,  0.902511733225530],
        [ 0.902511733227644, -0.902511733225861, -0.902511733215263]
    ]
    spreads_proj_only_wannier90 = [0.645497438759206, 0.443233874291733, 0.443233874292951,
        0.645419085582191, 0.443235745394081, 1.009068620386320, 1.009068620373993]
    for iw in 1:nwannier
        @test spreads_initial.centers[iw] ≈ r_proj_only_wannier90[iw] atol=1e-10
    end
    @test spreads_initial.spreads ≈ spreads_proj_only_wannier90
    @test spreads_initial.Ω ≈ 4.638757259080474
    @test spreads_initial.ΩI ≈ 3.801670014961136
    @test spreads_initial.ΩD ≈ 0.005395159613377665
    @test spreads_initial.ΩOD ≈ 0.8316920436570605

    # Test Wannierization without any window
    U_opt_no_window = run_wannier_minimization(p, U_initial, obj_spread; verbose=false)
    spreads_opt_no_window = compute_objective(U_opt_no_window, obj_spread).spreads

    @test spreads_opt_no_window.Ω ≈ 3.2927830611080546
    @test spreads_opt_no_window.ΩI ≈ 3.169871256263533
    @test spreads_opt_no_window.ΩD ≈ 0.008861537122230756
    @test spreads_opt_no_window.ΩOD ≈ 0.11405026772229071

    # Test Wannierization with frozen window
    dis_froz_max = 20.0
    p.l_frozen .= [eig[:, ik] .< dis_froz_max for ik in 1:p.nktot]
    p.l_not_frozen .= [.!x for x in l_frozen]
    U_opt_froz = run_wannier_minimization(p, U_initial, obj_spread; verbose=false)
    spreads_opt_froz = compute_objective(U_opt_froz, obj_spread).spreads

    @test spreads_opt_froz.Ω ≈ 3.795583395565954
    @test spreads_opt_froz.ΩI ≈ 3.4917706157204274
    @test spreads_opt_froz.ΩD ≈ 0.0015655121204359856
    @test spreads_opt_froz.ΩOD ≈ 0.3022472677250906

    # Check frozen window constraint is satisfied
    for ik in 1:p.nktot
        l_frozen = p.l_frozen[ik]
        @test U_opt_froz[:, :, ik]' * U_opt_froz[:, :, ik] ≈ I(p.nwannier)
        @test (U_opt_froz[:, :, ik] * U_opt_froz[:, :, ik]')[l_frozen, l_frozen] ≈ I(count(l_frozen))
    end
end

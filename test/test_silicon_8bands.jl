using Test
using LinearAlgebra
using StaticArrays
using WannierFunctions

@testset "silicon 8band" begin
    BASE_FOLDER = dirname(dirname(pathof(WannierFunctions)))
    seedname = joinpath(BASE_FOLDER, "test", "data", "si_8band", "si")

    win_data = WannierFunctions.read_win(seedname)
    (; nwannier, nband, lattice, ngrid, nktot, kpts) = win_data

    mmn, bvecs, neighbors = read_mmn(seedname, kpts)
    amn = read_amn(seedname)
    eig = read_eig(seedname)

    nnb = length(bvecs)
    recip_lattice = inv(lattice)' * 2π
    bvecs_cart = Ref(recip_lattice) .* bvecs
    wbs = WannierFunctions.compute_weights(bvecs_cart)

    # Initial Uk from amn
    U_proj_only = zeros(ComplexF64, nband, nwannier, nktot)
    for ik in 1:nktot
        u, s, v = svd(amn[:, :, ik])
        U_proj_only[:, :, ik] .= u * v'
    end

    p = (; nktot, nnb, nband, nwannier, bvecs_cart, wbs, neighbors, M_bands=mmn)
    obj_spread = MarzariVanderbiltObjective(; nband, nwannier, nktot, nnb, neighbors, wbs, bvecs_cart, mmn)

    dis_froz_max = 8.0
    dis_win_max = 20.0

    function _test_window(l_frozen, l_outer)
        l_not_frozen = [.!x for x in l_frozen]
        p_window = (; p..., l_frozen, l_not_frozen, l_outer)
        U = run_wannier_minimization(p_window, U_proj_only, obj_spread; verbose=false, g_tol=1e-3)

        for ik in 1:p.nktot
            l_frozen_ik = p_window.l_frozen[ik]
            l_outer_ik = p_window.l_outer[ik]
            @test U[:, :, ik]' * U[:, :, ik] ≈ I(p.nwannier)
            @test (U[:, :, ik] * U[:, :, ik]')[l_frozen_ik, l_frozen_ik] ≈ I(count(l_frozen_ik))
            @test all(U[.!l_outer_ik, :, ik] .≈ 0)
        end

        compute_objective(U, obj_spread).spreads, U
    end

    # Test frozen window
    l_frozen = [eig[:, ik] .< dis_froz_max for ik in 1:p.nktot]
    l_outer = fill(trues(nband), nktot)
    spreads_froz, _ = _test_window(l_frozen, l_outer)

    # Test frozen and outer window
    l_frozen = [eig[:, ik] .< dis_froz_max for ik in 1:p.nktot]
    l_outer = [eig[:, ik] .< dis_win_max for ik in 1:p.nktot]
    spreads_both, _ = _test_window(l_frozen, l_outer)
    @test spreads_both.Ω >= spreads_froz.Ω

    # Test window that is not monotonic in energy
    # Loosen some constraints (remove some states from frozen window, add some states to
    # outer window) and test whether those states are used in U.
    l_frozen = [eig[:, ik] .< dis_froz_max for ik in 1:p.nktot]
    l_outer = [eig[:, ik] .< dis_win_max for ik in 1:p.nktot]
    l_frozen[3][2] = false
    l_frozen[7][3] = false
    l_outer[14][13] = true
    l_outer[18][12:14] .= true
    _, U = _test_window(l_frozen, l_outer)
    @test abs((U[:, :, 3] * U[:, :, 3]')[2, 2]) < 0.999
    @test abs((U[:, :, 7] * U[:, :, 7]')[3, 3]) < 0.999
    @test norm(U[13, :, 14]) > 0.01
    @test norm(U[12, :, 18]) > 0.01
    @test norm(U[13, :, 18]) > 0.01
    @test norm(U[14, :, 18]) > 0.01
end



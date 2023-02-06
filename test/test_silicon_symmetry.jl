using Test
using LinearAlgebra
using StaticArrays
using WannierFunctions

@testset "silicon dmn" begin
    using WannierFunctions: AtomicOrbitalBasis, get_orbital

    BASE_FOLDER = dirname(dirname(pathof(WannierFunctions)))
    seedname = joinpath(BASE_FOLDER, "test", "data", "si_8band", "si")

    win_data = WannierFunctions.read_win(seedname)
    (; nwannier, nband, lattice, ngrid, nktot, kpts) = win_data

    mmn, bvecs, neighbors = read_mmn(seedname, kpts);
    amn = read_amn(seedname);
    eig = read_eig(seedname);
    nnb = length(bvecs)
    recip_lattice = inv(lattice)' * 2π
    bvecs_cart = Ref(recip_lattice) .* bvecs
    wbs = WannierFunctions.compute_weights(bvecs_cart)
    dmn = WannierFunctions.read_dmn(seedname, nwannier);

    positions = [[0.125, 0.125, 0.125], [-0.125, -0.125, -0.125]]
    atoms = [:Si, :Si]
    symmetry = WannierFunctions.symmetry_operations(lattice, positions, atoms)

    # Test symmetry of amn
    for ikirr in 1:dmn.nkirr, isym in 1:dmn.nsymmetry
        ik = dmn.ikirr_to_ik[ikirr]
        isk = dmn.ikirr_isym_to_isk[ikirr][isym]
        D_band = dmn.d_matrix_band[:, :, isym, ikirr]
        D_wann = dmn.d_matrix_wann[:, :, isym, ikirr]
        A_k = amn[:, :, ik]
        A_Sk = amn[:, :, isk]
        @test norm(A_Sk - D_band * A_k * D_wann') < 1e-6
    end

    # Find symmetry matrix for the atomic orbitals (d_matrix_wann)
    orbitals = [
        AtomicOrbitalBasis(1, 1, get_orbital(:sp3_1, xaxis=Vec3(0, 1, 0))),
        AtomicOrbitalBasis(1, 1, get_orbital(:sp3_2, xaxis=Vec3(0, 1, 0))),
        AtomicOrbitalBasis(1, 1, get_orbital(:sp3_3, xaxis=Vec3(0, 1, 0))),
        AtomicOrbitalBasis(1, 1, get_orbital(:sp3_4, xaxis=Vec3(0, 1, 0))),
        AtomicOrbitalBasis(2, 1, get_orbital(:sp3_1, xaxis=Vec3(0, 1, 0))),
        AtomicOrbitalBasis(2, 1, get_orbital(:sp3_2, xaxis=Vec3(0, 1, 0))),
        AtomicOrbitalBasis(2, 1, get_orbital(:sp3_3, xaxis=Vec3(0, 1, 0))),
        AtomicOrbitalBasis(2, 1, get_orbital(:sp3_4, xaxis=Vec3(0, 1, 0))),
    ]

    # Calculate d_matrix_band using symmetry and amn
    dmn_calculated = construct_dmn(lattice, positions, atoms, orbitals, kpts, ngrid, amn, eig);
    symmetry = dmn_calculated.symmetry

    @test dmn.ikirr_to_ik == dmn_calculated.ikirr_to_ik
    @test dmn.ik_to_ikirr == dmn_calculated.ik_to_ikirr

    # Test for each isym, d_matrix_wann calculated here has a match in dmn.d_matrix_wann
    isym_map = zeros(Int, length(symmetry))
    for isym in eachindex(symmetry)
        for jsym in 1:dmn.nsymmetry
            if norm(dmn_calculated.d_matrix_wann[:, :, isym, :] - dmn.d_matrix_wann[:, :, jsym, :]) < 1e-10
                isym_map[isym] = jsym
            end
        end
    end
    @test all(isym_map .> 0)

    for ikirr in 1:dmn.nkirr
        rng = dmn_calculated.projectability[dmn.ikirr_to_ik[ikirr]]
        @test norm(dmn.d_matrix_band[rng, rng, isym_map, ikirr] - dmn_calculated.d_matrix_band[rng, rng, :, ikirr]) < 1e-6
    end

    l_outer = dmn_calculated.projectability
end

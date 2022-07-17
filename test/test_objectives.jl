using Test
using WannierFunctions
using StaticArrays
using LinearAlgebra

@testset "objectives" begin
    using FiniteDifferences

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

    obj_spread = MarzariVanderbiltObjective(; nband, nwannier, nktot, nnb, neighbors, wbs, bvecs_cart, mmn)
    obj_symmetry = SymmetryConstraintObjective(seedname, nwannier)
    obj_augmentation = AugmentationObjective(obj_symmetry)

    @testset "CompositeWannierObjective" begin
        obj1 = obj_spread
        obj2 = obj_symmetry

        U = zeros(ComplexF64, nband, nwannier, nktot)
        for ik in 1:nktot
            u1, _ = qr(rand(ComplexF64, nband, nband))
            u2, _ = qr(rand(ComplexF64, nwannier, nwannier))
            U[:, :, ik] .= u1 * vcat(I(nwannier), zeros(nband - nwannier, nwannier)) * u2
        end

        @inferred 1. * obj1 + obj2 * 2 * 0.3

        obj_composite = 1. * obj1 + obj2 * 2 * 0.3
        @test obj_composite isa WannierFunctions.CompositeWannierObjective
        @test length(obj_composite.objs) == 2
        @test length(obj_composite.coeffs) == 2
        @test obj_composite.objs[1] === obj1
        @test obj_composite.objs[2] === obj2
        @test obj_composite.coeffs ≈ [1., 0.6]

        g = zero(U)
        g1 = zero(U)
        g2 = zero(U)
        x1 = compute_objective_and_gradient!(g1, U, obj1).objective
        x2 = compute_objective_and_gradient!(g2, U, obj2).objective
        x = compute_objective_and_gradient!(g, U, obj_composite).objective

        @test x ≈ x1 + x2 * 0.6
        @test g ≈ g1 .+ g2 .* 0.6
    end

    # Test correctness of gradient
    @testset "gradient" begin
        objs_to_test = [obj_spread, obj_symmetry, obj_augmentation, obj_spread * 2. + obj_symmetry * 3. + obj_augmentation * 4.]

        # Initialize random unitary matrix U0 and perturbation matrix ΔU
        U0 = zeros(ComplexF64, nband, nwannier, nktot)
        for ik in 1:nktot
            u1, _ = qr(rand(ComplexF64, nband, nband))
            u2, _ = qr(rand(ComplexF64, nwannier, nwannier))
            U0[:, :, ik] .= u1 * vcat(I(nwannier), zeros(nband - nwannier, nwannier)) * u2
        end
        ΔU = rand(ComplexF64, nband, nwannier, nktot)

        for obj in objs_to_test
            # Compute derivative of objective from the gradient
            gradient = compute_objective_and_gradient!(zero(U0), U0, obj).gradient
            ΔΩ_grad = real(sum(conj(gradient) .* ΔU))

            # Compute derivative of objective by finite difference
            compute_Ω(x) = compute_objective(U0 .+ x .* ΔU, obj).objective
            ΔΩ_finitediff = central_fdm(5, 1)(compute_Ω, 0)

            @test ΔΩ_finitediff ≈ ΔΩ_grad
        end
    end
end
using Test
using LinearAlgebra
using WannierFunctions

@testset "compute_weights" begin
    function test_weights(bvecs_cart)
        wbs = WannierFunctions.compute_weights(bvecs_cart)
        @test sum(wb * b * b' for (b, wb) in zip(bvecs_cart, wbs)) ≈ I(3)

        # Check b vectors in the same shell have same weights
        nb = length(bvecs_cart)
        inds_same_shell = [I.I for I in CartesianIndices((nb, nb))
            if abs(norm(bvecs_cart[I.I[1]]) - norm(bvecs_cart[I.I[2]])) < 1e-5]
        @test all([wbs[i] ≈ wbs[j] for (i, j) in inds_same_shell])
    end

    a, b, c = rand(3)
    bvecs_cart = [Vec3(a, 0, 0), Vec3(0, b, 0), Vec3(0, 0, c)]
    append!(bvecs_cart, .-bvecs_cart)
    test_weights(bvecs_cart)

    a, b, c = rand(3)
    bvecs_cart = [Vec3(0, 0, a), Vec3(b * √3, -b, c), Vec3(-b * √3, -b, c), Vec3(0, 2b, c)]
    append!(bvecs_cart, .-bvecs_cart)
    test_weights(bvecs_cart)
end

using WannierFunctions
using Test

@time @testset "WannierFunctions.jl" begin
    include("test_silicon_valence.jl")
end

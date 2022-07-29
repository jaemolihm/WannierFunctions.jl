using WannierFunctions
using Test

@time @testset "WannierFunctions.jl" begin
    include("test_compute_weights.jl")
    include("test_objectives.jl")
    include("test_silicon_valence.jl")
    include("test_Cu.jl")
    include("test_MarzariVanderbilt.jl")
end

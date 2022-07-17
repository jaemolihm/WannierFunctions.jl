export compute_objective
export compute_objective_and_gradient!

abstract type AbstractWannierObjective end

function compute_objective(U, obj::AbstractWannierObjective)
    compute_objective_and_gradient!(nothing, U, obj)
end

function compute_objective_and_gradient!(gradient, U, obj::AbstractWannierObjective)
    error("compute_objective_and_gradient! for $(typeof(obj)) not implemented")
    # Concrete subtypes should implement this method
end
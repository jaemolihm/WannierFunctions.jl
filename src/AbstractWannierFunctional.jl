export compute_objective
export compute_objective_and_gradient!

abstract type AbstractWannierFunctional end

function compute_objective(U, obj::AbstractWannierFunctional)
    compute_objective_and_gradient!(nothing, U, obj)
end

function compute_objective_and_gradient!(gradient, U, obj::AbstractWannierFunctional)
    error("compute_objective_and_gradient! for $(typeof(obj)) not implemented")
    # Concrete subtypes should implement this method
end
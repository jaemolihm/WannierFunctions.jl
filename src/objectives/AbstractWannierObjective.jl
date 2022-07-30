export compute_objective
export compute_objective_and_gradient!

abstract type AbstractWannierObjective end

"compute_objective(U, obj::AbstractWannierObjective, factor=1)"
function compute_objective(U, obj::AbstractWannierObjective, factor=1)
    compute_objective_and_gradient!(nothing, U, obj, factor)
end

"""
The gradient is defined to satisfy
``objective(U + δU) = objective(U) + real(sum(gradient .* δU)) + O(δU^2)``.
(See Damle, Levitt, Lin (2019))
"""
function compute_objective_and_gradient!(gradient, U, obj::AbstractWannierObjective, factor=1)
    error("compute_objective_and_gradient! for $(typeof(obj)) not implemented")
    # Concrete subtypes should implement this method
end

objs(obj::AbstractWannierObjective) = [obj]
coeffs(obj::AbstractWannierObjective) = [1.]

"""
Linear combination of multiple objectives
"""
struct CompositeWannierObjective <: AbstractWannierObjective
    objs::Vector{AbstractWannierObjective}
    coeffs::Vector{Float64}
end

function Base.show(io::IO, obj::CompositeWannierObjective)
    print(io, "CompositeWannierObjective with $(length(obj.objs)) objectives:")
    for (i, (obj_item, coeff)) in enumerate(zip(obj.objs, obj.coeffs))
        print(io, "\n$i. ")
        print(io, obj_item)
        print(io, " with coefficient $coeff")
    end
end

objs(obj::CompositeWannierObjective) = obj.objs
coeffs(obj::CompositeWannierObjective) = obj.coeffs

function compute_objective_and_gradient!(gradient, U, obj::CompositeWannierObjective, factor=1)
    objective = zero(real(eltype(U)))
    for (obj_item, coeff) in zip(obj.objs, obj.coeffs)
        objective += compute_objective_and_gradient!(gradient, U, obj_item, factor * coeff).objective::Float64
    end
    gradient !== nothing ? (; objective, gradient) : (; objective)
end

function Base.:*(x::Number, obj::AbstractWannierObjective)
    CompositeWannierObjective(objs(obj), coeffs(obj) .* x)
end

function Base.:*(obj::AbstractWannierObjective, x::Number)
    CompositeWannierObjective(objs(obj), coeffs(obj) .* x)
end

function Base.:+(obj1::AbstractWannierObjective, obj2::AbstractWannierObjective)
    CompositeWannierObjective(vcat(objs(obj1), objs(obj2)), vcat(coeffs(obj1), coeffs(obj2)))
end

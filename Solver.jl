module Solver

using JuMP
using Gurobi
using LinearAlgebra


export randomsubsetsum


const GRB_ENV = Gurobi.Env()

function randomsubsetsum(z; ε::Real, n::Real, minsecs::Real = Inf, verbose::Bool = false)
    @assert 0 < ε
    @assert n > 0
    @assert minsecs ≥ 0

    n = ceil(Int, n)
    # The random subset
    A = 2rand(1, n) .- 1
    if verbose
        @show ε n z
        flush(stdout)
    end

    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "Seed", 4231)
    verbose || set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, x[1:n], Bin)
    @variable(model, t)
    @objective(
        model,
        Min,
        t
    )
    @constraint(model, c1, (z .- (A * x)) .≤ t)
    @constraint(model, c2, -(z .- (A * x)) .≤ t)
    @constraint(model, c3, t ≤ ε)
    # print(model)

    # Search for `minsecs` seconds (keep going even if a solution is found)
    if minsecs > 0
        set_optimizer_attribute(model, "TimeLimit", minsecs)
        optimize!(model)
    end
    # If no solution has been found, keep searching for as long as necessary
    if !has_values(model)
        set_optimizer_attribute(model, "TimeLimit", Inf)
        set_optimizer_attribute(model, "SolutionLimit", 1)
        optimize!(model)
    end
    # If still no solution has been found, restart without optimizations that could have discarded feasable solutions
    if termination_status(model) == MOI.INFEASIBLE_OR_UNBOUNDED
        @info "Needs restart without `DualReductions`"
        set_optimizer_attribute(model, "DualReductions", 0)
        optimize!(model)
    end

    # If a solution is possible, it should be available by now
    if has_values(model)
        result = sum(value.(x) .* eachcol(A))
        verbose && @show(sum(value.(x)), maximum(abs, z .- result), z, result)
        return convert(eltype(z), first(result))
    end
    # At this point we can trust the infeasability
    if termination_status(model) == MOI.INFEASIBLE
        @warn "There is no solution for this instance. Termination status: $(raw_status(model))"
        @show ε n z A
    else
        @error "Something went wrong with the solver. Termination status: $(raw_status(model))"
    end
    @info "Returning zeros"
    flush(stdout); flush(stderr)
    return zero(z)
end

end

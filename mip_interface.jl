using MIPVerify
using JuMP
using ConditionalJuMP
import MathProgBase
using Memento

function get_model(
    input::Array{<:Real},
    reusable_model::Dict,
    set_input::Bool,
)::Dict
    d = reusable_model
    if set_input
        @constraint(d["Model"], d["Input"] .== input)
        # TODO: Riattivare?
        # delete!(d, :Input)
    end
    # NOTE (vtjeng): It is important to set the solver before attempting to add a
    # constraint, as the saved model may have been saved with a different solver (or
    # different) environment. Flipping the order of the two leads to test failures.
    return d
end

function get_reusable_model(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::MIPVerify.PerturbationFamily,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver,
    tightening_algorithm::MIPVerify.TighteningAlgorithm,
)::Dict
    d = MIPVerify.build_reusable_model_uncached(nn, input, pp, tightening_solver, tightening_algorithm)
    setsolver(d[:Model], tightening_solver)
    return d
end

function run_attack(
    nn::NeuralNet,
    input::Array{<:Real},
    target_selection::Union{Integer,Array{<:Integer,1}},
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver;
    invert_target_selection::Bool = false,
    pp::MIPVerify.PerturbationFamily = MIPVerify.UnrestrictedPerturbationFamily(),
    norm_order::Real = 1,
    tolerance::Real = 0.0,
    adversarial_example_objective::MIPVerify.AdversarialExampleObjective = MIPVerify.closest,
    tightening_algorithm::MIPVerify.TighteningAlgorithm = MIPVerify.DEFAULT_TIGHTENING_ALGORITHM,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver = MIPVerify.get_default_tightening_solver(
        main_solver,
    ),
    reusable_model::Dict,
    solve_if_predicted_in_targeted = true,
)::Dict

    total_time = @elapsed begin
        d = Dict()

        # Calculate predicted index
        predicted_output = input |> nn
        num_possible_indexes = length(predicted_output)
        predicted_index = predicted_output |> MIPVerify.get_max_index

        d[:PredictedIndex] = predicted_index

        # Set target indexes
        d[:TargetIndexes] = MIPVerify.get_target_indexes(
            target_selection,
            num_possible_indexes,
            invert_target_selection = invert_target_selection,
        )
        notice(
            MIPVerify.LOGGER,
            "Attempting to find adversarial example. Neural net predicted label is $(predicted_index), target labels are $(d[:TargetIndexes])",
        )

        # Only call solver if predicted index is not found among target indexes.
        if !(d[:PredictedIndex] in d[:TargetIndexes]) || solve_if_predicted_in_targeted
            merge!(
                d,
                get_model(
                    input,
                    reusable_model,
                    isa(pp, MIPVerify.UnrestrictedPerturbationFamily),
                ),
            )
            m = d["Model"]

            if adversarial_example_objective == MIPVerify.closest
                MIPVerify.set_max_indexes(m, d["Output"], d[:TargetIndexes], tolerance = tolerance)

                # Set perturbation objective
                # NOTE (vtjeng): It is important to set the objective immediately before we carry out
                # the solve. Functions like `set_max_indexes` can modify the objective.
                @objective(m, Min, MIPVerify.get_norm(norm_order, d["Perturbation"]))
            elseif adversarial_example_objective == worst
                (maximum_target_var, nontarget_vars) =
                MIPVerify.get_vars_for_max_index(d["Output"], d[:TargetIndexes])
                maximum_nontarget_var = MIPVerify.maximum_ge(nontarget_vars)
                # Introduce an additional variable since Gurobi ignores constant terms in objective, 
                # but we explicitly need these if we want to stop early based on the value of the objective
                # (not simply whether or not it is maximized).
                # See discussion in https://github.com/jump-dev/Gurobi.jl/issues/111 for more details.
                v_obj = @variable(m)
                @constraint(m, v_obj == maximum_target_var - maximum_nontarget_var)
                @objective(m, Max, v_obj)
            else
                error("Unknown adversarial_example_objective $adversarial_example_objective")
            end
            MIPVerify.setsolver(m, main_solver)
            solve_time = @elapsed begin
                d[:SolveStatus] = solve(m)
            end
            d[:SolveTime] = try
                MIPVerify.getsolvetime(m)
            catch err
                # CBC solver, used for testing, does not implement `getsolvetime`.
                isa(err, MethodError) || rethrow(err)
                solve_time
            end
        end
    end

    d[:TotalTime] = total_time
    return d
end

function find_adversarial_example(
    nn::NeuralNet,
    input::Array{<:Real},
    target_selection::Union{Integer,Array{<:Integer,1}},
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver;
    invert_target_selection::Bool = false,
    pp::MIPVerify.PerturbationFamily = MIPVerify.UnrestrictedPerturbationFamily(),
    norm_order::Real = 1,
    tolerance::Real = 0.0,
    adversarial_example_objective::MIPVerify.AdversarialExampleObjective = MIPVerify.closest,
    tightening_algorithm::MIPVerify.TighteningAlgorithm = MIPVerify.DEFAULT_TIGHTENING_ALGORITHM,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver = MIPVerify.get_default_tightening_solver(
        main_solver,
    ),
    rebuild::Bool = false,
    cache_model::Bool = true,
    solve_if_predicted_in_targeted = true,
    starting_point::Union{Array{<:Real}, Nothing} = nothing
)::Dict

    total_time = @elapsed begin
        d = Dict()

        # Calculate predicted index
        predicted_output = input |> nn
        num_possible_indexes = length(predicted_output)
        predicted_index = predicted_output |> MIPVerify.get_max_index

        d[:PredictedIndex] = predicted_index

        # Set target indexes
        d[:TargetIndexes] = MIPVerify.get_target_indexes(
            target_selection,
            num_possible_indexes,
            invert_target_selection = invert_target_selection,
        )
        notice(
            MIPVerify.LOGGER,
            "Attempting to find adversarial example. Neural net predicted label is $(predicted_index), target labels are $(d[:TargetIndexes])",
        )

        # Only call solver if predicted index is not found among target indexes.
        if !(d[:PredictedIndex] in d[:TargetIndexes]) || solve_if_predicted_in_targeted
            merge!(
                d,
                MIPVerify.get_model(
                    nn,
                    input,
                    pp,
                    tightening_solver,
                    tightening_algorithm,
                    rebuild,
                    cache_model,
                ),
            )
            m = d[:Model]
            
            for i in CartesianIndices(size(input))
                JuMP.setvalue(d[:PerturbedInput][i], starting_point[i])
                JuMP.setvalue(d[:Perturbation][i], starting_point[i] - input[i])
            end
            # TODO: è necessario?
            d[:Output] = d[:PerturbedInput] |> nn

            if adversarial_example_objective == MIPVerify.closest
                MIPVerify.set_max_indexes(m, d[:Output], d[:TargetIndexes], tolerance = tolerance)

                # Set perturbation objective
                # NOTE (vtjeng): It is important to set the objective immediately before we carry out
                # the solve. Functions like `set_max_indexes` can modify the objective.
                @objective(m, Min, MIPVerify.get_norm(norm_order, d[:Perturbation]))
            elseif adversarial_example_objective == worst
                (maximum_target_var, nontarget_vars) =
                    MIPVerify.get_vars_for_max_index(d[:Output], d[:TargetIndexes])
                maximum_nontarget_var = MIPVerify.maximum_ge(nontarget_vars)
                # Introduce an additional variable since Gurobi ignores constant terms in objective, 
                # but we explicitly need these if we want to stop early based on the value of the objective
                # (not simply whether or not it is maximized).
                # See discussion in https://github.com/jump-dev/Gurobi.jl/issues/111 for more details.
                v_obj = @variable(m)
                @constraint(m, v_obj == maximum_target_var - maximum_nontarget_var)
                @objective(m, Max, v_obj)
            else
                error("Unknown adversarial_example_objective $adversarial_example_objective")
            end
            MIPVerify.setsolver(m, main_solver)

            solve_time = @elapsed begin
                d[:SolveStatus] = solve(m)
            end
            d[:SolveTime] = try
                MIPVerify.getsolvetime(m)
            catch err
                # CBC solver, used for testing, does not implement `getsolvetime`.
                isa(err, MethodError) || rethrow(err)
                solve_time
            end
        end
    end

    d[:TotalTime] = total_time
    return d
end
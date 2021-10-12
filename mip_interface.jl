using MIPVerify
using JuMP
using ConditionalJuMP
import MathProgBase
using Memento
import Gurobi

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
)::Dict

    find_adversarial_example_start_timestamp = time()
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

        model_building_start_timestamp = time()

        if adversarial_example_objective == MIPVerify.closest
            MIPVerify.set_max_indexes(m, d[:Output], d[:TargetIndexes], tolerance = tolerance)

            # Set perturbation objective
            # NOTE (vtjeng): It is important to set the objective immediately before we carry out
            # the solve. Functions like `set_max_indexes` can modify the objective.
            @objective(m, Min, MIPVerify.get_norm(norm_order, d[:Perturbation]))
        elseif adversarial_example_objective == MIPVerify.worst
            (maximum_target_var, nontarget_vars) =
                MIPVerify.get_vars_for_max_index(d[:Output], d[:TargetIndexes])
            maximum_nontarget_var = MIPVerify.maximum_ge(nontarget_vars)
            @objective(m, Max, maximum_target_var - maximum_nontarget_var)
        else
            error("Unknown adversarial_example_objective $adversarial_example_objective")
        end

        model_building_end_timestamp = time()

        d[:ModelBuildingStartTimestamp] = model_building_start_timestamp
        d[:ModelBuildingEndTimestamp] = model_building_end_timestamp

        MIPVerify.setsolver(m, main_solver)

        wall_clock_solve_start_timestamp = time()
        d[:SolveStatus] = solve(m)
        wall_clock_solve_end_timestamp = time()

        d[:WallClockStartTimestamp] = wall_clock_solve_start_timestamp
        d[:WallClockEndTimestamp] = wall_clock_solve_end_timestamp

        d[:GurobiSolveTime] = try
            MIPVerify.getsolvetime(m)
        catch err
            # CBC solver, used for testing, does not implement `getsolvetime`.
            isa(err, MethodError) || rethrow(err)
            solve_time
        end
    end

    find_adversarial_example_end_timestamp = time()

    d[:FindAdversarialExampleStartTimestamp] = find_adversarial_example_start_timestamp
    d[:FindAdversarialExampleEndTimestamp] = find_adversarial_example_end_timestamp
    return d
end

function get_intattrarray_workaround(model::Gurobi.Model, name::String, start::Integer, len::Integer)
    @assert isascii(name)
    Gurobi.get_intattrarray!(Array{Cint}(undef, len), model, name, start)
end
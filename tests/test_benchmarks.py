from functools import partial

import numpy as np
import pytest

from iwopy.benchmarks import MixedProblem, ParetoProblem, QuadraticProblem


@pytest.mark.parametrize(
    "factory,ints,floats,expected_objs,expected_cons",
    [
        (QuadraticProblem, [], [0.25, 0.25], [0.0], []),
        (ParetoProblem, [], [-0.25, -0.25], [0.5, 0.5], []),
        (
            partial(ParetoProblem, constrained=True),
            [],
            [-0.25, -0.25],
            [0.5, 0.5],
            [-0.5],
        ),
        (MixedProblem, [0], [0.25], [0.0625], []),
    ],
)
def test_public_quadratic_benchmarks_match_individual_and_population_values(
    factory, ints, floats, expected_objs, expected_cons, capsys
):
    problem = factory()
    assert problem.initialized
    assert capsys.readouterr().out == ""
    vars_int = np.array(ints, dtype=np.int32)
    vars_float = np.array(floats)
    objs, cons = problem.evaluate_individual(vars_int, vars_float)
    np.testing.assert_allclose(objs, expected_objs)
    np.testing.assert_allclose(cons, expected_cons)
    population_ints = np.repeat(vars_int[None, :], 2, axis=0)
    population_floats = np.stack([vars_float, vars_float + 0.25])
    population_objs, population_cons = problem.evaluate_population(
        population_ints, population_floats
    )
    for index in range(2):
        row_objs, row_cons = problem.evaluate_individual(
            population_ints[index], population_floats[index]
        )
        np.testing.assert_allclose(population_objs[index], row_objs)
        np.testing.assert_allclose(population_cons[index], row_cons)


def test_constrained_pareto_boundary_and_infeasible_initial_point():
    problem = ParetoProblem(constrained=True)
    empty_ints = np.empty(0, dtype=np.int32)
    _, boundary = problem.evaluate_individual(empty_ints, np.array([-0.125, -0.125]))
    _, infeasible = problem.evaluate_individual(empty_ints, np.array([0.0, 0.0]))
    assert np.all(problem.check_constraints_individual(boundary, verbosity=0))
    assert not np.any(problem.check_constraints_individual(infeasible, verbosity=0))
    assert problem.cons.functions[0].tol == 0.0


def test_mixed_quadratic_jacobian_uses_float_after_integer_variables():
    problem = MixedProblem()
    jacobian = problem.get_jacobian(np.array([1], dtype=np.int32), np.array([0.5]))
    np.testing.assert_allclose(jacobian.toarray(), [[0.5]])


def test_initialized_quadratic_rejects_additional_objectives():
    problem = QuadraticProblem()
    with pytest.raises(ValueError, match="after initialization"):
        problem.add_objective(problem.objs.functions[0])

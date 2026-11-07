import numpy as np
import pytest

from udt.intervention import ReturnBudget, propose_actions, ensemble_scores


def test_clipping_does_not_change_raw_budget_recurrence():
    budget = ReturnBudget(raw=10., reward_scale=2.)
    assert budget.token(cap=4.) == 4.
    budget.observe_reward(6.)
    assert budget.raw == 7.
    assert budget.token() == 7.
    assert budget.token(cap=5.) == 5.


def test_candidates_include_original_and_respect_asymmetric_action_bounds():
    action = np.array([0.5, -1.])
    candidates = propose_actions(action, low=np.array([0., -2.]),
        high=np.array([1., 0.]), count=20, radius=2., rng=np.random.default_rng(10))
    np.testing.assert_array_equal(candidates[0], action)
    assert np.all(candidates >= [0., -2.]) and np.all(candidates <= [1., 0.])
    np.testing.assert_array_equal(propose_actions(action, low=np.array([0., -2.]),
        high=np.array([1., 0.]), count=1, radius=.05,
        rng=np.random.default_rng(10)), action[None])


def test_terminal_and_horizon_zero_continuation_and_ties_keep_original():
    states = np.zeros((2, 2, 1))
    scores = ensemble_scores(rewards=np.ones((2, 2)),
        next_returns=np.array([[100., 3.], [100., 5.]]),
        terminal_probabilities=np.array([[1., 0.], [1., 0.]]),
        next_states=states, beta=0., at_horizon=False)
    np.testing.assert_allclose(scores, [1., 5.])
    scores = ensemble_scores(rewards=np.ones((2, 2)),
        next_returns=np.full((2, 2), 100.), terminal_probabilities=np.zeros((2, 2)),
        next_states=states, beta=0., at_horizon=True)
    np.testing.assert_allclose(scores, [1., 1.])
    assert np.argmax(scores) == 0


def test_disagreement_penalizes_divergent_state_predictions():
    scores = ensemble_scores(rewards=np.zeros((2, 2)),
        next_returns=np.ones((2, 2)), terminal_probabilities=np.zeros((2, 2)),
        next_states=np.array([[[0.], [-2.]], [[0.], [2.]]]),
        beta=.5, at_horizon=False)
    np.testing.assert_allclose(scores, [1., 0.])


def test_invalid_probabilities_are_rejected():
    with pytest.raises(ValueError, match='probabilities'):
        ensemble_scores(rewards=np.zeros((1, 1)), next_returns=np.zeros((1, 1)),
            terminal_probabilities=np.array([[1.1]]), next_states=np.zeros((1, 1, 1)),
            beta=0., at_horizon=False)

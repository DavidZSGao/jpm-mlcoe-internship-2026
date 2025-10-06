"""Tests for differentiable particle filter with OT resampling."""

from __future__ import annotations

import tensorflow as tf

from mlcoe_q2.datasets import NonlinearStateSpaceModel
from mlcoe_q2.filters import differentiable_particle_filter


def _build_benchmark_model() -> NonlinearStateSpaceModel:
    process_cov = tf.constant([[0.05, 0.0], [0.0, 0.03]], dtype=tf.float32)
    observation_cov = tf.constant([[0.2]], dtype=tf.float32)

    def transition_fn(state: tf.Tensor, control: tf.Tensor | None) -> tf.Tensor:
        del control
        x0, x1 = state[0], state[1]
        new_x0 = 0.85 * x0 + 0.2 * tf.math.sin(x1)
        new_x1 = 0.9 * x1 + 0.15 * tf.math.tanh(x0)
        return tf.stack([new_x0, new_x1])

    def observation_fn(state: tf.Tensor, control: tf.Tensor | None) -> tf.Tensor:
        del control
        return tf.stack([0.6 * tf.math.sin(state[0]) + 0.1 * state[1]])

    return NonlinearStateSpaceModel(
        state_dim=2,
        observation_dim=1,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        process_noise_cov=process_cov,
        observation_noise_cov=observation_cov,
    )


def test_transport_plan_rows_sum_to_one() -> None:
    tf.random.set_seed(0)
    model = _build_benchmark_model()

    observations = tf.constant([[0.1], [-0.05]], dtype=tf.float32)
    initial_particles = tf.random.normal((32, model.state_dim), dtype=tf.float32)

    result = differentiable_particle_filter(
        model=model,
        observations=observations,
        num_particles=32,
        initial_particles=initial_particles,
        mix_with_uniform=0.2,
        ot_epsilon=0.5,
        ot_num_iters=10,
        epsilon_schedule=[0.5, 0.4],
        sinkhorn_tolerance=1e-4,
    )

    transport = result.transport_plans.numpy()
    assert transport.shape == (observations.shape[0], 32, 32)
    row_sums = transport.sum(axis=-1)
    assert ((row_sums > 0.999) & (row_sums < 1.001)).all()
    assert (transport >= 0.0).all()


def test_outputs_are_finite_and_weights_normalized() -> None:
    tf.random.set_seed(1)
    model = _build_benchmark_model()

    observations = tf.constant([[0.2], [0.05], [-0.1]], dtype=tf.float32)
    initial_particles = tf.random.normal((24, model.state_dim), dtype=tf.float32)

    result = differentiable_particle_filter(
        model=model,
        observations=observations,
        num_particles=24,
        initial_particles=initial_particles,
        mix_with_uniform=0.1,
        ot_epsilon=0.3,
        ot_num_iters=15,
        sinkhorn_tolerance=0.0,
    )

    weights = result.weights[-1]
    log_weights = result.log_weights[-1]

    assert tf.reduce_all(tf.math.is_finite(weights))
    assert tf.reduce_all(tf.math.is_finite(log_weights))
    assert tf.reduce_all(tf.math.is_finite(result.log_likelihood))

    weight_sum = tf.reduce_sum(weights)
    assert tf.abs(weight_sum - 1.0) < 1e-5

    ess = result.diagnostics["ess"].numpy()
    assert (ess > 0.0).all()

    mix_weight = result.diagnostics["mix_weight"].numpy()
    assert (mix_weight >= 0.0).all()
    assert (mix_weight <= 1.0).all()

    epsilon = result.diagnostics["epsilon"].numpy()
    assert epsilon.shape[0] == observations.shape[0]
    assert (epsilon > 0.0).all()



def test_differentiable_pf_gradients_match_finite_differences() -> None:
    tf.random.set_seed(4)
    model = _build_benchmark_model()

    observations = tf.constant([[0.1], [-0.2], [0.05]], dtype=tf.float32)
    initial_particles = tf.random.normal((8, model.state_dim), dtype=tf.float32)

    def loss_fn(particles: tf.Tensor) -> tf.Tensor:
        tf.random.set_seed(1234)
        result = differentiable_particle_filter(
            model=model,
            observations=observations,
            num_particles=8,
            initial_particles=particles,
            mix_with_uniform=0.05,
            ot_epsilon=0.4,
            ot_num_iters=8,
            sinkhorn_tolerance=0.0,
        )
        return result.log_likelihood

    analytical, numerical = tf.test.compute_gradient(loss_fn, [initial_particles])
    analytic_grad = analytical[0]
    numeric_grad = numerical[0]
    max_err = tf.reduce_max(tf.abs(analytic_grad - numeric_grad))
    assert max_err < 5e-2

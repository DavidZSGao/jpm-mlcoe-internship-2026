import json

import numpy as np
import tensorflow as tf

import mlcoe_q2.pipelines.neural_state_space_inference as neural_state_space_inference
from mlcoe_q2.data.nonlinear_ssm import NonlinearStateSpaceModel
from mlcoe_q2.models.filters.differentiable_pf import differentiable_particle_filter
from mlcoe_q2.models.inference.dpf_hmc import DPFHMCConfig, run_dpf_hmc
from mlcoe_q2.models.neural_ssm import NeuralLSTMStateSpace
from mlcoe_q2.models.resampling.neural_ot import (
    NeuralOTConfig,
    NeuralOTResampler,
    generate_ot_training_data,
)


def test_generate_ot_training_data_shapes():
    config = NeuralOTConfig(
        num_particles=4,
        state_dim=2,
        num_samples=12,
        hidden_units=(8,),
        sinkhorn_iters=5,
        epochs=1,
    )
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = generate_ot_training_data(config)
    assert x_train.shape[1] == config.num_particles ** 2 + config.num_particles + 1
    assert y_train.shape[1] == config.num_particles ** 2
    assert x_val.shape[0] + x_test.shape[0] + x_train.shape[0] == config.num_samples


def test_neural_ot_resampler_normalises_rows():
    num_particles = 3
    state_dim = 2
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(num_particles ** 2 + num_particles + 1,), dtype=tf.float32
            ),
            tf.keras.layers.Dense(num_particles ** 2, activation="relu"),
        ]
    )
    resampler = NeuralOTResampler(model, num_particles=num_particles, state_dim=state_dim)
    weights = tf.fill((num_particles,), 1.0 / num_particles)
    particles = tf.random.normal((num_particles, state_dim))
    transport, error = resampler.predict_transport(weights, particles, tf.constant(0.1))
    row_sums = tf.reduce_sum(transport, axis=1)
    tf.debugging.assert_near(row_sums, tf.ones_like(row_sums))
    assert transport.shape == (num_particles, num_particles)
    assert error is None or error.numpy() >= 0.0


def test_neural_lstm_state_space_simulation_runs():
    builder = NeuralLSTMStateSpace(latent_dim=2, observation_dim=1)
    model = builder.build_model(process_scale=0.3, observation_scale=0.2)
    initial_state = builder.initial_state()
    states, observations = model.simulate(5, initial_state)
    assert states.shape == (5, model.state_dim)
    assert observations.shape == (5, model.observation_dim)


def test_differentiable_pf_accepts_neural_resampler():
    state_dim = 2
    obs_dim = 1

    def transition(state, control):
        del control
        return 0.9 * state

    def observation(state, control):
        del control
        return state[:obs_dim]

    process_cov = tf.eye(state_dim) * 0.2
    obs_cov = tf.eye(obs_dim) * 0.1
    model = NonlinearStateSpaceModel(
        state_dim=state_dim,
        observation_dim=obs_dim,
        transition_fn=transition,
        observation_fn=observation,
        process_noise_cov=process_cov,
        observation_noise_cov=obs_cov,
    )

    num_particles = 3
    dummy_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(num_particles ** 2 + num_particles + 1,), dtype=tf.float32
            ),
            tf.keras.layers.Dense(num_particles ** 2, activation="relu"),
        ]
    )
    neural_resampler = NeuralOTResampler(dummy_model, num_particles=num_particles, state_dim=state_dim)

    observations = tf.random.normal((4, obs_dim))
    initial_particles = tf.random.normal((num_particles, state_dim))
    result = differentiable_particle_filter(
        model,
        observations,
        num_particles=num_particles,
        initial_particles=initial_particles,
        resampling_method="neural_ot",
        neural_resampler=neural_resampler,
    )
    assert result.particles.shape[0] == observations.shape[0] + 1


def test_run_dpf_hmc_produces_valid_chain():
    tf.random.set_seed(0)

    def build_model(theta: tf.Tensor) -> NonlinearStateSpaceModel:
        theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        log_proc, log_obs = tf.unstack(theta)
        process_scale = tf.exp(log_proc)
        observation_scale = tf.exp(log_obs)

        def transition(state, control):
            del control
            return 0.8 * state

        def observation_fn(state, control):
            del control
            return state[:1]

        process_cov = tf.reshape(tf.square(process_scale), (1, 1))
        obs_cov = tf.reshape(tf.square(observation_scale), (1, 1))
        return NonlinearStateSpaceModel(
            state_dim=1,
            observation_dim=1,
            transition_fn=transition,
            observation_fn=observation_fn,
            process_noise_cov=process_cov,
            observation_noise_cov=obs_cov,
        )

    true_theta = tf.constant([-0.2, -0.4], dtype=tf.float32)
    model = build_model(true_theta)
    _, observations = model.simulate(6, tf.zeros((1,), dtype=tf.float32))
    initial_particles = tf.random.normal((6, 1), dtype=tf.float32)

    config = DPFHMCConfig(
        num_results=6,
        num_burnin_steps=2,
        step_size=0.05,
        prior_scale=1.0,
        num_particles=6,
        proposal_noise=0.05,
    )

    results = run_dpf_hmc(
        build_model,
        observations,
        initial_particles,
        initial_theta=true_theta,
        config=config,
    )

    samples = results["samples"].numpy()
    assert samples.shape == (config.num_results, 2)
    ess = results["ess"].numpy()
    assert ess.shape == (2,)
    assert np.all(ess >= 0.0)
    acceptance = float(results["acceptance_rate"].numpy())
    assert 0.0 <= acceptance <= 1.0
    assert results["log_posterior"].shape[0] == config.num_results


def test_neural_state_space_pipeline_smoke(tmp_path):
    output_json = tmp_path / "artifact.json"
    status_md = tmp_path / "status.md"

    args = [
        "--num-timesteps",
        "3",
        "--latent-dim",
        "2",
        "--observation-dim",
        "1",
        "--dpf-num-particles",
        "4",
        "--pg-num-particles",
        "4",
        "--hmc-num-results",
        "4",
        "--hmc-burnin",
        "1",
        "--hmc-step-size",
        "0.04",
        "--hmc-proposal-noise",
        "0.07",
        "--pg-iterations",
        "6",
        "--pg-step-size",
        "0.07",
        "--random-seed",
        "5",
        "--output-json",
        str(output_json),
        "--status-md",
        str(status_md),
    ]

    neural_state_space_inference.main(args)

    payload = json.loads(output_json.read_text())
    assert "dpf_hmc" in payload
    assert "particle_gibbs" in payload
    summary_text = status_md.read_text()
    assert "Neural State-Space" in summary_text

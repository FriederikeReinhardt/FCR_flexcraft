import jax
import jax.numpy as jnp

import equinox as eqx
from protenix.backend import load_model
from protenix.protenij import InitialEmbedding, TrunkEmbedding, ConfidenceMetrics
from protenix.protenij import Protenix as _Protenix

from flexcraft.structure.protenix._data import ProtenixInput, ProtenixSpec
from flexcraft.structure.protenix._result import ProtenixResult, ProtenixPrediction

class Protenix:
    def __init__(self, model="protenix_base_default_v1.0.0"):
        self.model = load_model(model)

    def predictor(self, num_recycle=2, num_samples=1,
                  num_sampling_steps=25, deterministic=False):
        jit_predict = eqx.filter_jit(ProtenixEvaluator(
            model=self.model,
            num_recycle=num_recycle,
            num_sampling_steps=num_sampling_steps,
            deterministic=deterministic)._predict)
        def _predict(key, joltz_spec: ProtenixSpec):
            input, writer = joltz_spec.to_input(pad=True, cache=self.cache)
            # writer_features = {
            #     k: torch.tensor(np.array(v))
            #     for k, v in features.items() if k != "record"
            # }
            # writer_features["record"] = writer_spec["features_dict"]["record"]
            # writer_spec["features_dict"] = writer_features
            #features = jax.tree.map(jnp.array, features)
            prediction = jit_predict(key, input.features, num_samples=num_samples)
            return ProtenixPrediction(
                data=prediction.data,
                writer=writer)
        return _predict

    def evaluator(self, sampling_steps=25, sample_parallel=False,
                 num_recycle=4, stop_recycle_gradient=True):
        evaluator = ProtenixEvaluator(
            model=self.model, sampling_steps=sampling_steps,
            sample_parallel=sample_parallel,
            num_recycle=num_recycle,
            stop_recycle_gradient=stop_recycle_gradient)
        evaluator = eqx.tree_at(
            lambda m: (m.model.gamma0, m.model.step_scale_eta, m.model.noise_scale_lambda, m.model.N_steps),
            evaluator, 
            (0.0, 1.0, 1.0, sampling_steps))
        # return evaluator
        evaluator_params, evaluator_static = eqx.partition(evaluator, eqx.is_array)
        def _evaluator(params):
            return eqx.combine(params, evaluator_static)
        return _evaluator, evaluator_params

class ProtenixEvaluator(eqx.Module):
    model: _Protenix
    sampling_steps: int = 25
    num_recycle: int = 4
    sample_parallel: bool = False
    stop_recycle_gradient: bool = True

    def _predict(self, key, features, recycling_state: TrunkEmbedding | None = None, num_samples=4):
        keys = jax.random.split(key, 10)
        # run embedding
        embedding = self.embedding(keys[0], features)
        state = recycling_state
        if state is None:
            state = TrunkEmbedding(
                s=jnp.zeros_like(embedding.s_init),
                z=jnp.zeros_like(embedding.z_init))
        # run trunk
        state: TrunkEmbedding = self.trunk(keys[1], state, features, embedding)
        # run distogram
        log_distogram: jax.Array = self.model.distogram_head(state.z)
        # run diffusion
        (coords, confidence) = self.structure_module(
            keys[2], state, features, embedding,
            num_samples=num_samples,
            sampling_steps=self.sampling_steps,
            parallel=self.sample_parallel)
        return ProtenixResult(dict(
            features=features, embedding=embedding, state=state,
            distogram=log_distogram, samples=coords, confidence=confidence
        ))
    
    def predict(self, key, protenix_input: ProtenixInput,
                recycling_state: TrunkEmbedding | None = None,
                num_samples=4):
        features = protenix_input.features
        return self._predict(
            key, features, recycling_state=recycling_state,
            num_samples=num_samples)
        

    def embedding(self, key, features) -> InitialEmbedding:
        embedding = self.model.embed_inputs(input_feature_dict=features)
        return embedding

    def trunk(self, key, state: TrunkEmbedding, features, embedding: InitialEmbedding) -> TrunkEmbedding:
        def body(i, trunk_state):
            state, key = trunk_state
            state: TrunkEmbedding
            state = jax.tree.map(jax.lax.stop_gradient, state)
            s, z = state.s, state.z
            z = embedding.z_init + self.model.linear_no_bias_z_cycle(
                self.model.layernorm_z_cycle(z)
            )
            if self.model.template_embedder.n_blocks > 0:
                z = z + self.model.template_embedder(features, z, pair_mask=None, key=key)
            z = self.model.msa_module(
                features,
                z,
                embedding.s_inputs,
                pair_mask=None,
                key=key,
            )
            s = embedding.s_init + self.model.linear_no_bias_s(self.model.layernorm_s(s))
            s, z = self.model.pairformer_stack(
                s, z, pair_mask=None, key=jax.random.fold_in(key, 1)
            )
            return (TrunkEmbedding(s=s, z=z), jax.random.fold_in(key, 1))
        state, key = jax.lax.fori_loop(0, self.num_recycle - 1, body, (state, key))
        if self.stop_recycle_gradient:
            state = jax.lax.stop_gradient(state)
        return body(0, (state, key))[0]

    def structure_module(self, key, state, features, embedding,
                         num_samples=1, sampling_steps=25, parallel=False):
        def single_sample(carry, key):
            sample: jax.Array = self.model.sample_structures(
                initial_embedding=embedding,
                trunk_embedding=state,
                input_feature_dict=features,
                N_samples=1,
                N_steps=sampling_steps,
                key=key,
            )

            confidence: ConfidenceMetrics = self.model.confidence_metrics(
                initial_embedding=embedding,
                trunk_embedding=state,
                input_feature_dict=features,
                coordinates=sample,
                key=key,
            )
            return carry, (sample, confidence)

        if num_samples > 1:
            if parallel:
                vmap_sample = jax.vmap(single_sample, in_axes=(None, 0), out_axes=(None, (0, 0)))
                _, (samples, confidence) = vmap_sample(None, jax.random.split(key, num_samples))
            else:
                _, (samples, confidence) = jax.lax.scan(single_sample, None, jax.random.split(key, num_samples))
        else:
            _, (samples, confidence) = single_sample(None, key)
        return samples, confidence

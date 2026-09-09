from typing import Any

import jax
import jax.numpy as jnp

from flexcraft.structure.common._result import AF3LikeResult
from flexcraft.structure.protenix._data import ProtenixWriter

class ProtenixResult(AF3LikeResult):
    data: dict

    @property
    def residue_index(self):
        return self.data["features"]["residue_index"]

    @property
    def chain_index(self):
        return self.data["features"]["asym_id"]

    @property
    def mol_type(self):
        return jnp.argmax(jnp.stack((
            self.data["features"]["is_protein"],
            self.data["features"]["is_dna"],
            self.data["features"]["is_rna"],
            self.data["features"]["is_ligand"],
        ), axis=-1), axis=-1)[self.data["features"]["atom_rep_atom_idx"]]

    @property
    def atom_to_token(self):
        num_tokens = self.residue_index.shape[0]
        return jax.nn.one_hot(self.data["features"]["atom_to_token_idx"], num_tokens, axis=1)

    @property
    def plddt_logits(self):
        index = self.data["features"]["atom_rep_atom_idx"]
        if self.is_single_sample:
            return self.data["confidence"].plddt_logits[0, index]
        return self.data["confidence"].plddt_logits[:, 0, index]

    @property
    def plddt(self):
        return (jax.nn.softmax(self.plddt_logits, axis=-1) * (jnp.arange(50) + 0.5) / 50).sum(axis=-1)

    @property
    def pae(self):
        PAE_BINS = jnp.arange(start=0.25, stop=32.0, step=0.5)
        pae = (jax.nn.softmax(self.pae_logits, axis=-1) * PAE_BINS).sum(axis=-1)
        if self.is_single_sample:
            return jnp.fill_diagonal(pae, 0.0, inplace=False) / 32
        return jax.vmap(lambda x: jnp.fill_diagonal(x, 0.0, inplace=False))(pae) / 32

class ProtenixPrediction:
    data: Any
    writer: ProtenixWriter
    @property
    def result(self):
        return ProtenixResult(data=self.data)

    def save_pdb(self, path, sample_index=0):
        is_multisample = len(self.data["samples"].shape) == 4
        if is_multisample:
            self.writer.save_pdb(path, self.data["samples"][sample_index],
                                 plddt=self.data["confidence"].plddt[sample_index])
        else:
            self.writer.save_pdb(path, self.data["samples"],
                                 plddt=self.data["confidence"].plddt)

    def save_cif(self, path, sample_index=0):
        self.writer.save_cif(path, self.data["samples"][sample_index][None],
                             plddt=self.data["confidence"].plddt[sample_index][None])

    def save(self, path):
        self.result.save(path)

import gin
import optax

from imprl import modules
from imprl.mdp import value

gin.external_configurable(value.identity_offset)
gin.external_configurable(value.mean_offset)
gin.external_configurable(value.logsumexp_reduce)
gin.external_configurable(value.max_reduce)
gin.external_configurable(value.ValueIteration)

gin.external_configurable(modules.ExplicitMDP)
gin.external_configurable(modules.ExplicitWeights)
gin.external_configurable(modules.LinearModule)
gin.external_configurable(modules.MDPSolveWeights)

gin.external_configurable(optax.sgd)
gin.external_configurable(optax.adam)

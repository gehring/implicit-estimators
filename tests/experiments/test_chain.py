import pytest

import gin

import optax

from imprl import modules
from imprl.mdp import value

from experiments.chain.supervised import experiment as supervised_experiment

gin.enter_interactive_mode()

gin.external_configurable(value.identity_offset)
gin.external_configurable(value.mean_offset)
gin.external_configurable(value.logsumexp_reduce)
gin.external_configurable(value.max_reduce)
gin.external_configurable(value.ValueIteration)
gin.external_configurable(modules.ExplicitMDP)
gin.external_configurable(optax.sgd)
gin.external_configurable(optax.adam)

EXPLICIT_CONFIG = """
import experiments.chain.supervised.experiment
import experiments.chain.supervised.explicit

LEARNING_RATE = 1e-2
DISCOUNT = 0.99
MDP_MODULE_DISCOUNT = 0.99

weights/ValueIteration.tol = 1e-2
weights/ValueIteration.maxiter = 1000
weights/ValueIteration.reduce = @max_reduce
weights/ValueIteration.offset = @identity_offset

weights/ExplicitMDP.num_pseudo_actions = 1
weights/ExplicitMDP.discount_init = %MDP_MODULE_DISCOUNT

MDPSolveWeights.solver = @weights/ValueIteration()
MDPSolveWeights.mdp_module = @weights/ExplicitMDP()

ExplicitWeights.implicit_module = @MDPSolveWeights()

LinearModule.weight_module = @ExplicitWeights()

dataset/ValueIteration.tol = 1e-3
dataset/ValueIteration.maxiter = 2000
dataset/ValueIteration.reduce = @max_reduce
dataset/ValueIteration.offset = @identity_offset

create_chain_mdp.num_states = 11
create_chain_mdp.slip_prob = 0.
create_chain_mdp.good_reward = 10.
create_chain_mdp.bad_reward = 1.
create_chain_mdp.discount = %DISCOUNT

supervised_chain_dataset.value_solver = @dataset/ValueIteration()

batch_generator.data = @supervised_chain_dataset()
batch_generator.batch_size = 1
batch_generator.replace = True

adam.learning_rate = %LEARNING_RATE

experiment.train.model = @LinearModule()
experiment.train.optimizer = @adam()
experiment.train.test_data = @supervised_chain_dataset()
experiment.train.num_iterations = 100
experiment.train.eval_period = 10

experiment.run.seed = %SEED
"""

TEST_CONFIGS = {"explicit": EXPLICIT_CONFIG}


@pytest.fixture(params=TEST_CONFIGS.values(), ids=TEST_CONFIGS.keys())
def gin_config(request):
    config = request.param
    gin.parse_config(config)
    gin.parse_config("SEED = 1237")
    gin.finalize()
    yield config
    gin.clear_config()


class DummyLogger:

    def __init__(self):
        self.logged = []

    def write(self, data):
        self.logged.append(data)


def test_supervised(gin_config, tmp_path):
    logger = DummyLogger()
    supervised_experiment.run(run_id=0, log_dir=tmp_path, logger=logger)

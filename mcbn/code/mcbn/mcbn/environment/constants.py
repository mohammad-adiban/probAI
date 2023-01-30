import os

wd = os.getcwd()

# Dataset path
DATA_PATH = os.path.join(wd, 'mcbn/code/mcbn/data')

# Evaluation paths
EVALUATIONS_PATH = os.path.join(wd, 'mcbn/code/mcbn/evaluations')
HYPERPARAMS_EVAL_PATH = os.path.join(EVALUATIONS_PATH, 'mcbn/code/mcbn/hyperparameters')
TAU_EVAL_PATH = os.path.join(EVALUATIONS_PATH, 'tau')
TEST_EVAL_PATH = os.path.join(EVALUATIONS_PATH, 'test')
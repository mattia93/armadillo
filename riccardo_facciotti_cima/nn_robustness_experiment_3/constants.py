import tensorflow as tf
import optuna

# general constants
SAMPLES_BATCH = 1
CLASS_NAMES = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine"
}

OPTIMIZER = "rmsprop"
LOSS = "categorical_crossentropy"
METRICS = ["accuracy"]

EPOCHS = 10
# EPOCHS = 30
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
INPUT_SHAPE = (28 * 28,)
DEEP_NEURONS = 512
DEEP_ACTIVATION = {
    "sigmoid": tf.keras.initializers.GlorotUniform(),
    "tanh": tf.keras.initializers.GlorotUniform(),
    "relu": tf.keras.initializers.HeUniform(),
    "selu": tf.keras.initializers.LecunNormal(),
    "elu": tf.keras.initializers.HeUniform(),
    "exponential": tf.keras.initializers.GlorotUniform(),
    "swish": tf.keras.initializers.HeUniform(),
}
KERNEL_INITIALIZER = "he_uniform"
OUT_ACTIVATION = "softmax"
ES_PATIENCE = 10

MODELS_PATH = "./best_models"
SCORES_PATH = "./scores"
STUDY_DB_PATH = "./study_db"
ATTACK_SURFACE_FIG_PATH = "./attack_surface_fig"



# attack constants
MIN_XP, MAX_XP = (0, 27)
MIN_YP, MAX_YP = (0, 27)
MIN_ZP, MAX_ZP = (0.0, 1.0)
ATTACK_TYPE = ["max", "mean"]

# Optuna Constant
# N_TRIALS = 16
N_TRIALS = 2048

# this number must be a divisor of N_TRIALS
# N_CHECKPOINTS = 4
N_CHECKPOINTS = 16

TIMEOUT = 600
N_JOBS = -1
VERBOSITY = 0
DIRECTION = "minimize"
RANDOM_STATE = 3993

# Set optuna sampler and seed
SAMPLER = optuna.samplers.TPESampler(seed=RANDOM_STATE)  # Make the sampler behave in a deterministic way.
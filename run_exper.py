from hyperopt import Experiment
from trainer import train_mnist as trainer

config = {
# `computing_mode` accepts the following options:
#   "local-single" -> run all computation locally on a single thread (debug use)
#   "local-multi" -> run all computation locally and parallelized
#   "distributed" -> run on a cluster using remote workers
    "computing_mode": "local-multi",
# `search_strategy` accepts the following options:
#   "grid" -> perform a grid search in hyperparameter space
#   ...(other options not yet implemented)...
    "search_strategy": "grid",
# `grid_search_space` specifies values to be searched for each hyperparameter
#   each entry needs to follow the format {"param_name" : List(Any)}
    "grid_search_space": {
        "layer1_nodes": [16,32,64],
        "layer2_nodes": [16,32,64],
        "optimizer": ["adam", "sgd"],
    },
# `grid_search_settings` contain other settings for grid search strategy
#   `save_every_n_outputs`: how often should the trial results be saved.
#       The more often we save the results, the less likely we lose data in
#       the event of a crash, but it takes more time.
#   `num_samples`: how many repeated trials to run for each point in search space
#       [Not yet implemented]
    "grid_search_settings": {
        "save_every_n_outputs": 1,
        "num_samples": 1
    }
}

# trainer must be a function that takes an hpset as input and returns (hpset,
# metric, logs) as output
#   `hpset`: hyperparameter values, see hyperopt.Experiment._generate_hpsets
#   `metric`: objective for maximization, evaluated at the point specified by `hpset`
#   `logs`: any other useful information that should be saved

exper = Experiment(trainer, config)
exper.search()
exper.summary()

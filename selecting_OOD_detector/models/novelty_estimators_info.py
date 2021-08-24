from scipy.stats import uniform
from sklearn.utils.fixes import loguniform

from selecting_OOD_detector.models.ae import AE
from selecting_OOD_detector.models.dkl_due import DUE
from selecting_OOD_detector.models.flow import Flow
from selecting_OOD_detector.models.lof import LOF
from selecting_OOD_detector.models.ppca import PPCA
from selecting_OOD_detector.models.vae import VAE

IMPLEMENTED_MODELS = {
    "AE": AE,
    "DUE": DUE,
    "Flow": Flow,
    "LOF": LOF,
    "PPCA": PPCA,
    "VAE": VAE,
}

SCORING_FUNCTIONS = {
    "AE": "Reconstr Err",
    "DUE": "Standard Dev",
    "Flow": "Negative Log Prob",
    "LOF": "Outlier Score",
    "PPCA": "Negative Log Prob",
    "VAE":  "Reconstr Err",
}

HYPERPARAMETERS_MODEL_INIT = {
    "AE":
        {
            "hidden_sizes": [75],
            "latent_dim": 20,
            "lr": 0.006809,
        },
    "DUE":
        {
            "coeff": 3.222222,
            "depth": 7,
            "features": 256,
            "kernel": "Matern52",
            "lr": 0.005522,
            "n_inducing_points": 18,
        },
    "Flow":
        {
            "hidden_features": 256,
            "num_layers": 20,
            "batch_norm_between_layers": True,
            "lr": 0.001,
        },
    "LOF":
        {
            "n_neighbors": 5,
            "algorithm": "brute",
            "novelty": True,
        },

    "PPCA":
        {
            "n_components": 19,
        },
    "VAE":
        {
            "anneal": True,
            "beta": 0.20462,
            "hidden_sizes": [
                25,
                25,
                25
            ],
            "latent_dim": 10,
            "lr": 0.001565,
            "reconstr_error_weight": 0.238595,
        }
}

HYPERPARAMETERS_TRAINING = {
    "DUE": {"n_epochs": 5, "batch_size": 64},
    "PPCA": {},
    "LOF": {},
    "AE": {"n_epochs": 10, "batch_size": 64},
    "VAE": {"n_epochs": 6, "batch_size": 64},
    "Flow": {"n_epochs": 30, "batch_size": 128}
}

HYPERPARAMETERS_SEARCH_GRID = {
    "kernel": ["RFB", "Matern12", "Matern32", "Matern52", "RQ"],
    "n_inducing_points": range(10, 20),
    "num_layers": range(5, 40, 5),
    "hidden_features": [32, 64, 128, 256, 512],
    "batch_norm_between_layers": [True, False],
    "coeff": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
    "features": [64, 128, 256, 512, 1028],
    "depth": range(4, 8),
    "n_components": range(2, 20),
    "hidden_sizes": [
        [hidden_size] * num_layers
        for hidden_size in [25, 30, 50, 75, 100]
        for num_layers in range(1, 4)
    ],
    "latent_dim": [5, 10, 15, 20],
    "batch_size": [64, 128, 256],
    "lr": [0.0001, 0.001, 0.01, 0.1],
    "dropout_rate": [0, 0.1, 0.2, 0.3, 0.5],
    "reconstr_error_weight": loguniform(0.01, 0.9),
    "anneal": [True, False],
    "beta": uniform(loc=0.1, scale=2.4),
}

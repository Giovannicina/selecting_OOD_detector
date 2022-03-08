
# Selecting OOD Detector

## Your How-To Guide for OOD Detection

Out-of-distribution (OOD) detection is one of the crucial safety checks for the reliable deployment of machine learning models.  However, while it is a standard practice to tailor predictive models to a specific task, there is no universal way of testing OOD detection methods in practice.

<!-- <img src="https://raw.githubusercontent.com/Giovannicina/selecting_OOD_detector/docs/docs/img/1_TyogHTgvJx8mTemKYHJ0cg.png?token=ANKC3TFV7QTA3NSUX3C4UO3BGYLUO" 
     style="height: 120;"/> -->

This repository allows you to test and tailor OOD detection methods to a custom dataset and select the best OOD detector for your application.  

## Table of Contents
- [Selecting OOD Detector](#selecting-ood-detector)
  * [About](#about)
    + [Why is OOD detection important?](#why-is-ood-detection-important)
    + [Implemented OOD detection methods](#implemented-ood-detection-methods)
<!--   * [Usage](#usage) -->
  * [Examples](#examples)
    + [Detecting Clinically Relevant OOD Groups](#detecting-clinically-relevant-ood-groups)
    + [Fine-Tuning Hyperparmeters on a New Dataset](#fine-tuning-hyperparmeters-on-a-new-dataset)
  * [References](#references)

<br>

## About


### Why is OOD detection important?
Machine learning models have achieved great performance on a variety of tasks. However, models assume that new samples are similar to data they have been trained on and their performance can degrade rapidly when this assumption is violated.


### Implemented OOD detection methods
* Autoencoder (`AE`) with reconstruction error metric
* Variational Autoencoder (`VAE`; Kingma & Welling, 2014) with reconstruction error or log probability metric
* Spectral Normalized Gaussian Process (Deterministic Uncertainty Estimator `DUE`; Amersfoort et al., 2021) with standard deviation metric
* Masked Autoregressive Flow (`Flow`, Papamakarios et al., 2017) with log probability metric
* Probabilistic PCA (`PPCA`; Bishop et al., 1999) with log probability metric 
* Local Outlier Factor (`LOF`; de Vries et al., 2010) with outlier score 

<br>

<!-- ## Usage

    git clone https://github.com/Giovannicina/selecting_OOD_detector.git 
    cd selecting_OOD_detector
    pip install -r requirements.txt
 -->

## Examples
### Detecting Clinically Relevant OOD Groups

This example shows how to test OOD detectors on two groups using dummy variables of
* patients under 18 years
*  COVID-19 patients


First, define the in-distribution and OOD data:
```py
import pandas as pd
import numpy as np

n_features = 15

# Define training and testing in-distribution data
X_train = pd.DataFrame(np.random.rand(80, n_features))
X_test = pd.DataFrame(np.random.rand(20, n_features))

# Define OOD groups
X_under18 = pd.DataFrame(np.random.rand(12, n_features))
X_covid = pd.DataFrame(np.random.rand(7, n_features))

ood_groups = {
    "Patients under 18 years": X_under18,
    "COVID-19 patients": X_covid
}
			  
```

Next, initialize and fit OOD Pipeline to in-distribution data and score OOD groups:

```py
from selecting_OOD_detector.pipeline.ood_pipeline import OODPipeline

# Initialize the pipeline
oodpipe = OODPipeline()

# Fit OOD detection models on in-distribution training data and score in-distribution test data to calculate novelty baseline.
oodpipe.fit(X_train, X_test=X_test)

# Compute novelty scores of the defined OOD groups
oodpipe.evaluate_ood_groups(ood_groups, n_trials=3)

```

Finally, inspect AUC-ROC score of OOD detection:
```py
auc_scores = oodpipe.get_ood_aucs_scores(return_averaged=True)
```

|         |       AE |      DUE |       Flow |      LOF |     PPCA |      VAE |
|:--------|---------:|---------:|-----------:|---------:|---------:|---------:|
| Patients Under 18 | 0.513 | 0.552 | 0.493 | 0.489| 0.514 | 0.654 |
| COVID-19 patients | 0.525    | 0.631     | 0.567         | 0.567 | 0.474     | 0.553 |

AUC-ROC score of 1 would indicate perfect separation of an OOD group from testing data while a score of 0.5 suggests that models are unable to detect which samples are in- and out-of-distribution.

<br>
<br>

### Fine-Tuning Hyperparameters on a New Dataset

This example shows how to perform a hyperparameter search for each
dataset.

First, split your data into training, testing, and validation:

```py
import pandas as pd
from sklearn.model_selection import train_test_split

n_features = 32
n_samples = 150

X = pd.DataFrame(np.random.rand(n_samples, n_features))
y = np.random.binomial(n=1, p=0.95, size=[n_samples])

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)
```



Next, initialize `HyperparameterTuner`:

```py
from selecting_OOD_detector.pipeline.tuner import HyperparameterTuner

hyperparm_tuner = HyperparameterTuner(num_evals_per_model=5)
```



Run the hyperparameter search with the HyperparameterTuner. Note that
intermediate results can be saved during the run:

```py
hyperparm_tuner.run_hyperparameter_search(
  X_train = X_train,
  X_val=X_val,
  y_train=y_train,
  y_val=y_val,
)
```



To get the best parameters, simply use `get_best_parameters` function:

```py
hyperparm_tuner.get_best_parameteres()
```

``` {.sourceCode .py}
{
  'AE': {   'hidden_sizes': [50, 50],
            'input_size': 32,
            'latent_dim': 15,
            'lr': 0.01},
  'DUE': {   'coeff': 1,
             'depth': 4,
             'features': 512,
             'input_size': 32,
             'kernel': 'Matern52',
             'lr': 0.1,
             'n_inducing_points': 11},
  'Flow': {   'batch_norm_between_layers': True,
              'hidden_features': 128,
              'input_size': 32,
              'lr': 0.01,
              'num_layers': 15},
  'LOF': {    'input_size': 32, 
              'n_neighbors': 19},
  'PPCA': {  'input_size': 32,
             'n_components': 3},
  'VAE': {   'anneal': True,
             'beta': 1.786466646725514,
             'hidden_sizes': [30, 30, 30],
             'input_size': 32,
             'latent_dim': 5,
             'lr': 0.1,
             'reconstr_error_weight': 0.14695309349947033}
 }
```

You can save these best parameters and use them in the OODPipeline
later:

```py
hyperparm_tuner.save_best_parameters_as_json(save_dir = "../data/hyperparameters/custom/")
```

```py
from selecting_OOD_detector.pipeline.ood_pipeline import OODPipeline

# Initialize the pipeline
oodpipe = OODPipeline()

# Use the custom hyperparameters that were just saved
oodpipe.fit(X_train, X_test=X_test, hyperparameters_dir="../data/hyperparameters/custom/")
```




## References

```bib
	@article{2021arXiv210914885Z,
	       author = {{Zadorozhny}, Karina and {Thoral}, Patrick and {Elbers}, Paul and {Cin{\`a}}, Giovanni},
		title = "{Out-of-Distribution Detection for Medical Applications: Guidelines for Practical Evaluation}",
	      journal = {arXiv e-prints},
	     keywords = {Computer Science - Machine Learning},
		 year = 2021,
		month = sep,
		pages = {arXiv:2109.14885},
	archivePrefix = {arXiv},
	       eprint = {2109.14885},
	 primaryClass = {cs.LG},
	}

```





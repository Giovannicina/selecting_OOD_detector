
# Selecting OOD Detector

Out-of-distribution (OOD) detection is one of the crucial safety checks for reliable deployment of machine learning models.  However, while it is a standard practice to tailor predictive models to a specific task, there is no universal way of testing OOD detection methods in practice.

This repository allows you to test and tailor OOD detection methods to custom dataset and select the best OOD detector for your application.  

### Why is OOD detection important?
Machine learning models have achieved great performance on variety of tasks. However, models assume that new samples are similar to data they have been trained on and their performance can degrade rapidly when this assumption is violated.


### Implemented OOD detection methods
* Autoencoder (`AE`) with reconstruction error metric
* Variational Autoencoder (`VAE`; Kingma & Welling, 2014) with reconstruction error or log probability metric
* Spectral Normalized Gaussian Process (Deterministic Uncertainty Estimator `DUE`; Amersfoort et al., 2021) with standard deviation metric
* Masked Autoregressive Flow (`Flow`, Papamakarios et al., 2017) with log probability metric
* Probabilistic PCA (`PPCA`; Bishop et al., 1999) with log probability metric 
* Local Outlier Factor (`LOF`; de Vries et al., 2010) with outlier score 


## Example of Usage: Under 18 and COVID-19 Patients

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

ood_groups = {"Patients under 18 years": X_under18,
			  "COVID-19 patients": X_covid}
			  
```

Next, initialize and fit OOD Pipeline to in-distribution data and score OOD groups:

```py
from selecting_OOD_detector.pipeline.ood_pipeline import OODPipeline

# Initialize the pipeline
oodpipe = OODPipeline()

# Fit OOD detection models on in-distribution training data and score in-distribution test data to calculate novelty baseline.
oodpipe.fit(X_train, X_test=X_test)

# Compute novelty scores of the defined OOD groups
oodpipe.evaluate_ood_groups(ood_groups)

```

Finally, inspect AUC-ROC score of OOD detection:
```py
auc_scores = oodpipe.get_ood_aucs_scores(return_averaged=True)
```

|         |       AE |      DUE |       Flow |      LOF |     PPCA |      VAE |
|:--------|---------:|---------:|-----------:|---------:|---------:|---------:|
| Patients Under 18 | 0.513 | 0.552 | 0.493 | 0.489| 0.514 | 0.654 |
| COVID-19 patients | 0.525    | 0.631     | 0.567         | 0.567 | 0.474     | 0.553 |

AUC-ROC score of 1 would indicate perfect separation of an OOD group from testing data while score of 0.5 suggests that models are unable to detect which samples are in- and out-of-distribution.

## Usage

```
git clone https://github.com/Giovannicina/selecting_OOD_detector.git 
cd selecting_OOD_detector
pip install -r requirements.txt
sys.path.append(os.getcwd())
```

Import OOD pipeline and apply to your data as shown in the example above:
```py
from selecting_OOD_detector.pipeline.ood_pipeline import OODPipeline
```


## References

##########
Examples
##########
This page shows examples of how to use ``OODPipeline`` and ``HyperparameterTuner`` for your applications.


.. contents::
   :depth: 3
..

|

Detecting Clinically Relevant OOD Groups
*****************************************

This example shows how to test OOD detectors on two groups using dummy
variables of 

* patients under 18 years 

* COVID-19 patients

First, define the in-distribution and OOD data:

.. code:: py

    import pandas as pd
    import numpy as np
    
    # Define training and testing in-distribution data
    n_features = 15
    X_train = pd.DataFrame(np.random.rand(80, n_features))
    X_test = pd.DataFrame(np.random.rand(20, n_features))

    # Define OOD groups
    X_under18 = pd.DataFrame(np.random.rand(12, n_features))
    X_covid = pd.DataFrame(np.random.rand(7, n_features))

    ood_groups = {"Patients under 18 years": X_under18,
                  "COVID-19 patients": X_covid}
                  
|

Next, initialize and fit OOD Pipeline to in-distribution data and score
OOD groups:

.. code:: py

    from selecting_OOD_detector.pipeline.ood_pipeline import OODPipeline

    # Initialize the pipeline
    oodpipe = OODPipeline()

    # Fit OOD detection models on in-distribution training data and score in-distribution test data to calculate novelty baseline.
    oodpipe.fit(X_train, X_test=X_test)

    # Compute novelty scores of the defined OOD groups
    oodpipe.evaluate_ood_groups(ood_groups)

|

Finally, inspect AUC-ROC score of OOD detection:

.. code:: py

    auc_scores = oodpipe.get_ood_aucs_scores(return_averaged=True)

+---------------------+---------+---------+---------+---------+---------+---------+
|                     | AE      | DUE     | Flow    | LOF     | PPCA    | VAE     |
+=====================+=========+=========+=========+=========+=========+=========+
| Patients Under 18   | 0.513   | 0.552   | 0.493   | 0.489   | 0.514   | 0.654   |
+---------------------+---------+---------+---------+---------+---------+---------+
| COVID-19 patients   | 0.525   | 0.631   | 0.567   | 0.567   | 0.474   | 0.553   |
+---------------------+---------+---------+---------+---------+---------+---------+

AUC-ROC score of 1 would indicate perfect separation of an OOD group
from testing data while score of 0.5 suggests that models are unable to
detect which samples are in- and out-of-distribution.

|

To visualize distributions of novelty scores, plot histogram using `plot_score_distributions` or boxplots using `plot_box_plot` functions:

.. code:: py

    oodpipe.plot_box_plot()

.. image:: https://raw.githubusercontent.com/karinazad/selecting_OOD_detector/master/docs/img/download%20(1).png

.. image:: https://raw.githubusercontent.com/karinazad/selecting_OOD_detector/master/docs/img/download.png


The plot is annotated with the results of Mann-Whitney one-sided statistical test from ``statannot``.



|
|
|

Fine-Tuning Hyperparmeters on a New Dataset
*****************************************

This example shows how to perform hyperparameter search for each dataset.


First, split your data into training, testing, and validation:

.. code:: py

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    n_features = 32
    n_samples = 150
    X = pd.DataFrame(np.random.rand(n_samples, n_features))
    y = np.random.binomial(n=1, p=0.95, size=[n_samples])

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

             
|

Next, initialize ``HyperparameterTuner``:

.. code:: py

    from selecting_OOD_detector.pipeline.tuner import HyperparameterTuner

    hyperparm_tuner = HyperparameterTuner(num_evals_per_model=5)

|

Run the hyperparameter search with the HyperparameterTuner. Note that intermediate results can be saved during the run:

.. code:: py

    hyperparm_tuner.run_hyperparameter_search(X_train = X_train,
                                              X_val=X_val,
                                              y_train=y_train,
                                              y_val=y_val,
                                              save_intermediate_scores=True,
                                              save_dir="hyperparameter_search_test/")


|

To get the best parameters, simply use ``get_best_parameters`` function:

.. code:: py
    
    hyperparm_tuner.get_best_parameteres()
    
 
.. code:: py

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
    
|
You can save these best parameters and use them in ``OODPipeline`` later:


.. code:: py

    tuner.save_best_parameters_as_json(save_dir = "../data/hyperparameters/custom/")
    
    
.. code:: py

    from selecting_OOD_detector.pipeline.ood_pipeline import OODPipeline

    # Initialize the pipeline
    oodpipe = OODPipeline()

    # Use the custom hyperparameters that were just saved
    oodpipe.fit(X_train, X_test=X_test, hyperparameters_dir="../data/hyperparameters/custom/")
    
    
This way, the OOD detection models used by ``OODPipeline`` are fine-tuned to your dataset.




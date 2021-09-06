API Reference
####################################


.. contents::
   :depth: 3
..
      
.. container::

   -  .. rubric:: ``Classes``
         :name: classes

      -  .. rubric:: ``OODPipeline``
            :name: oodpipeline

         -  ``fit``
         -  ``evaluate_ood_groups``
         -  ``get_auc_scores``
         -  ``plot_auc_scores``
         -  ``plot_box_plot``
         -  ``plot_score_distr``
         
         
      -  .. rubric:: ``HyperparameterTuner``
            :name: hyperparametertuner

         -  ``run_hyperparameter_search``
         -  ``get_best_parameteres``
         -  ``save_best_parameters_as_json``
         -  ``save_scores_for_evaluated_paramaters``
         

|


OOD Pipeline
*****************


``selecting_OOD_detector.pipeline.ood_pipeline.OODPipeline``

   .. container:: section
      :name: section-intro

   .. container:: section

   .. container:: section

   .. container:: section

   .. container:: section

      .. rubric:: Classes
         :name: header-classes
         :class: section-title

      ``class OODPipeline (**kwargs)``
         .. container:: desc

            Pipeline to fit novelty estimators on in-distribution data
            and evaluate novelty of Out-of-Distribution (OOD) groups.

            Example of usage:

            ::

               # Initialize the pipeline
               oodpipeline = OODPipeline()

               # Fit the pipeline on in-distribution training data and compute novelty scores for in-distribution test data
               oodpipeline.fit(X_train= X_train, X_test=X_test)

               # Define OOD groups and evaluate by the pipeline
               ood_groups = {"Flu patients": X_flu, "Ventilated patients": X_vent}
               oodpipeline.evaluate(ood_groups)

               # Inspect AUC-ROC scores of detecting OOD groups
               oodpipeline.get_ood_aucs_scores()

            .. rubric:: Parameters
               :name: parameters

            ``kwargs``:
               ``model_selection``: ``set`` Define which models to train, e.g.
               ``{"PPCA", "LOF", "VAE"}``. If selection is not provided, all
               available models are used.


         .. rubric:: Ancestors
            :name: ancestors

         -  `BasePipeline <base.html#selecting_OOD_detector.pipeline.base.BasePipeline>`__


         .. rubric:: Methods
            :name: methods

         ``def fit(self, X_train, X_test, **kwargs)``
            .. container:: desc

               Fits models on training data with n_trials different
               runs. Novelty estimators from each run are stored in a
               nested dictionary in self.novelty_estimators. (E.g.: {0:
               {"AE": NoveltyEstimator, "PPCA": NoveltyEstimator},  1:
               {"AE": NoveltyEstimator, "PPCA": NoveltyEstimator}} )``
               
               .. rubric:: Parameters

               --------------

               ``X_train`` : ``pd.DataFrame``
                  Training in-distribution data. Used to fit novelty
                  estimators.
               ``X_test`` : ``pd.DataFrame``
                  Test in-distribution data. Used to calculate
                  self.in_domain_scores which are taken as base novelty
                  scores for the dataset and used for comparison against
                  OOD groups later.

               ``kwargs``:
                  ``y_train``: ``pd.DataFrame``
                           Labels corresponding to training data.
                  ``n_trials``: ``int`` 
                           Number of independent trials to run, default is set to 5. All subsequent results will be averaged from the indicated number of runs. 
                           
         |
          
                  
         ``def evaluate_ood_groups(self, ood_groups, return_averaged=False)``
            .. container:: desc

               Gives novelty scores to OOD groups. Returns and stores
               dictionary of novelty scores given by each model for each
               sample in every OOD group. If the function is called
               repeadetly, updates internally stored novelty scores for
               the OOD groups.

               .. rubric:: Parameters
                  :name: parameters

               ``ood_groups`` : ``dict``
                  Dictionary of OOD groups. Dictionary has to contain a
                  name of each OOD group and features in a pd.DataFrame.
                  Example: {"Flu patients": X_flu, "Ventilated
                  patients": X_vent}
               ``return_averaged`` : ``bool``
                  If true, returns averaged novelty score for each
                  sample. The shape of novelty scores given by each
                  model then corresponds to (1, n_samples). Else, the
                  shape of novelty scores given by each model is
                  (n_trials, n_samples) where n_trials is the number of
                  trials used in the fit function.

               .. rubric:: Returns
                  :name: returns

             ``out_domain_scores`` : ``dict``
                  Returns a dictionary of novelty scores given by each
                  model for each sample in every OOD group.

         |


         ``def get_auc_scores(self, ood_groups_selections=None, return_averaged=True)``
            .. container:: desc

               Computes AUC-ROC scores of OOD detection for each OOD
               group as compared to the in-distribution test data. By
               default, returns scores for every group evaluated by the
               pipeline (evaluate_ood_groups).

               .. rubric:: Parameters
                  :name: parameters

               ``ood_groups_selections``: ``Optional(list)``
                  Optionally provide a selection of OOD groups for which
                  AUC-ROC score should be returned. If no selection is
                  provided, all groups ever evaluate by the pipeline
                  will be included.
               ``return_averaged``: ``bool``
                  Indicates whether to return averaged AUC-ROC scores
                  over n_trials run or a list of scores for every trial.

               .. rubric:: Returns
                  :name: returns

               ``aucs_dict_groups``: ``dict``
                  A nested dictionary that contains a name of OOD group,
                  name of novelty estimator and either a float (if
                  averaged) or a list of AUC-ROC scores.

         |
        


         ``def plot_auc_scores(self, ood_groups_selections=None, show_stderr=True, save_dir=None, **plot_kwargs)``
            .. container:: desc

               Plots a heatmap of AUC-ROC scores of OOD detection for
               each OOD group as compared to the in-distribution test
               data.

               .. rubric:: Parameters
                  :name: parameters

               ``ood_groups_selections`` : ``Optional(list)``
                  Optionally provide a selection of OOD groups for which
                  AUC-ROC score should be returned. If no selection is
                  provided, all groups ever evaluate by the pipeline
                  will be included.
               ``show_stderr``: ``Optional(bool)``
                  If True (default), annotates the heatmpa with means
                  and standard error (calculated using jacknife
                  resampling). Else, plots the mean values only.
               ``save_dir``: ``Optional(str)``
                  If a path to a directory is provided, saves plots for
                  each OOD group separately.
               ``plot_kwargs``
                  Other arguments to be passed to sns.heatmap function.

      
         |

         
         ``def plot_box_plot(self, ood_groups_selections=None, save_dir=None)``
            .. container:: desc
            
                  Plots boxplots for each OOD group as compared to the in-distribution test
                  data. Adds statistical annotation of difference significance 
                  under Mann-Whitney one sided test using ``statannot`` package.
                  
                  p-values: 
                  **** 𝑝 < 0.0001; ***  𝑝 < 0.001; **    𝑝 < 0.01; *     𝑝 < 0.05; else ns 

                  

               .. rubric:: Parameters
                  :name: parameters

               ``ood_groups_selections`` : ``Optional(list)``
                  Optionally provide a selection of OOD groups for which
                  AUC-ROC score should be returned. If no selection is
                  provided, all groups ever evaluate by the pipeline
                  will be included.
               ``save_dir``: ``Optional(str)``
                  If a path to a directory is provided, saves plots for
                  each OOD group separately.
                  
      
         |

         ``def plot_score_distr(self, ood_groups_selections=None, save_dir=None)``
            .. container:: desc
            
                  Plots histograms for each OOD group as compared to the in-distribution test
                  data. To avoid outliers from skewing the distributions on the plots to the left, clips values 
                  of novely scores to 0-95% range of in-distribution novelty scores.
                  
               .. rubric:: Parameters
                  :name: parameters

               ``ood_groups_selections``: ``Optional(list)``
                  Optionally provide a selection of OOD groups for which
                  AUC-ROC score should be returned. If no selection is
                  provided, all groups ever evaluate by the pipeline
                  will be included.
               ``save_dir``: ``Optional(str)``
                  If a path to a directory is provided, saves plots for
                  each OOD group separately.
                  
|
|

Hyperparameter Tuner
********************

``selecting_OOD_detector.pipeline.tuner.HyperparameterTuner``


.. container::

   .. container:: section
      :name: section-intro

      This module implements HyperparameterTuner which performs
      hyperparameter search for density estimators and discriminators.
      
   .. container:: section

   .. container:: section

   .. container:: section

   .. container:: section

      .. rubric:: Classes
         :name: header-classes
         :class: section-title


      ``class HyperparameterTuner (hyperparameter_search_grid=None, hyperparameters_names=None, train_params=None, model_selection=None, num_evals_per_model=20)``
         .. container:: desc

            Performs hyperparameter search for implemented novelty
            estimators and provided data.
            
            Example of usage:

            ::

               # Initialize the hyperparameter tuner
               hyperparam_tuner = HyperparameterTuner()

               # Run hyperparameter search on your data
               hyperparam_tuner.run_hyperparameter_search(X_train= X_train, X_val=X_val)

               # Display or save the best hyperparameters found
               hyperparam_tuner.get_best_parameteres()
               hyperparam_tuner.save_best_parameters_as_json(save_dir="search_results/")
               

            .. rubric:: Parameters
               :name: parameters

            ``hyperparameter_search_grid`` : ``Optional(dict)``
               A dictionary that specifies all possible values of
               hyperparameters for all models to be evaluated. Example:
               {"kernel": ["RFB", "Matern12", "Matern32", "Matern52",
               "RQ"], "n_inducing_points": range(10, 20)}
            ``hyperparameters_names`` : ``Optional(dict)``
               A dictionary of lists with strings that specifies the
               names of parameters that each model uses for
               initialization. Example: {"AE": ["hidden_sizes",
               "latent_dim", "lr"], "PPCA": ["n_components"]}
            ``train_params`` : ``Optional(dict)``
               A dictionary that specifies hyperparameters used in the
               training function.
            ``model_selection`` : ``Optional(set)``
               Specifies a subset of models to be evaluated. If no
               selectio is provided, evaluates all implemented models.
            ``num_evals_per_model`` : ``int``
               Number of evaluations to run for each model.
                 

         |
         .. rubric:: Methods
            :name: methods
            

         ``def run_hyperparameter_search(self, X_train, X_val, y_train=None, y_val=None, save_intermediate_scores=True, save_dir=None)``
            .. container:: desc

               Performs hyperparameters search for all models and stores
               the results internally. 


               .. rubric:: Parameters
                  :name: parameters               

               ``X_train`` : ``pd.DataFrame``
                  Training data.
               ``X_val`` : ``pd.DataFrame``
                  Validation data to calculate scores on.
               ``y_train`` : ``Optional(pd.DataFrame):``
                  Labels corresponding to the training data. Only used
                  for discriminator models.
               ``y_val`` : ``Optional(pd.DataFrame)``
                  Labels corresponding to the validation data. Only used
                  for discriminator models.
               ``save_intermediate_scores`` : ``bool``
                  If True, saves results after each run in case of an
                  abrupt termination of the run.
               ``save_dir`` : ``Optional(str)``
                  If save_intermediate_scores is True, saves the results
                  to provided directory. If no directory is provided,
                  saves the results to
                  "../data/hyperparameters/scores/")


         |
         


         ``def get_best_parameteres(self)``
            .. container:: desc

               Returns the top performing paramaters for each model in a
               nested dictionary. Returns

               
               .. rubric:: Parameters
                  :name: parameters
               
               ``best_params``: ``dict``
                  A nested dictionary that stores the best parameters
                  found for each model.

            

         |

         ``def save_best_parameters_as_json(self, save_dir=None)``
            .. container:: desc

               Saves the top performing paramaters for each model in a
               nested dictionary. Parameters

               
               .. rubric:: Parameters
                  :name: parameters
               
               ``save_dir`` : ``Optional(str)``
                  Directory to save the results to. If no directory is
                  provided, saves the paramaters to:
                  "../data/hyperparameters/custom/"


         |

         ``def save_scores_for_evaluated_paramaters(self, save_dir=None)``
            .. container:: desc

               Saves scores for all evaluated parameters for each model.
               Parameters


               .. rubric:: Parameters
                  :name: parameters
               
               ``save_dir`` : ``Optional(str)``
                  Directory to save the results to. If no directory is
                  provided, saves the paramaters to:
                  "../data/hyperparameters/scores/")

           


Generated by `pdoc 0.10.0 <https://pdoc3.github.io/pdoc>`__.

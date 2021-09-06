
########
About
########


About OOD Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Machine learning models have achieved great performance on variety of
tasks. However, models assume that new samples are similar to data they
have been trained on and their performance can degrade rapidly when this
assumption is violated.

.. image:: https://raw.githubusercontent.com/karinazad/selecting_OOD_detector/master/docs/img/Screen%20Shot%202021-08-19%20at%2012.17.46%20PM.png


Implemented Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Autoencoder (``AE``) with reconstruction error metric
-  Variational Autoencoder (``VAE``; Kingma & Welling, 2014) with
   reconstruction error or log probability metric
-  Spectral Normalized Gaussian Process (Deterministic Uncertainty
   Estimator ``DUE``; Amersfoort et al., 2021) with standard deviation
   metric
-  Masked Autoregressive Flow (``Flow``, Papamakarios et al., 2017) with
   log probability metric
-  Probabilistic PCA (``PPCA``; Bishop et al., 1999) with log
   probability metric
-  Local Outlier Factor (``LOF``; de Vries et al., 2010) with outlier
   score
   

Authors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Karina Zadorozhy

* Giovanni Cinà

*Pacmed BV, Netherlands*



Reference 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

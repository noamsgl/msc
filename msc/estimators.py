import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

SEC = 1e6
MIN = 60 * SEC
HOUR = 60 * MIN

class BSLE(BaseEstimator):
    """ The Bayesian Seizure Likelihood Estimator classifier.
    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(
        self,
        prior=None,
        horizon=30 * MIN
        ):
        self.prior = prior
        self.horizon = horizon

    def fit(self, X, y=None, sample_time=None):
        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_time : array-like of shape (n_samples,), default=None
            Array of times that are assigned to individual
            samples. If not provided,
            then the prior must be uniform.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X = self._validate_data(X)
        self.X_ = X
        self.is_fitted_ = True

        # Return the classifier
        return self

    def predict_proba(self, X):
        """ calculate seizure likelihood for each sample
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the more likely label given by self.predict_proba
        """
        X = self._validate_data(X)
        
        return np.ones(len(X))

    def predict(self, X):
        """ prediction for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the more likely label given by self.predict_proba
        """
        # Check if fit had been called
        check_is_fitted(self, ['X_', 'is_fitted_'])

        # Input validation
        X = check_array(X)
        
        # Predict class probabilities
        proba = self.predict_proba(X)

        # return most likely class
        closest = (proba > 0.5).astype(int)
        return closest



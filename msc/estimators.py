from functools import partial
import numpy as np
from scipy.special import i0
from scipy.stats import percentileofscore
from sklearn import mixture
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .prior_utils import vm_density, event_times_to_circadian_hist

SEC = 1
MIN = 60 * SEC
HOUR = 60 * MIN


class BSLE(BaseEstimator):
    """ The Bayesian Seizure Likelihood Estimator classifier.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    """

    def __init__(self, thresh: float = 0.05):
        """
        Parameters
        ----------
        thresh : float
            used to differentiate inliers from outliers for the likelihood function
        """
        self.thresh = thresh

    def fit(self, X, y=None, prior_events=None):
        """learn the density of data samples.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : Ignored
            Not used, present here for API consistency by convention.
        prior_events : array-like of shape (n_samples,), default=None
            Array of times that are known to have had seizures. If not provided,
            then the resulting prior will be uniform.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X = self._validate_data(X)
        # X, y = check_X_y(X, y)
        self.X_ = X
        self.is_fitted_ = True

        # initialize density estimator
        de = mixture.GaussianMixture(n_components=4, covariance_type="full")
        de.fit(X)

        # compute sample likelihoods
        likelihoods = de.score_samples(X)

        # compute p-values
        percentiles = np.array([percentileofscore(likelihoods, i) for i in likelihoods])
        p_values = percentiles / 100

        cutoff = np.percentile(p_values, self.thresh * 100)

        # divide into inliers and outliers
        inliers = X[p_values <= cutoff]
        outliers = X[p_values > cutoff]

        assert len(inliers) > 1, f"error: {len(inliers)=} must be greater than 1. try changing thresh"
        assert len(outliers) > 1, f"error: {len(outliers)=} must be greater than 1. try changing thresh"
        
        # compute density for each
        self.inlier_de_ = mixture.GaussianMixture(n_components=4, covariance_type="full")
        self.outlier_de_ = mixture.GaussianMixture(n_components=4, covariance_type="full")
        self.inlier_de_.fit(inliers)
        self.outlier_de_.fit(outliers)

        # initialize prior
        if prior_events is None:
            self.prior_ = None
        else:
            prior_events = prior_events.to_numpy()
            self.prior_ = self._get_vm_prior(prior_events)

        # Return the classifier
        return self

    def predict_proba(self, X, samples_times=None):
        """ calculate seizure likelihood for each sample
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The probability of the sample belonging to the seizure class
        """
        check_is_fitted(self)
        X = self._validate_data(X)
        if samples_times is not None:
            assert len(X) == len(samples_times), f"error: length mismatch, {len(X)} != {len(samples_times)}"

        log_likelihoods = self.outlier_de_.score_samples(X)
        log_likelihoods_training = self.outlier_de_.score_samples(self.X_)

        
        if samples_times is None:  # unsupervised
            # compute p-values from observed log-likelihoods based on training samples
            percentiles = np.array([percentileofscore(log_likelihoods_training, i) for i in log_likelihoods])
            p_values = percentiles / 100
            return p_values
        else:   # weakly supervised
            if self.prior_ is None:
                raise AttributeError("self.prior_ is None. you should fit the estimator with prior events.")
            else:
                # compute p_values 
                priors = np.array([self.prior_(t) for t in samples_times])
                evidence = np.exp(self.outlier_de_.score_samples(X)) * priors + np.exp(self.outlier_de_.score_samples(X)) * (1 - priors)
                posteriors = priors * np.exp(self.outlier_de_.score_samples(X)) / evidence
                percentiles = np.array([percentileofscore(posteriors, i) for i in posteriors])
                p_values = percentiles / 100
                return p_values

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

    @staticmethod
    def _get_vm_prior(prior_events: np.ndarray):
        """
        return a prior over seizure occurrence as a function of time t
        Parameters
        ----------
        prior_events : array-like of shape (n_samples,)
            Array of times that are known to have had seizures.
        Returns
        -------

        """
        circadian_hist = event_times_to_circadian_hist(prior_events)  # ensure only past events affect current prior
        def vm_prior(t: float):
            """
            the von Mises mixture model prior over the 24 hour cycle
            Parameters
            ----------
            t       time

            Returns
            -------

            """
            N = 24  # hours
            mus = np.arange(N) + 0.5

            def vm_mixture(x): return sum(
                [circadian_hist[i] * partial(vm_density, mu=mu)(x) for i, mu in enumerate(mus)])

            return vm_mixture(t) / np.trapz(vm_mixture(mus))

        return vm_prior
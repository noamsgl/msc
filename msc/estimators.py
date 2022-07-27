from functools import partial
import numpy as np
from scipy.special import i0
from scipy.stats import percentileofscore
from sklearn import mixture
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .prior_utils import vm_density, event_times_to_circadian_hist, PercentileOfScore
from .time_utils import SEC, MIN, HOUR, DAY


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
        self.hdpi_cutoff_value_ = None
        self.de_ = None
        self.p_E_ = None
        self.prior_ = None

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
        # Check that X has correct shape
        X = self._validate_data(X)
        # X, y = check_X_y(X, y)
        self.X_ = X
        self.is_fitted_ = True

        # initialize density estimator
        self.de_ = mixture.GaussianMixture(n_components=2, covariance_type="full")
        self.de_.fit(X)

        # compute sample likelihoods
        # density estimation of training samples
        self.p_E_ = self.de_.score_samples(X)
        # compute cutoff_value
        self.hdpi_cutoff_value_ = np.partition(self.p_E_, int(len(self.p_E_) * self.thresh))[int(len(self.p_E_) * self.thresh)]

        # initialize prior
        if prior_events is not None:
            assert isinstance(prior_events, np.ndarray), f"error: prior_events must be a numpy array"
            self.prior_ = self._get_vm_prior(prior_events)
        else:
            base_rate_per_day = 1/(10 * DAY)
            sample_length = 10 * SEC / 24 * HOUR
            self.prior_ = lambda t: base_rate_per_day * sample_length

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
        check_is_fitted(self, ['hdpi_cutoff_value_', 'de_', 'p_E_'])
        X = self._validate_data(X)
        if samples_times is not None:
            assert len(X) == len(samples_times), f"error: length mismatch, {len(X)} != {len(samples_times)}"

        p_E = self.de_.score_samples(X)
        # compute p-values from observed log-likelihoods based on training samples
        P_E = np.array(PercentileOfScore(self.p_E_).pct(p_E)) / 100

        # compute p_values
        not_S = p_E >= self.hdpi_cutoff_value_
        P_E_given_not_S = np.where(not_S, P_E, np.zeros_like(p_E))
        P_S_given_t = self.prior_(samples_times)
        P_not_S_given_t = 1 - P_S_given_t
        P_not_S_given_E = np.where(not_S, P_E_given_not_S * P_not_S_given_t / P_E, np.zeros_like(p_E))
        P_S_given_E = 1 - P_not_S_given_E
        return P_S_given_E

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
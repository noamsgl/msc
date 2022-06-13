from sklearn.utils.estimator_checks import check_estimator

from msc.estimators import BSLE


bsle = BSLE()

check_estimator(bsle)
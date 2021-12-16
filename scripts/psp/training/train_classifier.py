from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, classification_report, ConfusionMatrixDisplay, \
    RocCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

from msc.dataset.dataset import PSPDataset
import matplotlib.pyplot as plt

data_dir = r"/results/epilepsiae/max_cross_corr/surfCO/pat_3500/20211213T182128"

dataset = PSPDataset(data_dir)
X, labels = dataset.get_X(), dataset.get_labels()

le = LabelEncoder()
le.fit(labels)

# clf = make_pipeline(StandardScaler(), SVC(verbose=1))
clf = XGBClassifier(objective='reg:squarederror')
scores = cross_val_score(clf, X, le.transform(labels), cv=5, scoring='neg_mean_squared_error')
print((-scores)**0.5)
X_train, X_test, y_train, y_test = train_test_split(
    X, le.transform(labels), test_size=0.4, random_state=0)
# clf = LogisticRegression(solver='lbfgs', max_iter=1000)
clf.fit(X_train, y_train)

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
# plt.show()
RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.show()
#
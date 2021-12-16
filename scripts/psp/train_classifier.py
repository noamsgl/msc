from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,classification_report

from msc.dataset.dataset import PSPDataset
import matplotlib.pyplot as plt

data_dir = r"C:\Users\noam\Repositories\noamsgl\msc\results\epilepsiae\max_cross_corr\surfCO\pat_3500\20211213T182128"

dataset = PSPDataset(data_dir)
X, labels = dataset.X, dataset.labels

clf = LogisticRegression()
scores = cross_val_score(clf, X, labels, cv=5, scoring='roc_auc')
print(f"{scores=}")
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.4, random_state=0)
clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, y_train)

plot_confusion_matrix(clf, X_test, y_test)
plt.show()
plot_roc_curve(clf, X_test, y_test)
plt.show()

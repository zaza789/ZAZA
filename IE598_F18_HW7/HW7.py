import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/'
                      'wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']

from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test =\
    train_test_split(X, y,
                     test_size=0.1,
                     random_state=1,
                     stratify=y)

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
for i in [10, 30, 50 , 100, 150, 200, 250, 500, 1000]:
    forest = RandomForestClassifier(criterion='gini',
                                    n_estimators=i,
                                    max_depth=4,
                                    random_state=1)
    forest.fit(X_train, y_train)
    y_test_pred = forest.predict(X_test)
    cvs = cross_val_score(estimator = forest , X = X_train , y = y_train, cv=10,n_jobs=1)
    print('N_estimator= %.0f' % i)
    print('Out of Sample Accuracy: %.5f' % metrics.accuracy_score(y_test, y_test_pred))
    print('Cross Validation Mean Score: %.5f' % np.mean(cvs))
    print('Cross Validation Standard Deviation: %.5f' % np.std(cvs))



import matplotlib.pyplot as plt
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=250,
                                random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
    feat_labels[indices[f]],
    importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),
    importances[indices],
    align='center')
plt.xticks(range(X_train.shape[1]),
    feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

print("My name is Sa Yang")
print("My NetID is: say2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

in_sample_accuracy = []
out_of_sample_accuracy = []

def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))
    
from sklearn.model_selection import train_test_split

for i in range (1,11):
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=i, stratify=y)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(criterion='gini', 
                                  max_depth=4, 
                                  random_state=1)
    from sklearn import metrics
    tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    in_sample_accuracy.append(metrics.accuracy_score(y_train, y_train_pred))
    out_of_sample_accuracy.append(metrics.accuracy_score(y_test, y_test_pred))
    
    print('Random State: %.0f' % i)
    print('Train Accuracy: %.3f' % metrics.accuracy_score(y_train, y_train_pred))
    print('Test Accuracy: %.3f' % metrics.accuracy_score(y_test, y_test_pred))

print('In sample accuracy: mean = %.3f' % np.mean(in_sample_accuracy))
print('In sample accuracy: std = %.3f' % np.std(in_sample_accuracy))
print('Out of sample accuracy: mean = %.3f' % np.mean(out_of_sample_accuracy))
print('Out of sample accuracy: std = %.3f' % np.std(out_of_sample_accuracy))


from sklearn.model_selection import KFold
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1, stratify=y)
kfold = KFold(n_splits=10,random_state=1).split(X_train, y_train)
tree = DecisionTreeClassifier(criterion='gini', 
                                  max_depth=4, 
                                  random_state=1)
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)

from sklearn.model_selection import cross_val_score
out_of_sample_score = metrics.accuracy_score(y_test, y_pred)
cvs = cross_val_score(estimator = tree , X = X_train , y = y_train, cv=10,n_jobs=1)

print("Cross Validation Score: ", cvs)
print("Mean Score: ", np.mean(cvs))
print("Standard Deviation: ", np.std(cvs))
print("Out of Sample Accuracy: ", out_of_sample_score)
    
print("My name is Sa Yang")
print("My NetID is: say2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df_wine[df_wine.columns], size=2.5)
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.2,
                 stratify=y,
                 random_state=42)
# standardize the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

import numpy as np
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in
           sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


plt.bar(range(1,14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
                for i in range(len(eigen_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)

X_train_std[0].dot(w)
X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
                    plt.scatter(X_train_pca[y_train==l, 0],
                    X_train_pca[y_train==l, 1],
                    c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_std, y_train)
from sklearn import metrics
y_train_pred = lr.predict(X_train)
print( metrics.accuracy_score(y_train, y_train_pred) )
y_pred = lr.predict(X_test)
print( metrics.accuracy_score(y_test, y_pred) )


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train_std, y_train)
y_train_pred = svm.predict(X_train)
print( metrics.accuracy_score(y_train, y_train_pred) )
y_pred = svm.predict(X_test)
print( metrics.accuracy_score(y_test, y_pred) )


print('PCA:')
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

y_train_pred = lr.predict(X_train_pca)
print( metrics.accuracy_score(y_train, y_train_pred) )
y_pred = lr.predict(X_test_pca)
print( metrics.accuracy_score(y_test, y_pred) )


svm.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

plot_decision_regions(X_test_pca, y_test, classifier=svm)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

y_train_pred = svm.predict(X_train_pca)
print( metrics.accuracy_score(y_train, y_train_pred) )
y_pred = svm.predict(X_test_pca)
print( metrics.accuracy_score(y_test, y_pred) )


print('LDA:')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()

X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()

y_train_pred = lr.predict(X_train_lda)
print( metrics.accuracy_score(y_train, y_train_pred) )
y_pred = lr.predict(X_test_lda)
print( metrics.accuracy_score(y_test, y_pred) )

svm.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=svm)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()

X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=svm)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()

y_train_pred = svm.predict(X_train_lda)
print( metrics.accuracy_score(y_train, y_train_pred) )
y_pred = svm.predict(X_test_lda)
print( metrics.accuracy_score(y_test, y_pred) )


print('kPCA:')
from sklearn.decomposition import KernelPCA
range = [0.01, 0.1, 1, 2.5, 5, 15]
train_scores = []
test_scores = []
for g in range:
    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=g)
    X_train_skernpca = scikit_kpca.fit_transform(X_train_std, y_train)
    X_test_skernpca = scikit_kpca.transform(X_test_std)
    lr.fit(X_train_skernpca, y_train)
    y_train_pred = lr.predict(X_train_skernpca)
    y_pred = lr.predict(X_test_skernpca)
    train_scores.append(metrics.accuracy_score(y_train, y_train_pred))
    test_scores.append(metrics.accuracy_score(y_test, y_pred))
i = 0
while i < len(range):
    print("gamma =", range[i], "", 'Accuracy score of train set: ', train_scores[i], 'Accuracy score of test set: ', test_scores[i])
    i += 1
    
range = [0.01, 0.1, 1, 2.5, 5, 15]
train_scores = []
test_scores = []
for g in range:
    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=g)
    X_train_skernpca = scikit_kpca.fit_transform(X_train_std, y_train)
    X_test_skernpca = scikit_kpca.transform(X_test_std)
    svm.fit(X_train_skernpca, y_train)
    y_train_pred = svm.predict(X_train_skernpca)
    y_pred = svm.predict(X_test_skernpca)
    train_scores.append(metrics.accuracy_score(y_train, y_train_pred))
    test_scores.append(metrics.accuracy_score(y_test, y_pred))
i = 0
while i < len(range):
    print("gamma =", range[i], "", 'Accuracy score of train set: ', train_scores[i], 'Accuracy score of test set: ', test_scores[i])
    i += 1

print("My name is Sa Yang")
print("My NetID is: say2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

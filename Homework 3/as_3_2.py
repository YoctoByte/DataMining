from scipy.io import loadmat
from sklearn import tree
from sklearn import cross_validation
import matplotlib.pyplot as plt

filename = 'Data/wine.mat'
data = loadmat(filename)
classNames = []
attributeNames = []

for index in range(len(data['classNames'])):
    classNames.append(data['classNames'][index][0][0])
for index in range(len(data['attributeNames'][0])):
    attributeNames.append(data['attributeNames'][0][index][0])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(data['X'], data['y'])

# 3.2.1
error = []
for depth in range(2, 21):
    clf = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=100, max_depth=depth)
    clf.fit(X_train, y_train)
    good = 0
    bad = 0
    for i, sample in enumerate(X_test):
        if clf.predict(sample) == y_test[i]:
            good += 1
        else:
            bad += 1
    error.append(good/(good+bad)*100)

# plt.scatter(range(2, 21), error)
# plt.show()

# 3.2.2
averageError = []
for depth in range(2, 21):
    error = []
    for i in range(100):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(data['X'], data['y'])
        clf = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=100, max_depth=depth)
        clf.fit(X_train, y_train)
        good = 0
        bad = 0
        for i, sample in enumerate(X_test):
            if clf.predict(sample) == y_test[i]:
                good += 1
            else:
                bad += 1
        error.append(good/(good+bad)*100)
    averageError.append(sum(error)/len(error))
plt.scatter(range(2, 21), averageError)
plt.show()

from scipy.io import loadmat
from sklearn import tree

filename = 'Data/wine.mat'
data = loadmat(filename)
attributes = data['X']
wineClass = data['y']
classNames = []
attributeNames = []
for index in range(len(data['classNames'])):
    classNames.append(data['classNames'][index][0][0])
for index in range(len(data['attributeNames'][0])):
    attributeNames.append(data['attributeNames'][0][index][0])

clf = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=1)
clf.fit(attributes, wineClass)
good = 0
bad = 0
for sample in enumerate(attributes):
    if clf.predict(attributes[sample[0]]) == wineClass[sample[0]]:
        good += 1
    else:
        bad += 1
print(good, bad)

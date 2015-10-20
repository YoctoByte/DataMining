from scipy.io import loadmat
from sklearn import tree
# from sklearn.externals.six import StringIO

# 3.1.1
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
# 3.1.2
clf = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=100)
clf.fit(attributes, wineClass)
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
# 3.1.3
wineData = [6.9, 1.09, 0.06, 2.1, 0.0061, 12, 31, 0.99, 3.5, 0.64, 12]
print("The wine from assignment 3.1.3 is classified as", classNames[clf.predict(wineData)])
# 3.1.4
good = 0
bad = 0
for i, sample in enumerate(attributes):
    if clf.predict(sample) == wineClass[i]:
        good += 1
    else:
        bad += 1
print("Right predictions:", good)
print("Wrong predictions:", bad)
print("Percentage right:", good/(good+bad)*100, "%")
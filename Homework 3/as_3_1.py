from scipy.io import loadmat

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
print(classNames)
print(attributeNames)
print(attributes)
print(wineClass)

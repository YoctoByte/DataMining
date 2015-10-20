from scipy.io import loadmat

filename = 'Data/wine.mat'
data = loadmat(filename)
attributes = data['X']
tempAttributeNames = data['attributeNames']
tempClassNames = data['classNames']
classNames = []
attributeNames = []
for index in range(len(tempClassNames)):
    classNames.append(tempClassNames[index][0][0])
for index in range(len(tempAttributeNames[0])):
    attributeNames.append(tempAttributeNames[0][index][0])
print(classNames)
print(attributeNames)

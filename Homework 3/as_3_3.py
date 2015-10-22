from __future__ import print_function
import sklearn.metrics
import xlrd
import numpy as np
import pylab
def load_data():
    wb = xlrd.open_workbook('Data/classprobs.xls')
    datasheet = wb.sheet_by_index(0)
    l1=[]
    for irow in range(datasheet.nrows):
        l2=[]
        for icol in range(datasheet.ncols):
            l2.append(datasheet.cell(irow, icol).value)
        l1.append(l2)
    data = np.matrix(l1)
    return data


def rows(data,cls):
    for row in data:
        row = np.array(row)[0]
        if row[0] == cls:
            yield row


def auc(data,sample):
    m = 0.
    n = 0.
    for row in data:
        row = np.array(row)[0]
        if row[0] == 0.:
            n += 1.
        else:
            m += 1.
    t = 0.
    for i in rows(data,1.):
        for j in rows(data,0.):
            if i[sample] > j[sample]:
                t += 1.
    return (1./(m*n)) *t


def plotROC(data, clasifier):
    fpr, tpr, treshold = sklearn.metrics.roc_curve(np.array(data.T[0])[0], np.array(data.T[clasifier])[0])
    pylab.plot(fpr, tpr)
    pylab.show()


def predict(clasifierdata, treshold):
    returnval = []
    for point in clasifierdata:
        if point > treshold:
            returnval.append(1.)
        else:
            returnval.append(0.)
    return np.array(returnval)


def compute_accuricy(real_data, clasifierdata):
    fp = 0.
    fn = 0.
    tp = 0.
    tn = 0.
    for i,val in enumerate(real_data):
        if val != clasifierdata[i]:
            if val == .0:
                fp += 1.
            else:
                fn += 1.
        else:
            if val == .0:
                tn += 1.
            else:
                tp += 1.
    ft = fp + fn
    tt = tn + tp
    t  = ft + tt
    return tt/t


data = load_data()
print (np.array(data.T[0])[0])
plotROC(data, 0)
plotROC(data, 1)
plotROC(data, 1)
print(auc(data, 0))
print(auc(data, 1))
print(auc(data, 2))
predicted0 = predict(np.array(data.T[0])[0],0.5)
predicted1 = predict(np.array(data.T[1])[0],0.5)
predicted2 = predict(np.array(data.T[2])[0],0.5)
print(compute_accuricy(np.array(data.T[0])[0],predicted0))
print(compute_accuricy(np.array(data.T[0])[0],predicted1))
print(compute_accuricy(np.array(data.T[0])[0],predicted2))

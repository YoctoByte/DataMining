from __future__ import print_function
import sklearn.metrics
import xlrd
import numpy as np
import pylab
import pylab as plt


def load_data():
    wb = xlrd.open_workbook('Data/classprobs.xls')
    data_sheet = wb.sheet_by_index(0)
    l1 = []
    for irow in range(data_sheet.nrows):
        l2 = []
        for icol in range(data_sheet.ncols):
            l2.append(data_sheet.cell(irow, icol).value)
        l1.append(l2)
    data = np.matrix(l1)
    return data


def rows(data, cls):
    for row in data:
        row = np.array(row)[0]
        if row[0] == cls:
            yield row


def auc(data, sample):
    m = 0.
    n = 0.
    for row in data:
        row = np.array(row)[0]
        if row[0] == 0.:
            n += 1.
        else:
            m += 1.
    t = 0.
    for i in rows(data, 1.):
        for j in rows(data, 0.):
            if i[sample] > j[sample]:
                t += 1.
    return (1./(m*n)) * t


def plotROC(data, classifier, title):
    fpr, tpr, threshold = sklearn.metrics.roc_curve(np.array(data.T[0])[0], np.array(data.T[classifier])[0])
    fig = plt.figure()
    fig.suptitle(title)
    plt.xlabel('False positive Rate')
    plt.ylabel('True positive Rate')
    plt.axis([0, 1.0, 0, 1.0])
    pylab.plot(fpr, tpr)
    fig.savefig('output/'+title.replace(' ', '_').replace('/', '-')+'.jpg')
    pylab.show()


def predict(classifier_data, threshold):
    return_val = []
    for point in classifier_data:
        if point > threshold:
            return_val.append(1.)
        else:
            return_val.append(0.)
    return np.array(return_val)


def compute_accuracy(real_data, classifier_data):
    fp = 0.
    fn = 0.
    tp = 0.
    tn = 0.
    for i, val in enumerate(real_data):
        if val != classifier_data[i]:
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
    t = ft + tt
    return tt/t


data = load_data()
plotROC(data, 0, 'Always correct classifier')
plotROC(data, 1, 'Classifier 1')
plotROC(data, 1, 'Classifier 2')
print('AUC Always Correct classifier = ', auc(data, 0))
print('AUC Classifier 1 = ', auc(data, 1))
print('AUC Classifier 2 = ', auc(data, 2))
predicted0 = predict(np.array(data.T[0])[0], 0.5)
predicted1 = predict(np.array(data.T[1])[0], 0.5)
predicted2 = predict(np.array(data.T[2])[0], 0.5)
print('Accuracy Always Correct classifier = ', compute_accuracy(np.array(data.T[0])[0], predicted0))
print('Accuracy Classifier 1 = ', compute_accuracy(np.array(data.T[0])[0], predicted1))
print('Accuracy Classifier 2 = ', compute_accuracy(np.array(data.T[0])[0], predicted2))

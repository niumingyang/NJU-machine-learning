from numpy import *
import warnings

warnings.filterwarnings('ignore')

def sigmoid(x):
    return 1.0/(1+exp(-x))

X = list()
Y = list()

# load train set
f = open('train_set.txt', 'r')
for line in f.readlines():
    lineArr = line.strip().split(',')
    data = []
    for i in lineArr:
        data.append(float(i))
    y = data.pop()
    data.append(1.0)
    X.append(data)
    Y.append(int(y))
X = mat(X)
Y = mat(Y).transpose()
m, n = shape(X)

# gradient descent
for i in range(1, 27):
    Y_local = Y.copy()
    for j in range(0, m):
        if Y_local[j] != i:
            Y_local[j] = 0
        else:
            Y_local[j] = 1
    w = ones((n,1))
    loops = 500
    for j in range(loops):
        if j >= 0:
            gamma = 0.001
        if j > 400:
            gamma = 0.0001
        if j > 480:
            gamma = 0.00001
        h = sigmoid(X * w)
        error = (Y_local - h)
        w += gamma * X.transpose() * error
        print("classifier:", i, " loops:", j)
    if i == 1:
        W = w
    else:
        W = append(W, w, axis = 1)
W = mat(W)

#load test set
X_test = list()
Y_test = list()
f = open('test_set.txt', 'r')
for line in f.readlines():
    lineArr = line.strip().split(',')
    data = []
    for i in lineArr:
        data.append(float(i))
    y = data.pop()
    data.append(1.0)
    X_test.append(data)
    Y_test.append(int(y))
X_test = mat(X_test)
Y_test= mat(Y_test).transpose()
p, q = shape(X_test)

# predict
TP = zeros((26, 1))
FN = zeros((26, 1))
FP = zeros((26, 1))
TN = zeros((26, 1))
Y_predict = zeros((p, 1))
for i in range(0, p):
    f = []
    prob = -99999
    index = 0
    for j in range(0, 26):
        local = sum(X_test[i] * W[:, j])
        if local > prob:
            prob = local
            index = j
        f.append(local)
    for j in range(0, 26):
        if j == index:
            if f[j] <= 0: 
                FN[j] += 1
            else:
                TP[j] += 1
        else:
            if f[j] <= 0:
                TN[j] += 1
            else:
                FP[j] += 1
    Y_predict[i] = index+1
cnt = 0
for i in range(0, p):
    if Y_predict[i] == Y_test[i]:
        cnt += 1
print(Y_predict)
print("accuracy: %.2f%%" % float(cnt/p*100))
allTP = sum(TP)
allTN = sum(TN)
allFP = sum(FP)
allFN = sum(FN)
miP = float(allTP/(allTP+allFP))
miR = float(allTP/(allTP+allFN))
print("micro-P:  %.2f%%" % float(miP*100))
print("micro-R:  %.2f%%" % float(miR*100))
print("micro-F1: %.2f%%" % float(2*miP*miR/(miP+miR)*100))
P = []
R = []
for i in range(0, 26):
    P.append(float(TP[i]/(TP[i]+FP[i])))
    R.append(float(TP[i]/(TP[i]+FN[i])))
maP = sum(P)/26
maR = sum(R)/26
print("macro-P:  %.2f%%" % float(maP*100))
print("macro-R:  %.2f%%" % float(maR*100))
print("macro-F1: %.2f%%" % float(2*maP*maR/(maP+maR)*100))
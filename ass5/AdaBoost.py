import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import KFold

def load_data(trainfile, testfile, allcol, strcol):
    traindata = pd.read_csv(trainfile, header = None)
    trainnum = traindata.shape[0]
    testdata = pd.read_csv(testfile, header = None)
    testnum = testdata.shape[0]
    alldata = pd.concat([traindata, testdata], ignore_index = True)
    alldata.columns = allcol
    alldata.loc[alldata['label']==' >50K', 'label'] = 1
    alldata.loc[alldata['label']==' >50K.', 'label'] = 1
    alldata.loc[alldata['label']==' <=50K', 'label'] = 0
    alldata.loc[alldata['label']==' <=50K.', 'label'] = 0
    for i in strcol:
        alldata = alldata.join(pd.get_dummies(alldata[i], prefix = i))
    alldata = alldata.drop(strcol, axis = 1)
    label = alldata['label'].values.astype('int')
    sample = alldata.drop(['label'], axis = 1).values
    return sample[0:trainnum],  label[0:trainnum], sample[trainnum:], label[trainnum:]

class AdaBoost:
    def __init__(self, T):
        self.T = T
        self.base = []
        self.alpha = []
    
    def train(self, data, label):
        m = label.shape[0]
        D = np.ones(m)/m
        for i in range(self.T):   
            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(data, label, sample_weight=D)
            h = tree.predict(data)
            diff = h - label
            diff[diff != 0] = 1
            error = np.dot(D , diff)
            if(error>0.5):
                break
            if(error == 0):
                alpha_t = 0
                self.base.append(tree)
                self.alpha.append(alpha_t)
                return
            else:
                alpha_t = 0.5*np.log((1-error)/error)
            diff[diff == 0] = -1
            D = D * np.exp(diff*alpha_t)
            D = D/np.sum(D)
            D = np.array(D)
            self.base.append(tree)
            self.alpha.append(alpha_t)
            
    def predict(self, data):    
        res = np.zeros(data.shape[0])
        for i in range(self.T):
            tree = self.base[i]
            alpha_t = self.alpha[i]
            h_t = tree.predict(data)
            h_t[h_t == 0] = -1
            res = res + h_t * alpha_t
        res[res < 0] = 0
        res[res > 0] = 1
        return res

    def predict_with_prob(self, data):    
        res = np.zeros(data.shape[0])
        alpha_sum = np.sum(self.alpha)
        for i in range(self.T):
            tree = self.base[i]
            alpha_t = self.alpha[i]
            h_t = tree.predict(data)
            res = res + h_t * alpha_t/alpha_sum
        return res
        
if __name__ == '__main__':
    #load data
    allcol = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
             'marital-status', 'occupation', 'relationship', 'race', 'sex', 
             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
    strcol = ['workclass', 'education', 'marital-status', 'occupation', 
             'relationship', 'race', 'sex' , 'native-country']
    train_x, train_y, test_x, test_y = load_data("adult.data", "adult.test", allcol, strcol)
    '''
    # 5-fold
    x = []
    y = []
    for T in range(1, 51):
        model = AdaBoost(T)
        x.append(T)
        kf = KFold(n_splits=5)
        AUC = 0
        acc = 0
        for train_index, test_index in kf.split(train_x):
            val_train_x, val_test_x = train_x[train_index], train_x[test_index]
            val_train_y, val_test_y = train_y[train_index], train_y[test_index]
            model.train(val_train_x, val_train_y)
            res = model.predict(val_test_x)
            res_with_prob = model.predict_with_prob(val_test_x)
            AUC += metrics.roc_auc_score(val_test_y, res_with_prob)
            acc += metrics.accuracy_score(val_test_y, res)
        AUC /= 5
        acc /= 5
        y.append(AUC)
        print("K-Fold validation: base classifier number = ", T, ", AUC = ", AUC, ", accuracy = ", acc)

    plt.xlabel("number of base classifiers")
    plt.ylabel("AUC")
    plt.title("Adaboost")
    plt.plot(x, y, color='blue')
    plt.savefig("Adaboost.jpg")
    plt.show()
    '''
    #create AdaBoost model
    T = 25
    model = AdaBoost(T)
    #train and test
    model.train(train_x, train_y)
    res = model.predict(test_x)
    res_with_prob = model.predict_with_prob(test_x)
    AUC = metrics.roc_auc_score(test_y, res_with_prob)
    acc = metrics.accuracy_score(test_y, res)
    print("base classifier number = ", T, ", AUC = ", AUC, ", accuracy = ", acc)
    
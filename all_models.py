from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, average_precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np


# Overview
# This code contains different classification models

class AllModels:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    # Label encoding the target values.
    def preprocessing(self):
        x = self.data
        y = self.label
        lb = LabelEncoder()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0,
                                                            shuffle=True)
        lb.fit(list(y_train))
        y_test = lb.transform(y_test)
        y_train = lb.transform(y_train)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        return x_train, x_test, y_train, y_test

    # bagging classifier
    def bag(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        classifier = BaggingClassifier()
        y_pred = classifier.fit(x_train, y_train).predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print('Bagging Classifier:- ', acc)
        conf = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # Light GBM classifier
    def lgbm(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        classifier = LGBMClassifier()
        y_pred = classifier.fit(x_train, y_train).predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print('Light GBM Classifier:- ', acc)
        conf = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # XG boost classifier
    def xg(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        classifier = XGBClassifier()
        y_pred = classifier.fit(x_train, y_train).predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print('XG boost Classifier:- ', acc)
        conf = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # Ridge classifier
    def ridge(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        classifier = RidgeClassifier()
        y_pred = classifier.fit(x_train, y_train).predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print('Ridge Classifier:- ', acc)
        conf = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # Passive aggressive classifer
    def passive(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        classifier = PassiveAggressiveClassifier()
        y_pred = classifier.fit(x_train, y_train).predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print('Passive Aggressive Classifier:- ', acc)
        conf = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # Extra tree classifier
    def extra_tree(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        extra_tree_model = ExtraTreeClassifier()
        y_pred = extra_tree_model.fit(x_train, y_train).predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print('Extra Tree Classifier:- ', acc)
        conf = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # Gaussian
    def gauss_model(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        gaussian_model = GaussianNB()
        y_pred_gaussian = gaussian_model.fit(x_train, y_train).predict(x_test)
        acc_gaussian = accuracy_score(y_test, y_pred_gaussian)
        print('Gaussian:- ', acc_gaussian)
        conf = confusion_matrix(y_test, y_pred_gaussian)
        f1 = f1_score(y_test, y_pred_gaussian, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # Binomial
    def bino_model(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        binomial_model = BernoulliNB()
        y_pred_binomial = binomial_model.fit(x_train, y_train).predict(x_test)
        acc_binomial = accuracy_score(y_test, y_pred_binomial)
        print('Binomial:- ', acc_binomial)
        conf = confusion_matrix(y_test, y_pred_binomial)
        f1 = f1_score(y_test, y_pred_binomial, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # Multinomial
    def multi_model(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        multinomial_model = MultinomialNB()

        y_pred_multinomial = multinomial_model.fit(x_train, y_train).predict(x_test)
        acc_multinomial = accuracy_score(y_test, y_pred_multinomial)
        print('Multinomial:- ', acc_multinomial)
        conf = confusion_matrix(y_test, y_pred_multinomial)
        f1 = f1_score(y_test, y_pred_multinomial, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # Stochastic gradient descent
    def stoc_model(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        stochastic_model = SGDClassifier(loss='modified_huber', shuffle=True, random_state=101,
                                         max_iter=1000)
        y_pred_stochastic = stochastic_model.fit(x_train, y_train).predict(x_test)
        acc_stochastic = accuracy_score(y_test, y_pred_stochastic)
        print('Stochastic Gradient Descent:- ', acc_stochastic)
        conf = confusion_matrix(y_test, y_pred_stochastic)
        f1 = f1_score(y_test, y_pred_stochastic, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # Decision tree
    def dec_tree(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        dt_model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=7)
        y_pred_DT = dt_model.fit(x_train, y_train).predict(x_test)
        acc_DT = accuracy_score(y_test, y_pred_DT)
        print('Decision Tree:- ', acc_DT)
        conf = confusion_matrix(y_test, y_pred_DT)
        f1 = f1_score(y_test, y_pred_DT, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # Random Forest
    def rand_forest(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        rf_model = RandomForestClassifier(n_jobs=2, criterion='entropy', n_estimators=55,
                                          random_state=23)
        y_pred_RF = rf_model.fit(x_train, y_train).predict(x_test)
        acc_RF = accuracy_score(y_test, y_pred_RF)
        print('Random Forest:- ', acc_RF)
        conf = confusion_matrix(y_test, y_pred_RF)
        f1 = f1_score(y_test, y_pred_RF, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # K nearest neighbors
    def k_nearest(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        knn = KNeighborsClassifier(n_neighbors=5)
        y_pred_KNN = knn.fit(x_train, y_train).predict(x_test)
        acc_KNN = accuracy_score(y_test, y_pred_KNN)
        print('KNN:- ', acc_KNN)
        conf = confusion_matrix(y_test, y_pred_KNN)
        f1 = f1_score(y_test, y_pred_KNN, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # logistic regression
    def log_reg(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        lr = LogisticRegression(C=0.50, multi_class='ovr', max_iter=10000, solver='lbfgs')
        y_pred_LR = lr.fit(x_train, y_train).predict(x_test)
        acc_LR = accuracy_score(y_test, y_pred_LR)
        print('Log reg:- ', acc_LR)
        conf = confusion_matrix(y_test, y_pred_LR)
        f1 = f1_score(y_test, y_pred_LR, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # SVM with linear kernel
    def svm_linear(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        svL = svm.SVC(kernel='linear', gamma='auto')
        y_pred_svL = svL.fit(x_train, y_train).predict(x_test)
        acc_svL = accuracy_score(y_test, y_pred_svL)
        print('SVM linear:- ', acc_svL)
        conf = confusion_matrix(y_test, y_pred_svL)
        f1 = f1_score(y_test, y_pred_svL, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # SVM with rbf kernel
    def svm_rbf(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        svR = svm.SVC(kernel='rbf', gamma='auto')
        y_pred_svR = svR.fit(x_train, y_train).predict(x_test)
        acc_svR = accuracy_score(y_test, y_pred_svR)
        print('SVM RBF:- ', acc_svR)
        conf = confusion_matrix(y_test, y_pred_svR)
        f1 = f1_score(y_test, y_pred_svR, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # SVM with poly kernel
    def svm_poly(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        svP = svm.SVC(kernel='poly', gamma='auto')
        y_predsvP = svP.fit(x_train, y_train).predict(x_test)
        accsvP = accuracy_score(y_test, y_predsvP)
        print('SVM Poly:- ', accsvP)
        conf = confusion_matrix(y_test, y_predsvP)
        f1 = f1_score(y_test, y_predsvP, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # Adaboost
    def adaboost_model(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        ada_model = AdaBoostClassifier()
        y_pred_ada = ada_model.fit(x_train, y_train).predict(x_test)
        acc_ada = accuracy_score(y_test, y_pred_ada)
        print('AdaBoost Classifier:- ', acc_ada)
        conf = confusion_matrix(y_test, y_pred_ada)
        f1 = f1_score(y_test, y_pred_ada, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # MultiLayer perceptron
    def mlpc_nn(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        mlpc_model = MLPClassifier()
        y_pred_mlpc = mlpc_model.fit(x_train, y_train).predict(x_test)
        acc_mlpc = accuracy_score(y_test, y_pred_mlpc)
        print('MLPC Neural Network:- ', acc_mlpc)
        conf = confusion_matrix(y_test, y_pred_mlpc)
        f1 = f1_score(y_test, y_pred_mlpc, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # Quadratic Discriminant Analysis
    def qda(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        QDA_model = QuadraticDiscriminantAnalysis()
        y_pred_qda = QDA_model.fit(x_train, y_train).predict(x_test)
        acc_qda = accuracy_score(y_test, y_pred_qda)
        print('Quadratic Discriminant Analysis:- ', acc_qda)
        conf = confusion_matrix(y_test, y_pred_qda)
        f1 = f1_score(y_test, y_pred_qda, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)

    # Linear Discriminant Analysis
    def lda(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        LDA_model = LinearDiscriminantAnalysis()
        y_pred_lda = LDA_model.fit(x_train, y_train).predict(x_test)
        acc_lda = accuracy_score(y_test, y_pred_lda)
        print('Linear Discriminant Analysis:- ', acc_lda)
        conf = confusion_matrix(y_test, y_pred_lda)
        f1 = f1_score(y_test, y_pred_lda, average='micro')
        print('and its f1 score- ', f1)
        print('confusion matrix: \n', conf)


# if __name__ == '__main__':
#
#
#     obj = AllModels(data_main, labels)
#     obj.xg()
#     obj.dec_tree()

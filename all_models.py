import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, average_precision_score, \
    r2_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier, \
    LinearRegression, RANSACRegressor, ARDRegression, HuberRegressor, LogisticRegression, \
    LogisticRegressionCV, SGDRegressor, TheilSenRegressor, PassiveAggressiveRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, \
    RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn import svm
from xgboost import XGBClassifier, XGBRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, DecisionTreeRegressor, \
    ExtraTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

np.random.seed(3)


# Overview
# This code contains different classification models

class ClassificationModels:
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


class RegressionModels:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def preprocessing(self):
        x = self.data
        y = self.label
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0,
                                                            shuffle=True)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        return x_train, x_test, y_train, y_test

    def printing(self, y_test, y_pred, name):
        mape = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print('{} RMSE- {}'.format(name, round(rmse, 4)))
        print('{} MAPE- {}%'.format(name, round(mape, 4)))
        r2 = r2_score(y_test, y_pred)
        print('{} R2 Score- {}'.format(name, round(r2, 4)))

    # Adaboost
    def adaboost_regressor(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        model = AdaBoostRegressor()
        y_pred = model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Adaboost')

    # Linear
    def linear_regressor(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        model = LinearRegression()
        y_pred = model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Linear Regression')

    # Stochastic
    def sgd_regressor(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        model = SGDRegressor()
        y_pred = model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Stochastic Gradient Descent')

    # Decision Tree
    def dec_tree_regressor(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        model = DecisionTreeRegressor()
        y_pred = model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Decision Tree')

    # Random Forest
    def random_forest_regressor(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        model = RandomForestRegressor()
        y_pred = model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Random Forest')

    # RANSAC
    def ransac_regressor(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        model = RANSACRegressor()
        y_pred = model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'RANSAC')

    # ARD
    def ard_regressor(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        model = ARDRegression()
        y_pred = model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'ARD')

    # Huber
    def huber_regressor(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        model = HuberRegressor()
        y_pred = model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Huber')

    # Theilsen
    def theilsen_regressor(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        model = TheilSenRegressor()
        y_pred = model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Theilsen')

    # Passive Aggressive
    def passive_aggressive_regressor(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        model = PassiveAggressiveRegressor()
        y_pred = model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Passive Aggressive')

    # MLP
    def mlp_regressor(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        model = MLPRegressor()
        y_pred = model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Multi Layer Perceptron')

    # Bagging
    def bagging_regressor(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        model = BaggingRegressor()
        y_pred = model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Bagging')

    # XG Boost
    def xgb_regressor(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        model = XGBRegressor()
        y_pred = model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'XG Boost')

    # Light GBM
    def lgb_regressor(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        model = LGBMRegressor()
        y_pred = model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Light GBM')

    # KNN
    def knn_regressor(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        model = KNeighborsRegressor()
        y_pred = model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'KNN')

    # Extra Tree
    def extra_tree_regressor(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        model = ExtraTreeRegressor()
        y_pred = model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Extra Tree')


if __name__ == '__main__':
    # from sklearn import datasets
    #
    # bh = datasets.load_boston()
    # data = bh.data
    # labels = bh.target
    # obj = RegressionModels(data, labels)
    # obj.xgb_regressor()
    # obj.lgb_regressor()
    # obj.knn_regressor()
    # obj.extra_tree_regressor()

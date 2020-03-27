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
# This code contains different classification adn regression models

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

    def printing(self, y_test, y_pred, name):
        acc = accuracy_score(y_test, y_pred)
        print('{} Classifier:- {}%'.format(name, round(acc * 100, 4)))
        conf = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='micro')
        print('{} f1 score- {}'.format(name, round(f1, 4)))
        print('{} confusion matrix: \n{}'.format(name, conf))

    # bagging classifier
    def bag(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        classifier = BaggingClassifier()
        y_pred = classifier.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Bagging')

    # Light GBM classifier
    def lgbm(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        classifier = LGBMClassifier()
        y_pred = classifier.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Light GBM')

    # XG boost classifier
    def xg(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        classifier = XGBClassifier()
        y_pred = classifier.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'XG Boost')

    # Ridge classifier
    def ridge(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        classifier = RidgeClassifier()
        y_pred = classifier.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Ridge')

    # Passive aggressive classifer
    def passive(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        classifier = PassiveAggressiveClassifier()
        y_pred = classifier.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Passive')

    # Extra tree classifier
    def extra_tree(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        extra_tree_model = ExtraTreeClassifier()
        y_pred = extra_tree_model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Extra Tree')

    # Gaussian
    def gauss_model(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        gaussian_model = GaussianNB()
        y_pred = gaussian_model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Gaussian')

    # Binomial
    def bino_model(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        binomial_model = BernoulliNB()
        y_pred = binomial_model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Binomial')

    # Multinomial
    def multi_model(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        multinomial_model = MultinomialNB()
        y_pred = multinomial_model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Multinomial')

    # Stochastic gradient descent
    def stoc_model(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        stochastic_model = SGDClassifier(loss='modified_huber', shuffle=True, random_state=101,
                                         max_iter=1000)
        y_pred = stochastic_model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Stochastic Gradient Descent')

    # Decision tree
    def dec_tree(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        dt_model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=7)
        y_pred = dt_model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Decision Tree')

    # Random Forest
    def rand_forest(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        rf_model = RandomForestClassifier(n_jobs=2, criterion='entropy', n_estimators=55,
                                          random_state=23)
        y_pred = rf_model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Random Forest')

    # K nearest neighbors
    def k_nearest(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        knn = KNeighborsClassifier(n_neighbors=5)
        y_pred = knn.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'KNN')

    # logistic regression
    def log_reg(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        lr = LogisticRegression(C=0.50, multi_class='ovr', max_iter=10000, solver='lbfgs')
        y_pred = lr.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Logistic Regression')

    # SVM with linear kernel
    def svm_linear(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        svL = svm.SVC(kernel='linear', gamma='auto')
        y_pred = svL.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'SVM Linear')

    # SVM with rbf kernel
    def svm_rbf(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        svR = svm.SVC(kernel='rbf', gamma='auto')
        y_pred = svR.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'SVM RBF')

    # SVM with poly kernel
    def svm_poly(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        svP = svm.SVC(kernel='poly', gamma='auto')
        y_pred = svP.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'SVM Poly')

    # Adaboost
    def adaboost_model(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        ada_model = AdaBoostClassifier()
        y_pred = ada_model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'Adaboost')

    # MultiLayer perceptron
    def mlpc_nn(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        mlpc_model = MLPClassifier()
        y_pred = mlpc_model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'mlpc')

    # Quadratic Discriminant Analysis
    def qda(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        QDA_model = QuadraticDiscriminantAnalysis()
        y_pred = QDA_model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'QDA')

    # Linear Discriminant Analysis
    def lda(self):
        x_train, x_test, y_train, y_test = self.preprocessing()
        LDA_model = LinearDiscriminantAnalysis()
        y_pred = LDA_model.fit(x_train, y_train).predict(x_test)
        self.printing(y_test, y_pred, 'LDA')


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
    # bh = datasets.load_iris()
    # data = bh.data
    # labels = bh.target
    # obj = ClassificationModels(data, labels)
    # obj.dec_tree()
    # obj.rand_forest()
    # obj.lgbm()
    # obj.gauss_model()

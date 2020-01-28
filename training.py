#libraries 
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score 
from sklearn.metrics import f1_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import make_scorer 
from sklearn.metrics import recall_score


def tt_split(df, features, target, test_size, random_state):
    
    """
    
    Train test split function.
    
    Input: 
        - df: pandas dataframe of data
        - features: list of features names
        - target: name of target column 
        - test_size: size of the set
        - random_state: random seed for train/test split
        
    Return: 
        - X_train, y_train, X_test, y_test
        
    """
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(df[features].values, df[target].values, 
                                                        test_size=test_size, 
                                                        shuffle=True, random_state=random_state)
    
    # use oversampling method to balance classes:
    sm = SMOTETomek(random_state=random_state)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    
    return X_train, X_test, y_train, y_test


def cross_validate(model, X_train, y_train, params, param_grid, cv, scorer): 
    
    """
    
    Cross validation of specified parameters.
    
    Input:
        - model: classification model 
        - X_train: training set
        - y_train: labels of the training set 
        - params: dict of initial values of parameters
        - param_grid: grid of possible parameters
        - cv: number of fold for cross validation
        - scorer: scorer used to evaluate model performance
        
    Return: 
        - params: updated dict of parameters
        
    """
        
    # cross validate and search optimal params
    gs = GridSearchCV(estimator=model, param_grid=param_grid,
                      cv=cv, n_jobs=-1, verbose=True, return_train_score=True, 
                      scoring=scorer)
    gs.fit(X_train, y_train)
        
    # update best params
    for best_param in param_grid.keys(): 
        params[best_param] = gs.best_params_[best_param]
            
    # return dict of updated best params
    return params



def train_GBT(X_train, y_train, cv, scorer): 
    
    """
    
    Train xgboost cross-validating main parameters
    
    Input: 
        - X_train: training set
        - y_train: training set labels
        - cv: number of fold for cross validation
        - scorer: scorer used to evaluate model performance
        
    Return: 
        - model fitted with optimal parameters
        
    """
    
    # initial params
    params = {'eta'              : 0.001,
              'objective'        : 'multi:softmax', 
              'num_class'        : 3,
              'max_depth'        : 5,
              'min_child_weight' : 1,
              'n_estimators'     : 100,
              'gamma'            : 0,
              'subsample'        : 0.8,
              'colsample_bytree' : 0.8,
              'scale_pos_weight' : 1, 
              'reg_alpha'        : 0}
    
    # n. of trees
    param_grid = {'n_estimators': [i for i in np.arange(10,100,10)]}
    params     = cross_validate(XGBClassifier(**params), 
                                X_train, y_train, params, param_grid, cv, scorer)
    
    # max_depth and min_child_weight
    param_grid = {'max_depth': range(3, 10, 1), 'min_child_weight': range(1, 6, 1)}
    params     = cross_validate(XGBClassifier(**params), 
                                X_train, y_train, params, param_grid, cv, scorer)
    
    # gamma
    param_grid = {'gamma': [i / 10.0 for i in range(1, 5)]}
    params     = cross_validate(XGBClassifier(**params), 
                                X_train, y_train, params, param_grid, cv, scorer)
    
    # subsample and col_sample_bytree
    param_grid = {'subsample'       : [i / 10.0 for i in range(6, 10)], 
                  'colsample_bytree': [i / 10.0 for i in range(6, 10)]}
    params     = cross_validate(XGBClassifier(**params), 
                                X_train, y_train, params, param_grid, cv, scorer)

    # reg_alpha
    param_grid = {'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]}
    params     = cross_validate(XGBClassifier(**params), 
                                X_train, y_train, params, param_grid, cv, scorer)

    # fit the final model 
    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)
    
    return clf

   
def train_model(model, X_train, y_train, cv, scorer, params, param_grid):
    
    """
    
    Train a general model using cross validation 
    
    Input: 
        - model: ML model 
        - X_train: training set
        - y_train: training set labels
        - cv: number of fold for cross validation
        - scorer: scorer used to evaluate model performance
        - params: dict of initial parameters
        - param_grid: grid of possible parameters
        
    Return: 
        - model fitted with optimal parameters
        
    """

    # cv
    optimal_params = cross_validate(model, X_train, y_train, params, param_grid, cv, scorer)

    # fit the final model 
    clf = model.set_params(**optimal_params)
    clf.fit(X_train, y_train)
    
    # return 
    return clf


def scoring(y_true, y_pred):
    
    """
    
    Evaluate the performance of a model using several metrics 
    
    Input: 
        - y_true: list of true labels
        - y_pred: list of predicted labels
        
    Return: 
        - precision score, balanced accuracy, recall, f1 score
        
    """
    
    prec    =  precision_score(y_true, y_pred, average='weighted')
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    rec     = recall_score(y_true, y_pred, average='weighted')
    f1score = f1_score(y_true, y_pred, average='weighted')
    
    return prec, bal_acc, rec, f1score
    
    
def dummy_score(X_train, y_train, X_test, y_test, strategy, label=None):
    
    """
    
    Evaluate the performance of a random classificator using several metrics 
    
    Input: 
        - X_train: training set
        - y_train: training set labels
        - X_test: test set
        - y_test: test set labels
        - strategy: strategy for random clf ('uniform', 'stratified', 'constant', 'most_frequent')
        - label: constant label for prediction (only used when strategy='constant')
        
    Return: 
        - precision score, balanced accuracy, recall, f1 score of random clf
        
    """
    
    dummyclf = DummyClassifier(strategy=strategy, constant=label)
    dummyclf.fit(X_train, y_train)
    
    # get statitics of rnd performance
    if strategy=='stratified' or strategy=='uniform':
        prec_, bal_acc_, rec_, f1score_ = [], [], [], []
        # iterate
        for i in range(100):
            p, ba, r, f1s = scoring(y_test, dummyclf.predict(X_test))
            prec_.append(p)
            bal_acc_.append(ba)
            rec_.append(r)
            f1score_.append(f1s)
            
        # average
        prec = np.mean(prec_)
        bal_acc = np.mean(bal_acc_)
        rec = np.mean(rec_)
        f1score = np.mean(f1score_)
        
    else:
        prec, bal_acc, rec, f1score = scoring(y_test, dummyclf.predict(X_test))
        
    return prec, bal_acc, rec, f1score


def main():
    
    # import data 
    path     = '../data/dataset1.csv'
    df       = pd.read_csv(path, sep=',')
    features = ['perceived susceptibility', 'disease score', 'peak',
                'perceived severity', 'efficacy', 'gender', 'flu frequency',
                'age', 'information', 'prevalence', 'elderly',
                'children', 'contacts', 'preventive', 'vaccination last year',
                'vaccination', 'info seeking', 'flu', 'smoke',
                'diet', 'allergy', 'disease', 'public transport']
    target   = 'be_chg_type'
    
    # parameters
    test_size    = 0.30
    random_state = 35
    cv           = 10
    scorer       =  make_scorer(precision_score, average='weighted')

    # train test split
    print('Train test split...')
    X_train, X_test, y_train, y_test = tt_split(df, features, target, test_size, random_state)

    # random benchmark
    print('Random Predictions...')
    print('RND - uniform')
    prec, bal_acc, rec, f1score = dummy_score(X_train, y_train, X_test, y_test, 'uniform')
    print('\tPrecision        : %.3f' % prec)
    print('\tBalanced accuracy: %.3f' % bal_acc)
    print('\tRecall           : %.3f' % rec)
    print('\tF1 score         : %.3f' % f1score)
    print('RND - stratified')
    prec, bal_acc, rec, f1score = dummy_score(X_train, y_train, X_test, y_test, 'stratified')
    print('\tPrecision        : %.3f' % prec)
    print('\tBalanced accuracy: %.3f' % bal_acc)
    print('\tRecall           : %.3f' % rec)
    print('\tF1 score         : %.3f' % f1score)
    print('RND - most frequent')
    prec, bal_acc, rec, f1score = dummy_score(X_train, y_train, X_test, y_test, 'most_frequent')
    print('\tPrecision        : %.3f' % prec)
    print('\tBalanced accuracy: %.3f' % bal_acc)
    print('\tRecall           : %.3f' % rec)
    print('\tF1 score         : %.3f' % f1score)
    print('RND - constant (-1)')
    prec, bal_acc, rec, f1score = dummy_score(X_train, y_train, X_test, y_test, 'constant', label=-1)
    print('\tPrecision        : %.3f' % prec)
    print('\tBalanced accuracy: %.3f' % bal_acc)
    print('\tRecall           : %.3f' % rec)
    print('\tF1 score         : %.3f' % f1score)
    print('RND - constant (0)')
    prec, bal_acc, rec, f1score = dummy_score(X_train, y_train, X_test, y_test, 'constant', label=0)
    print('\tPrecision        : %.3f' % prec)
    print('\tBalanced accuracy: %.3f' % bal_acc)
    print('\tRecall           : %.3f' % rec)
    print('\tF1 score         : %.3f' % f1score)
    print('RND - constant (+1)')
    prec, bal_acc, rec, f1score = dummy_score(X_train, y_train, X_test, y_test, 'constant', label=+1)
    print('\tPrecision        : %.3f' % prec)
    print('\tBalanced accuracy: %.3f' % bal_acc)
    print('\tRecall           : %.3f' % rec)
    print('\tF1 score         : %.3f' % f1score)

    #xgboost
    print('\n\nTraining XGBoost...')
    xgbmodel = train_GBT(X_train, y_train, cv, scorer)
    prec, bal_acc, rec, f1score = scoring(y_test, xgbmodel.predict(X_test))
    print('XGBoost performance:')
    print('\tPrecision        : %.3f' % prec)
    print('\tBalanced accuracy: %.3f' % bal_acc)
    print('\tRecall           : %.3f' % rec)
    print('\tF1 score         : %.3f' % f1score)

    # random forest
    print('\n\nTraining Random Forest...')
    # grid of possible parameters
    rf_grid = {'n_estimators'     : [i for i in np.arange(10,100,10)],
               'max_features'     : ['auto', 'sqrt'],
               'max_depth'        : range(3, 10, 1),
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf' : [1, 2, 4],
               'bootstrap'        : [True, False]}
    rf      = RandomForestClassifier()
    rfmodel = train_model(rf, X_train, y_train, cv, scorer, rf.get_params(), rf_grid)
    prec, bal_acc, rec, f1score = scoring(y_test, rfmodel.predict(X_test))
    print('Random Forest performance:')
    print('\tPrecision        : %.3f' % prec)
    print('\tBalanced accuracy: %.3f' % bal_acc)
    print('\tRecall           : %.3f' % rec)
    print('\tF1 score         : %.3f' % f1score)

    # svm (linear kernel)
    print('\n\nTraining SVM (Linear Kernel)...')
    # grid of possible parameters
    svmlin_grid = {'kernel': ['linear'], 'C': [1, 5, 10]}
    svm         = SVC()
    svmlinmodel = train_model(svm, X_train, y_train, cv, scorer, svm.get_params(), svmlin_grid)
    prec, bal_acc, rec, f1score = scoring(y_test, svmlinmodel.predict(X_test))
    print('SVM (Linear Kernel) performance:')
    print('\tPrecision        : %.3f' % prec)
    print('\tBalanced accuracy: %.3f' % bal_acc)
    print('\tRecall           : %.3f' % rec)
    print('\tF1 score         : %.3f' % f1score)

    # svm (rbf kernel)
    print('\n\nTraining SVM (RBF Kernel)...')
    # grid of possible parameters
    svmrbf_grid = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 5, 10, 20]}
    svm         = SVC()
    svmrbfmodel = train_model(svm, X_train, y_train, cv, scorer, svm.get_params(), svmrbf_grid)
    prec, bal_acc, rec, f1score = scoring(y_test, svmrbfmodel.predict(X_test))
    print('SVM (RBF Kernel) performance:')
    print('\tPrecision        : %.3f' % prec)
    print('\tBalanced accuracy: %.3f' % bal_acc)
    print('\tRecall           : %.3f' % rec)
    print('\tF1 score         : %.3f' % f1score)

    # lg 
    print('\n\nTraining Logistic Regression...')
    # grid of possible parameters
    lg_grid = {'C'            : np.logspace(-3,3,100),
               'solver'       : ['newton-cg', 'sag', 'saga', 'lbfgs'],
               'fit_intercept': [True, False]}
    lg     = LogisticRegression(multi_class='multinomial')
    lgmodel = train_model(lg, X_train, y_train, cv, scorer, lg.get_params(), lg_grid)
    prec, bal_acc, rec, f1score = scoring(y_test, lgmodel.predict(X_test))
    print('Logistic Regression performance:')
    print('\tPrecision        : %.3f' % prec)
    print('\tBalanced accuracy: %.3f' % bal_acc)
    print('\tRecall           : %.3f' % rec)
    print('\tF1 score         : %.3f' % f1score)
    
#run
if __name__=='__main__':
    main()
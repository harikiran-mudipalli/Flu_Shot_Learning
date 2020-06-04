import numpy as np
import model

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import warnings
warnings.filterwarnings("ignore")

class ModelTune:
    def __init__(self, X, y):
        X, _, y, _ = model.scale_and_split(X, y, scale=True)
        self.X = X
        self.y = y
        
        
    def logreg(self):
        print("*********** Logistic Regression ***********")
        
        Model = LogisticRegression()
        param_grid = [    
            {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
            'C' : np.logspace(-4, 4, 20),
            'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
            'max_iter' : [100, 1000,2500, 5000]
            }
        ]

        clf = RandomizedSearchCV(Model,
                                 param_distributions=param_grid,
                                 n_iter=5,
                                 scoring='roc_auc',
                                 n_jobs=-1,
                                 cv=10,
                                 verbose=3)
        best_clf = clf.fit(self.X,self.y)

        print (f'Accuracy during search - : {best_clf.score(self.X,self.y):.3f}')
        params = best_clf.best_estimator_.get_params()
        estimator = Model.set_params(**params)
        return estimator
    
    def GBC(self):
        print("*********** Gradient Boosting Classifier ***********")
        Model = GradientBoostingClassifier()
        param_grid = [
            {
                'loss' : ['deviance', 'exponential'],
                'n_estimators': np.arange(10,200,5), #[10, 40, 70, 80, 90, 100, 120, 140, 150],
                'learning_rate': np.arange(0,1,0.01), #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                'subsample' : np.arange(0.1,1,0.05), #[0.1,0.3,0.5,0.7,0.9,1],
                'min_samples_split' : [2,4,5,7,9,10],
                'min_samples_leaf' : [1,2,3,4,5],
                'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                'max_features' : ['auto', 'sqrt', 'log2']
            }
        ]
        
        clf = RandomizedSearchCV(Model,
                                 param_distributions=param_grid,
                                 n_iter=5,
                                 scoring='roc_auc',
                                 n_jobs=-1,
                                 cv=10,
                                 verbose=3)
        best_clf = clf.fit(self.X,self.y)

        print (f'Accuracy during search - : {best_clf.score(self.X,self.y):.3f}')
        params = best_clf.best_estimator_.get_params()
        estimator = Model.set_params(**params)
        return estimator
    
    def knn(self):
        print("*********** K neighbors Classifier ***********")
        Model = KNeighborsClassifier()
        param_grid = {
                      'n_neighbors': range(1,51),
                      'weights' : ['uniform', 'distance'],
                      'metric':["euclidean", "manhattan"]
                     }
        
        clf = RandomizedSearchCV(Model,
                                 param_distributions=param_grid,
                                 n_iter=5,
                                 scoring='roc_auc',
                                 n_jobs=-1,
                                 cv=10,
                                 verbose=3)
        best_clf = clf.fit(self.X,self.y)

        print (f'Accuracy during search - : {best_clf.score(self.X,self.y):.3f}')
        params = best_clf.best_estimator_.get_params()
        estimator = Model.set_params(**params)
        return estimator
    
    def dtc(self):
        print("*********** Decision Tree Classifier ***********")
        Model = DecisionTreeClassifier()
        param_grid = {"max_depth": np.random.randint(1, (len(X_train.columns)*.85),20),
                      "max_features": np.random.randint(3, len(X_train.columns),20),
                      "min_samples_leaf": [2,3,4,5,6],
                      "criterion": ["gini", "entropy"],
                     }
        clf = RandomizedSearchCV(Model,
                                 param_distributions=param_grid,
                                 n_iter=5,
                                 scoring='roc_auc',
                                 n_jobs=-1,
                                 cv=10,
                                 verbose=3)
        best_clf = clf.fit(self.X,self.y)

        print (f'Accuracy during search - : {best_clf.score(self.X,self.y):.3f}')
        params = best_clf.best_estimator_.get_params()
        estimator = Model.set_params(**params)
        return estimator
    
    def rfc(self):
        print("*********** Random Forest Classifier ***********")
        Model = RandomForestClassifier()
        param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                      'min_samples_split': [2, 5, 7, 9, 10],
                      'min_samples_leaf' : [1, 2, 4],
                      'max_features' : ['auto', 'sqrt', 'log2'],
                      'bootstrap': [True, False]
                     }
        
        clf = RandomizedSearchCV(Model,
                                 param_distributions=param_grid,
                                 n_iter=5,
                                 scoring='roc_auc',
                                 n_jobs=-1,
                                 cv=10,
                                 verbose=3)
        best_clf = clf.fit(self.X,self.y)

        print (f'Accuracy during search - : {best_clf.score(self.X,self.y):.3f}')
        params = best_clf.best_estimator_.get_params()
        estimator = Model.set_params(**params)
        return estimator
        
    def ada(self):
        print("*********** Ada Boost Classifier ***********")
        Model = RandomForestClassifier()
        param_grid = {'n_estimators':  np.arange(10,200,5), #[10, 40, 70, 80, 90, 100, 120, 140,150],
                      'learning_rate': np.arange(0,1,0.01), #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                      'algorithm' : ["SAMME", "SAMME.R"]
                     } 
        clf = RandomizedSearchCV(Model,
                                 param_distributions=param_grid,
                                 n_iter=5,
                                 scoring='roc_auc',
                                 n_jobs=-1,
                                 cv=10,
                                 verbose=3)
        best_clf = clf.fit(self.X,self.y)
        
    
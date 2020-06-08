import lightgbm as lgb
from lightgbm import LGBMClassifier
from lightgbm import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


class LGBM_classifier(LGBMClassifier):
    def __init__(self,objective='binary',metric='auc',is_unbalance=True,max_depth=7, learning_rate=0.1,
                 num_iterations=100,feature_names='auto'):
        self.objective = objective
        self.metric = metric
        self.is_unbalance = is_unbalance
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.feature_names= feature_names

    def get_params(self,deep=True):
        return {'objective': self.objective,
                'metric': self.metric,
                'is_unbalance': self.is_unbalance,
                'max_depth' : self.max_depth,
                'learning_rate': self.learning_rate,
                'num_iterations': self.num_iterations,
                'feature_names': self.feature_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    def fit(self, X_train,y_train,X_val=None,y_val=None):

        # to record eval results for plotting
        evals_result = {}

        lgb_train = lgb.Dataset(X_train,label=y_train)

        if X_val != None:
            lgb_val = lgb.Dataset(X_val, label=y_val)
            valid_sets = [lgb_train,lgb_val]
            valid_names = ['training_set','validation_set']
        else:
            valid_sets = [lgb_train]
            valid_names = ['training_set']

        self.lgbm_classifier = lgb.train(self.get_params(),lgb_train,
                                         feature_name=self.feature_names,
                                         evals_result=evals_result,
                                         valid_sets=valid_sets,
                                         valid_names=valid_names,
                                         verbose_eval=10)
        self.evals_result = evals_result
        
    def predict(self,X_transf):
        return  self.lgbm_classifier.predict(X_transf)

    def score(self, X_transf, y_true):
        y_pred = self.predict(X_transf)
        return roc_auc_score(y_true, y_pred)
    
    @property
    def feature_importance_(self):
        return self.lgbm_classifier.feature_importance()
    
class RFClassifier(RandomForestClassifier):
    
    def __init__(self,n_estimators=100, random_state=2020,max_depth=10,max_features='sqrt',verbose=10,
                 n_jobs=-1):
        self.n_estimators=n_estimators
        self.random_state=random_state
        self.max_depth = max_depth
        self.max_features=max_features
        self.verbose=verbose
        self.n_jobs=2

        super().__init__(n_estimators = self.n_estimators, random_state=self.random_state,
                         max_depth=self.max_depth, max_features=self.max_features, verbose=self.verbose,
                         n_jobs=self.n_jobs)

    def get_params(self,deep=True):
        return {'n_estimators': self.n_estimators,
                'random_state': self.random_state,
                'max_depth': self.max_depth,
                'max_features': self.max_features,
                'verbose': self.verbose,
                'n_jobs': self.n_jobs}


    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self,X,y):
        super().fit(X,y)
    
    def predict(self,X_transf):
        return self.predict_proba(X_transf)[:,1]

    def score(self, X_transf, y_true):
        y_pred = self.predict(X_transf)
        return roc_auc_score(y_true, y_pred)

    @property
    def feature_importance_(self):
        return super().feature_importances_
    

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


class LGBM_classifier():
    def __init__(self,objective='binary',metric='auc',is_unbalance=True,max_depth=7, learning_rate=0.1,
                 num_iterations=100):
        self.objective = objective
        self.metric = metric
        self.is_unbalance = is_unbalance
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations


    def get_params(self,deep=True):
        return {'objective': self.objective,
                'metric': self.metric,
                'is_unbalance': self.is_unbalance,
                'max_depth' : self.max_depth,
                'learning_rate': self.learning_rate,
                'num_iterations': self.num_iterations}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X,y):
        
        lgb_train = lgb.Dataset(X,label=y)                    
        self.lgbm_classifier = lgb.train(self.get_params(), lgb_train)
        
    def predict(self,X_transf):
        return  self.lgbm_classifier.predict(X_transf)

    def score(self, X_transf, y_true):
        y_pred = self.predict(X_transf)
        return roc_auc_score(y_true, y_pred)
    
    @property
    def feature_importance_(self):
        return self.lgbm_classifier.feature_importance()
    
class RFClassifier(RandomForestClassifier):
    
    def __init__(self,n_estimators=50, random_state=2020,max_depth=10,max_features='sqrt',verbose=10):
        self.n_estimators=n_estimators
        self.random_state=random_state
        self.max_depth = max_depth
        self.max_features=max_features
        self.verbose=verbose

        super().__init__(n_estimators = self.n_estimators, random_state=self.random_state,
                         max_depth=self.max_depth, max_features=self.max_features, verbose=self.verbose)

    def get_params(self,deep=True):
        return {'n_estimators': self.n_estimators,
                'random_state': self.random_state,
                'max_depth': self.max_depth,
                'max_features': self.max_features,
                'verbose': self.verbose}


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
    

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier


class LGBM_classifier():
    def __init__(self,**kwargs):
        self.params = kwargs
        default_model_params = {'objective': 'binary', 
                                    'metric': 'auc',
                                    'is_unbalance': True}
            
        if not self.params:
            self.params = default_model_params
        else:
            for key in default_model_params:
                self.params[key] = default_model_params[key]

    
    def fit(self, X,y):
        
        lgb_train = lgb.Dataset(X,label=y)                    
        self.lgbm_classifier = lgb.train(self.params, lgb_train)
        
    def predict(self,X_transf):
        return  self.lgbm_classifier.predict(X_transf)
    
    @property
    def feature_importance_(self):
        return self.lgbm_classifier.feature_importance()
    
class RFClassifier(RandomForestClassifier):
    
    def __init__(self,**kwargs):
        self.params= kwargs
        default_model_params = {'n_estimators': 50, 
                                'random_state': 2020,
                                'max_depth': 10,
                               'max_features':'sqrt',
                               'verbose': 10}
        
        if not self.params:
            self.params = default_model_params
        else:
            for key in default_model_params:
                self.params[key] = default_model_params[key]
                
        super().__init__(**self.params)
    
    def fit(self,X,y):
        super().fit(X,y)
    
    def predict(self,X_transf):
        return self.predict_proba(X_transf)[:,1]
    @property
    def feature_importance_(self):
        return super().feature_importances_
    
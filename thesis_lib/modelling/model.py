import os
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.model_selection import GridSearchCV
from pactools.grid_search import GridSearchCVProgressBar
from thesis_lib.modelling.data import *
from thesis_lib.modelling.pipeline import *
from thesis_lib.modelling.classifiers import *

class Model():
    def __init__(self, **kwargs):

        #self.model_params = kwargs
        self.classifier_name = kwargs.get('classifier','lgbm')
        self.categorical_features = kwargs.get('categorical_features',[])
        self.numerical_features = kwargs.get('numerical_features',[])
        self.text_features = kwargs.get('text_features', [])
        self.sequence_features = kwargs.get('sequence_features', [])
        self.scale_numerical = kwargs.get('scale_numerical', False)
        self.accepts_sparse = kwargs.get('accepts_sparse',True)

        self.pipeline = CustomPipeline(categorical_features= self.categorical_features,
                                       numerical_features= self.numerical_features,
                                       text_features=self.text_features,
                                       sequence_features=self.sequence_features,
                                       accepts_sparse=self.accepts_sparse,
                                       scale_numerical=self.scale_numerical
                                       ).build_pipeline()

    def get_model_params(self, deep=True):
        return {'classifier': self.classifier_name,
                'classifier_params': self.get_classifier_params(),
                'categorical_features': self.categorical_features,
                'numerical_features': self.numerical_features,
                'text_features': self.text_features,
                'sequence_features': self.sequence_features,
                'scale_numerical': self.scale_numerical,
                'accepts_sparse': self.accepts_sparse}

    def get_classifier_params(self):
        return self.classifier.get_params()


    @property
    def model_features(self):

        feature_names = []
        transformers = self.pipeline.named_steps['feature_engineering'].transformer_list
        for transformer in transformers:
            transformer_features = transformer[1].get_feature_names()
            feature_names.extend(transformer_features)

        #HACK
        feature_names = ["".join (c if c.isascii() and c.isalnum() else "_" for c in str(x))
                         for x in feature_names]

        return feature_names

    @property
    def n_features(self):
        return(len(self.model_features))

    def transform(self,data,transform_test=False):

        print('Fitting pipeline...')
        self.pipeline.fit(data.train.X)

        print('Transforming data...')
        self.X_train = self.pipeline.transform(data.train.X)
        self.y_train = data.train.y

        self.X_val = self.pipeline.transform(data.val.X)
        self.y_val = data.val.y

        if transform_test:
            self.X_test = self.pipeline.transform(data.test.X)
            self.y_test = data.test.y


    def fit_classifier(self, **kwargs):
        if self.classifier_name == 'lgbm':
            self.classifier = LGBM_classifier(feature_names=self.model_features)
            self.classifier.set_params(**kwargs)
        elif self.classifier_name  == 'random_forest':
            self.classifier = RFClassifier()
            self.classifier.set_params(**kwargs)

        print('Training classifier')
        if self.classifier_name == 'lgbm':
            self.classifier.fit(self.X_train,self.y_train,
                                self.X_val,self.y_val)
        else:
            self.classifier.fit(self.X_train, self.y_train)


    def fit_best_classifier(self):
        assert hasattr(self, 'model_selection')
        self.fit_classifier(**self.model_selection.best_params_)

    def predict(self,X_transf):
        return self.classifier.predict(X_transf)

    def score(self,X_transf, y):
        return self.classifier.score(X_transf, y)

    @property
    def get_feature_importance(self):
        return {'feature_importance': self.classifier.feature_importance_,
                'feature_name': self.model_features}

    def plot_feature_importance(self, n_features=30):
        importance_df = pd.DataFrame(self.get_feature_importance).sort_values(
            by='feature_importance',ascending=True)
        importance_df[-n_features:].plot.barh(
            x=1,y=0, title='Feature Importance',legend=False,figsize=(8,10))


    def get_performance_metrics(self):

        auc_train = self.score(self.X_train, self.y_train)
        auc_val = self.score(self.X_val, self.y_val)

        print('training AUC ROC score: ',auc_train)

        print('validation AUC ROC score: ',auc_val)

        print('relative over-fitting: ',abs(auc_train-auc_val)/auc_train)

    def optimize_hyperparams(self, params_dict, n_iter=10, n_folds=5, search_type='random'):

        if self.classifier_name == 'lgbm':
            tmp_classifier =  LGBM_classifier()
        elif self.classifier_name == 'random_forest':
            tmp_classifier = RFClassifier()

        if search_type == 'random':
            self.model_selection = RandomizedSearchCV(estimator=tmp_classifier,
                                        param_distributions=params_dict,refit=False,
                                        random_state=2020,n_iter=n_iter,cv=n_folds,verbose=10,
                                        n_jobs=-1)
        elif search_type == 'grid':
            self.model_selection = GridSearchCVProgressBar(estimator=tmp_classifier,
                                                param_grid=params_dict, refit=False,
                                                cv=n_folds,verbose=10, n_jobs=-1)

        self.model_selection.fit(self.X_train,self.y_train)

        return self.model_selection.cv_results_

    def get_model_selection_results(self):
        results = self.model_selection.cv_results_
        results_df = pd.DataFrame({key: results[key] for key in results
                                if key != 'params'})
        return results_df






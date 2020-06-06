import pandas as pd
import os

class Data():
    def __init__(self,target='discharge'):
        self.target = target
        
    def load(self,path='data/hospital_dataset'):
        
        self.train = Dataset(path,'train',target_col=self.target)
        self.val = Dataset(path,'validation',target_col=self.target)
        self.test = Dataset(path,'test',target_col=self.target)
        
        return self
        
    def get_stats(self):
        total_data = 0
        summary = []

        for dataset_type in ['train','val','test']:
            dataset = getattr(self,dataset_type)
            data_len,n_cols = dataset.X.shape
            total_data +=  data_len

            negatives, positives = dataset.y.value_counts()
            negative_prop, positive_prop = dataset.y.value_counts(normalize=True)
            
            start_date = min(dataset.X.date)
            end_date = max(dataset.X.date)

            dataset_summary = {'dataset_type': dataset_type,
                               'n_observations': data_len,
                               'relative_size': None,
                               'n_cols': n_cols,
                               'positives': positives,
                               'negatives':negatives,
                               'positive_prop': positive_prop,
                               'negative_prop': negative_prop,
                               'min_date': start_date,
                               'max_date': end_date
                              }
            summary.append(dataset_summary)

        summary_df = pd.DataFrame(summary)
        summary_df['relative_size'] = summary_df['n_observations']/total_data
        
        return summary_df.set_index('dataset_type').T
    
    def get_variables_dict(self,max_categories=10000):
        
        categorical = []
        numerical = []
        
        for col in self.train.X:
            if self.train.X[col].dtype == 'object':
                if self.train.X[col].nunique() < max_categories:
                    categorical.append(col)
                       
            elif self.train.X[col].dtype in ['float', 'int'] and col != 'discharge':
                numerical.append(col)
                
        return {'categorical_variables': categorical,
               'numerical_variables': numerical}



class Dataset():
    def __init__ (self, path, dataset_type, target_col):
        
        self.dataset_type = dataset_type
        self.parquets_folder = path
        self.target_col = target_col

        self.X,self.y = self.load_dataset()
        self.X_transf = None
        
    def get_dataset(self, parquets_folder,dataset_type):    
    
        path= parquets_folder+'/'
        filenames = ['hospital_train_data.parquet','hospital_val_data.parquet','hospital_test_data.parquet']

        for file in filenames:
            if not os.path.isfile(path+file):
                print(file, 'not available in the specified folder')
            else:
                if '_train_' in file and dataset_type == 'train':
                    print('Loading dataset: ',file)
                    data = pd.read_parquet(path+file)
                elif '_val_' in file and dataset_type == 'validation':
                    print('Loading dataset: ',file)
                    data = pd.read_parquet(path+file)
                elif '_test_' in file and dataset_type == 'test':
                    print('Loading dataset: ',file)
                    data = pd.read_parquet(path+file)
        
        return data 
    
    def load_dataset(self):
        
        raw_data = self.get_dataset(self.parquets_folder, 
                                   self.dataset_type)
        
        y_data = raw_data[self.target_col]
        X_data = raw_data.drop(self.target_col,axis=1)
        
        return X_data, y_data

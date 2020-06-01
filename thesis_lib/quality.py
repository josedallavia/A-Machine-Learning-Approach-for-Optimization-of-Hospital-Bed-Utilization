import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt 
import thesis_lib
from datetime import datetime, date 
from thesis_lib.data_processing import *
from thesis_lib.visual import *
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from math import ceil
from numpy import nan
from thesis_lib.dataset import *
from thesis_lib.quality import *


import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt 
import thesis_lib
from datetime import datetime, date 
from thesis_lib.data_processing import *
from thesis_lib.visual import *
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from math import ceil
from numpy import nan
from IPython.display import display, HTML




class QualityReport:
    def __init__(self, dataset):
        self.df = dataset.df
        self.categorical = dataset.categorical_variables
        self.numerical = dataset.numerical_variables
        self.df_name = dataset.df_name
        
    
    @property
    def categorical_to_plot(self):
        categorical_to_plot = []
        for column in self.categorical:
            max_agregation_prop = self.df[column].value_counts(normalize=True)[0]
            if max_agregation_prop > 0.05 and max_agregation_prop < 0.99 :
                    categorical_to_plot.append(column)

        return categorical_to_plot
    
    def get_categorical_plots(self):
        
        print('\t Getting histograms for categorical variables...')
        print('\t\t Categorical variables with max relative frequency < 0.05 or >0.95 will be excluded')
        
        n_cols = 3
        for i in range (len(self.categorical_to_plot)):
            if (i + 3)  % n_cols  == 0:

                fig, axs = plt.subplots(1, 3,figsize=(20,5))
                #plt.tight_layout()
                categorical = self.categorical_to_plot[i:i+3]

                for variable in categorical:
                    for j, ax in enumerate(fig.axes):
                        if j < len(categorical):
                            variable = categorical[j]
                            bars_data = self.df[variable].value_counts(normalize=True)
                            if len(bars_data) > 12:
                                bars_data[:10].plot.bar(ax = ax, title=variable)
                                labels= ax.get_xticklabels()
                                ax.set_xticklabels(labels, rotation=45)

                            else:
                                bars_data.plot.bar(ax = ax, title=variable)
                                labels= ax.get_xticklabels()
                                ax.set_xticklabels(labels, rotation=45)
                        else:
                            ax.axis('off')
        plt.show()
    
    def get_numerical_plots(self):
        
        print('\t Getting distribution plots for numerical variables...')
        
        for numerical in self.numerical:
            fig, axs = plt.subplots(1, 3,figsize=(20,5))
            fig.suptitle(numerical)
            #plt.tight_layout()
            self.df[numerical].plot.hist(ax = axs[0])
            self.df[numerical].plot.density(ax = axs[1])
            self.df[numerical].plot.box(ax = axs[2])
            plt.setp(axs[2].get_xticklabels(), visible=False)
        plt.show()
            
            
   
    def get_tabular_report(self,variable_type):

        """
        variable_type (str) options are: 'numerical' or 'variables'
        """
        print('\t Getting tabular report for {variable_type} variables...'.format(variable_type=variable_type))
        
        
        if variable_type == 'numerical':
            report_columns = ['dataset','variable', 'count', 'mean', 'std', 
                              'min', '25%', '50%', '75%', 'max','missings',
                             'missings %']
            report_df_columns = self.df[self.numerical]
            
        else:
            report_columns = ['dataset','variable','count','unique','top',
                              'freq','missings', 'missings %']
            report_df_columns = self.df[self.categorical]

            
        raw_report = report_df_columns.describe(include='all')
        report = raw_report.T.reset_index().rename(columns={'index':'variable'})
        
        
        len_df = self.df.shape[0]
        report['missings'] = len_df - report['count']
        report['missings %'] = (report['missings']/len_df)*100
        report['dataset'] = self.df_name


        return report[report_columns]
    
    def get_numerical_report(self):
        
        report = self.get_tabular_report('numerical')
        display(HTML(report.to_html()))
        self.get_numerical_plots()
        
    
    def get_categorical_report(self):
        
        report = self.get_tabular_report('categorical')
        display(HTML(report.to_html()))
        self.get_categorical_plots()
        
   
            
            

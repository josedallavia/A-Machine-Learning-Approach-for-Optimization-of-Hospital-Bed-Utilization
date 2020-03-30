import pandas as df
import matplotlib.pyplot as plt 

def plot_top_categories_mean(df,categorical,n,to_plot,figsize=(20,10),title=None):
    
    categories_top = df.groupby([categorical]).nunique().sort_values(by='admission_id', 
                                                            ascending=False).head(n)[categorical]
    
    
  
    
    df[df[categorical].isin(categories_top.axes[0])].groupby(categorical).mean()[to_plot].sort_values(
        ascending=False).plot.bar(figsize=figsize,title=title,color='blue')

    
def plot_top_categories_count(df,categorical,n,figsize=(20,10), title=None):
    
    categories_top = df.groupby([categorical]).nunique().sort_values(by='admission_id', 
                                                            ascending=False).head(n)[categorical]
    
    
    df[df[categorical].isin(categories_top.axes[0])].groupby(categorical).count()['admission_id'].sort_values(
        ascending=False).plot.bar(figsize=figsize,title=title, color='blue')

def plot_top_categories(df,categorical,n,figsize=(20,10),title=None):
    
    categories_top = df.groupby([categorical]).nunique().sort_values(by='admission_id', 
                                                            ascending=False).head(n)[categorical]
    

    df[
       df[categorical].isin(categories_top.axes[0])
       ].groupby(categorical).agg(
                                {'admission_id': 'count',
                                 'length': 'mean'}
                                ).sort_values(by='admission_id',
                                             ascending=False).plot.bar(figsize=figsize,
                                           title=['# of admissions by '+categorical,
                                                  'Avg. hospitalization length (hours) by '+categorical],
                                           subplots=True,
                                           sharex=True,
                                          legend=False)
    
  
    
    
def plot_scatter_length_vs_count(df,
                                 dimension,
                                 title=None,
                                 labeled=False,
                                 xlim=None,
                                 ylim=None,
                                 figsize=(15,8),
                                 N_rows=500):
    
    f, ax = plt.subplots(figsize=figsize)
   
    tmp = df.groupby(dimension).aggregate({'admission_id': 'count', 
                                        'length': 'mean'}).sort_values(by='admission_id',
                                                                      axis=0,
                                                                      ascending=False).head(N_rows)
    
    x = tmp['admission_id']
    y= tmp['length']
    
    labels = list(tmp.index)
    
    ax.scatter(x,y)
    
    if labeled:
        for i, label in enumerate(labels):
            if xlim is None or (x[i] < xlim[1] and y[i]+50 < ylim[1]):
                ax.annotate(label[:15], (x[i], y[i]+50),fontsize=8)
           
    ax.set_xlabel('# of admissions')
    ax.set_ylabel('avg. hospitalization length')
    
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    f.suptitle(title if title else dimension)
    plt.show()
import pandas as df

def plot_top_categories_mean(df,categorical,n,to_plot,figsize=(20,10),title=None):
    
    categories_top = df.groupby([categorical]).nunique().sort_values(by='admission_id', 
                                                            ascending=False).head(n)[categorical]
    
    
  
    
    df[df[categorical].isin(categories_top.axes[0])].groupby(categorical).mean()[to_plot].plot.bar(figsize=figsize,
                                                                                                  title=title)

    
def plot_top_categories_count(df,categorical,n,figsize=(20,10), title=None):
    
    categories_top = df.groupby([categorical]).nunique().sort_values(by='admission_id', 
                                                            ascending=False).head(n)[categorical]
    
    
    df[df[categorical].isin(categories_top.axes[0])].groupby(categorical).count()['admission_id'].plot.bar(figsize=figsize,
                                                                                                      title=title)

def plot_top_categories(df,categorical,n,figsize=(20,10),title=None):
    
    categories_top = df.groupby([categorical]).nunique().sort_values(by='admission_id', 
                                                            ascending=False).head(n)[categorical]
    

    df[df[categorical].isin(categories_top.axes[0])].groupby(categorical).agg({'admission_id': 'count',
                                                                               'length': 'mean'}).plot.bar(figsize=figsize,
                                                                                                  title=title,
                                                                                                subplots=True,sharex=True)
    
    
def plot_scatter_length_vs_count(df,
                                 dimension,
                                 title=None,
                                 labeled=False,
                                 xlim=None,
                                 ylim=None,
                                 figsize=(15,8)):
    
    f, ax = plt.subplots(figsize=figsize)
   
    tmp = df.groupby(dimension).aggregate({'admission_id': 'count', 
                                        'length': 'mean'})
    
    x = tmp['admission_id']
    y= tmp['length']
    
    labels = list(tmp.index)
    
    ax.scatter(x,y)
    
    if labeled:
        for i, label in enumerate(labels):
            ax.annotate(label[:15], (x[i], y[i]+50),fontsize=8)
           
    ax.set_xlabel('# of admissions')
    ax.set_ylabel('avg. hospitalization length')
    
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    f.suptitle(title if title else dimension)
    plt.show()
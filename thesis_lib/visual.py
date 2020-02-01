import pandas as df

def plot_top_categories_mean(df,categorical,n,to_plot,figsize=(20,10),title=None):
    
    categories_top = df.groupby([categorical]).nunique().sort_values(by='Nro Adm', 
                                                            ascending=False).head(n)[categorical]
    
    
  
    
    df[df[categorical].isin(categories_top.axes[0])].groupby(categorical).mean()[to_plot].plot.bar(figsize=figsize,
                                                                                                  title=title)

    
def plot_top_categories_count(df,categorical,n,figsize=(20,10), title=None):
    
    categories_top = df.groupby([categorical]).nunique().sort_values(by='Nro Adm', 
                                                            ascending=False).head(n)[categorical]
    
    
    df[df[categorical].isin(categories_top.axes[0])].groupby(categorical).count()['Nro Adm'].plot.bar(figsize=figsize,
                                                                                                      title=title)

def plot_top_categories(df,categorical,n,figsize=(20,10),title=None):
    
    categories_top = df.groupby([categorical]).nunique().sort_values(by='Nro Adm', 
                                                            ascending=False).head(n)[categorical]
    

    df[df[categorical].isin(categories_top.axes[0])].groupby(categorical).agg({'Nro Adm': 'count',
                                                                               'Length': 'mean'}).plot.bar(figsize=figsize,
                                                                                                  title=title,
                                                                                                subplots=True,sharex=True)
    
    

    
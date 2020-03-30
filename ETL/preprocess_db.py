import os, sys
import pandas as pd
from os import path
import shutil
import matplotlib.pyplot as plt 
from random import random
import numpy as np 
import etl_helpers 
from etl_helpers.data_processing import *
from etl_helpers.db_utils import * 
from datetime import timedelta
from datetime import datetime
from collections import namedtuple
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm


def main(origin_directory,destination_directory):
    
    db = get_database(origin_directory) 
    
    try:
        os.mkdir(destination_directory)
    except:
        pass
    
    for dataset in db.keys():
        df = process_df(db[dataset], dataset)
        print('Saving dataset:', dataset)
        df.to_parquet(destination_directory+dataset+'.parquet')
    
   
if __name__ == '__main__':
    
    origin_directory = '/Users/josefinadallavia/Documents/MIM/Tesis/AML-hospital/p_data'
    destination_directory = origin_directory+'/parquet_data/'
    main(origin_directory, destination_directory)
    
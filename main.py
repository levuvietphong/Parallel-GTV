#from matplotlib import pyplot as plt
#import seaborn as sns
import pandas as pd
import geopandas as gpd
import numpy as np
from mpi4py import MPI
from utils import *
from gtv import *
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":

    """
    0. Activate MPI
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
        
    """
    1. Load climate division for area
    """
    fp = "CONUS_CLIMATE_DIVISIONS/GIS.OFFICIAL_CLIM_DIVISIONS.shp"
    shpdata = gpd.read_file(fp)
    shpdata["area"] = shpdata['geometry'].area  # unit square degree
      
    
    """
    2. Create a DataFrame that have States & Divisions we want to extract areal-mean Precipitation    
        numbers before ":" represent state ID
        numbers after ":" represent climate division within each state
        Example: Areal mean precipitation over SWUS region (17 divisions)
        CA - StateID 04, divisions 5-7
        AZ - StateID 02, divisions 1-7
        UT - StateID 42, divisions 1,2,4,6,7
        NV - StateID 26, divisions 3,4
    """
    fdata = './CONUS_CLIMATE_DIVISIONS/climdiv-pcpndv-v1.0.0-20201104'    
    # # SWUS
    # dict = {4:[5, 6, 7, np.nan, np.nan, np.nan, np.nan], 
    #         2:[1, 2, 3, 4, 5, 6, 7], 
    #         42:[1, 2, 4, 6, 7, np.nan, np.nan],         
    #         26: [3, 4, np.nan, np.nan, np.nan, np.nan, np.nan]}
    # # NorthWest
    # dict = {45:[1, 2, 3, 4], 
    #         35: [1, 2, 3, 4]}
    # Florida
    dict = {8:[1, 2, 3, 4, 5, 6, 7]}    
        
    df_data = pd.DataFrame(dict) 
    df, dfS, dfC = Extract_Precip_Divisions(fdata,shpdata,df_data,1940,2018)
    
    
    """
    3. Load data and compute covariance matrix
    """
    X = pd.read_csv('data/X_obs.csv').values
    #y = pd.read_csv('data/y_avg.csv').values.reshape(1,-1)[0]
    dfC['PrecipAnomaly'].to_csv('data/ygroup_avg.csv',index=None)
    y = dfC['PrecipAnomaly'].values.reshape(1,-1)[0]
    
    Xlens = pd.read_csv('data/X_lens.csv').values
    fts = pd.read_csv('data/sst_columns.csv')

    Slens = Xlens.T@Xlens/Xlens.shape[0]
    Dlens = edge_incidence(Slens, .5) # we threshold the covariance matrix at .5

    # split data into train/test
    X_train = X[:50]
    y_train = y[:50]
    X_test = X[50:]
    y_test = y[50:]
    
    print("Searching....")
    lambda_lasso_glob = np.linspace(1e-6, 0.5, 100)
    lamda_tv = np.logspace(-5, 0, 50)
    lambda_lasso_path, col = DomainDecompose(comm,rank,size,lambda_lasso_glob)
    
    # Search on subdomain of lambda_lasso_path
    df2 = gtv_cvx_path(X, y, Dlens, lambda_lasso_path, lamda_tv, alpha=.9)
    nrows = df2.shape[0]
    print(rank,nrows)
    output = df2.values
    send_data = output.flatten()    
    comm.Barrier()
        
    if rank != 0: # workers
        comm.send(nrows, dest=0, tag=1)
        comm.Send(send_data, dest=0, tag=2)
    comm.Barrier()        
    
    if rank==0:
        for i in range(0,size):
            if i==0:
                data = send_data.reshape(nrows,4)
            else:
                nrows = comm.recv(source=i,tag=1)
                send_data = np.empty(nrows*4, dtype=np.float)
                comm.Recv(send_data, source=i, tag=2)
                data2 = send_data.reshape(nrows,4)
                data = np.concatenate((data,data2), axis=0)
        df2 = pd.DataFrame(data, columns=['lambda_1', 'lambda_tv', 'r2', 'mse'])
        df2.to_csv('Florida_res.csv',index=False)
    comm.Barrier()        
    
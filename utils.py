#import seaborn as sns
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import numpy as np
# from matplotlib import pyplot as plt
# import matplotlib as mpl
#from matplotlib.patches import Rectangle
#from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def DomainDecompose(comm,rank,size,arr):   
    if rank == 0:
        print('Total number of models: %d' % len(arr))
        numpairs = np.shape(arr)[0]
        counts = np.arange(size,dtype=np.int32)
        displs = np.arange(size,dtype=np.int32)
        ave = int(numpairs / size)
        extra = numpairs % size
        offset = 0

        for i in range(0,size):
            col = ave if i<size-extra else ave+1
            counts[i] = col

            if i==0:
                col0 = col
                offset += col
                displs[i] = 0
            else:
                comm.send(offset, dest=i)
                comm.send(col, dest=i)
                offset += col
                displs[i] = displs[i-1] + counts[i-1]

            for j in range(offset-col,offset):
                print('Rank: %d - %s' % (i, arr[j]))

        offset = 0
        col = col0

    comm.Barrier()

    if rank != 0: # workers
        offset = comm.recv(source=0)
        col = comm.recv(source=0)

    comm.Barrier()
    sub_arr = arr[offset:offset+col]
    return sub_arr, col


def Extract_Precip_Divisions(filename, shpdata, dfin, yrs, yre):
    """
    This function loads Precipitation from different divisions of climate across the US.
    """    
    df = pd.read_csv(filename, delimiter=r"\s+", header = None, dtype=str)
    num_pts = df.shape[0]
    state = np.zeros(num_pts).astype(int)
    div = np.zeros(num_pts).astype(int)
    month = np.zeros(num_pts).astype(int)
    year = np.zeros(num_pts).astype(int)    
    
    for i in range(1,13):
        df[i]=df[i].astype(np.float)
    
    for i in range(num_pts):
        state[i] = df.loc[i,0][0:2]
        div[i] = df.loc[i,0][2:4]
        month[i] = df.loc[i,0][4:6]
        year[i] = df.loc[i,0][6:10]

    df['State'] = state
    df['Division'] = div
    df['Month'] = month
    df['Year'] = year
    num_years = year[-1]-year[0]
    area_all = [0]
    
    for k,sta in enumerate(dfin.columns): # State loop
        for i,div in enumerate(dfin[sta]):
            if ~np.isnan(div):
                # Get area of the division (unit in square degree)
                dfgeo = shpdata[(shpdata['STATE_CODE']==sta) & (shpdata['CD_NEW']==div)]
                area = dfgeo['area'].values[0]

                dfS = df[(df['State']==sta) & (df['Division']==div)]
                dfS.reset_index(inplace=True)
                Precip_winter = np.zeros(num_years)        
                for j,yr in enumerate(dfS['Year'][:-1]):
                    Precip_winter[j] = dfS.loc[j][12:14].sum()+dfS.loc[j+1][2:5].sum()

                if (i==0) & (k==0):
                    Precip_winter2 = Precip_winter * area
                else:
                    Precip_winter2 += Precip_winter * area
                area_all.append(area)
    Precip_winter_ave = Precip_winter2/np.sum(area_all) * 25.4    # Convert to mm
        
    dfw = pd.DataFrame(Precip_winter_ave,columns=['PrecipW[mm]'])
    dfw['Year'] = dfS['Year'][:-1]
    dfw = dfw[(dfw['Year']>=yrs) & (dfw['Year']<=yre)]     # Only keep years we want
    Precip_winter_ave = dfw['PrecipW[mm]'].values
    Precip_winter_anom = (Precip_winter_ave - np.mean(Precip_winter_ave))/np.std(Precip_winter_ave)  # Get the anomaly and unit variance    
    dfw['PrecipAnomaly'] = Precip_winter_anom    
    return df, dfS, dfw


def edge_incidence(S, threshold=0):
    """
    Computes the thresholded edge-incidence matrix of a covariance matrix for use in GTV
    :param S: covariance matrix (standardized so diagonals are all 1)
    :param threshold: float between 0 and 1 that all entries of the covariance matrix below are ignored
    :return: thresholded edge-incidence matrix
    """
    edges = np.where(abs(S)>threshold)
    edges_ix = np.where(edges[0]<edges[1])
    ix1 = edges[0][edges_ix]
    ix2 = edges[1][edges_ix]
    ix = np.arange(len(ix1))

    D = np.zeros([len(ix1), S.shape[0]])
    D[ix, ix1] = np.sqrt(abs(S[ix1, ix2]))
    D[ix, ix2] = -np.sign(S[ix1, ix2])*np.sqrt(abs(S[ix1, ix2]))
    return D


def plot_coefs(fts, color='sign', title='', month='', se=None,
               savefig=False, colorbar=False, vmin=None, vmax=None, cmap=None):
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.figure(figsize=(6,8))
    ax = plt.axes(projection=ccrs.Mercator(central_longitude=150, min_latitude=-60, max_latitude=60))
    ax.set_extent([80, 280, -60, 60])
    ax.stock_img()

    #nzi
    ax.add_patch(Rectangle(xy=[170, -40], width=30, height=15,
                           edgecolor='blue',
                           alpha=0.2,
                           transform=ccrs.PlateCarree())
                 )

    #nino
    for pts in [(190, 240)]:
        ur = (pts[1], 5)
        lr = (pts[0], -5)
        box = Rectangle(lr, ur[0]-lr[0], ur[1]-lr[1],
                        edgecolor='blue',
                        alpha=0.2,
                        transform=ccrs.PlateCarree())
        ax.add_patch(box)

    #pc = PatchCollection(boxes, edgecolor='blue', alpha=.1)
    #ax.add_collection(pc)

    #gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                  linewidth=.5, color='gray', alpha=0.5)
    # Label axes of a Plate Carree projection with a central longitude of 180:

    ax.set_xticks(list(np.arange(80, 281, 10)), crs=ccrs.PlateCarree())
    plt.xticks(rotation=45)
    ax.set_yticks(list(np.arange(-60,61,10)), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    plt.grid(linewidth=.25)

    if month == '':
        mfts = fts
    else:
        mfts = fts[fts.month==month]
    if se is not None:
        plt.scatter(mfts.lon, mfts.lat, c=np.sign(mfts.coef), s=(abs(mfts.coef)+se) * 1000,
                    transform=ccrs.PlateCarree(), vmin=-1, vmax=1, alpha=.2)
    if color == 'sign':
        pt = plt.scatter(mfts.lon, mfts.lat, c=np.sign(mfts.coef), s=abs(mfts.coef)*1000,
                    transform=ccrs.PlateCarree(), vmin=-1, vmax=1)
    elif color == 'size':
        pt = plt.scatter(mfts.lon, mfts.lat, c=mfts.coef, cmap=cmap,
                         transform=ccrs.PlateCarree())
    if colorbar:
        if vmin is not None:
            plt.clim(vmin, vmax)
            plt.colorbar(pt, fraction=.03)
        else:
            plt.colorbar(pt, fraction=.03)
    plt.title(month+str(title))
    plt.show()


def draw_lambda_contour(df, metric, vmin=None, vmax=None):
    sns.set(font_scale=1.25)
    d = df.pivot(index='lambda_1', columns='lambda_tv', values=metric)
    if vmin is None:
        vmin = df[metric].min()
        vmax = df[metric].max()
    d = d.interpolate(method='linear', axis=1, limit_direction='both')
    d = d.sort_index(ascending=False)
    sns.heatmap(d, vmin=vmin, vmax=vmax, cmap='coolwarm')
    plt.xlabel('$\lambda_{TV}$', fontweight='bold')
    plt.ylabel('$\lambda_1$', fontweight='bold')
    #plt.xticks(np.linspace(0, d.shape[0], min(d.shape[0], 10)), rotation='vertical')
    #plt.yticks(np.linspace(0, d.shape[0], min(d.shape[0], 10)), rotation='horizontal')

def plot_covariance(covs, titles, h = 4, ori='vertical', bar=True):
    n = len(covs)
    if bar:
        fig = plt.figure(figsize=(n*h+h/2, h))
    else:
        fig = plt.figure(figsize=(n*h, h))
    axes=[]
    plots = []
    subplot_ix = 100 + n*10
    for i in range(n):
        ax = fig.add_subplot(subplot_ix+i+1)
        axes.append(ax)
    for i, S in enumerate(covs):
        g = sns.heatmap(S, ax=axes[i], vmin=-1, vmax=1, cmap='coolwarm', xticklabels=False, yticklabels=False, cbar=False)
        plots.append(g)
        plots[i].set_title(titles[i])
    mappable = plots[0].get_children()[0]
    if bar:
        plt.colorbar(mappable, ax = axes, orientation = ori)


def shorten_region(region):
    if region in ['Area-weighted average', 'Areal Average']:
        return 'Areal \n Average'
    if len(region) > 5:
        state_map = {
            'Arizona': 'AZ',
            'Utah': 'UT',
            'Nevada': 'NV',
            'California': 'CA'
        }
        return state_map[region.split('(')[0]] + '(' + region.split('(')[1]
    else:
        return region


# def plot_regional_metrics(df, metric='mse', order=None, pal=['navy', 'orange', 'green', 'firebrick'], figsize=(25,5)):
#     """
#     Dataframe should have the following columns:
#     - Region
#     - Method
#     - Test MSE or R2
#     """
#     # make sure regions are short
#     df['Region'] = df.Region.apply(lambda x: shorten_region(x))
#     plt.figure(figsize=figsize)
#     sns.set(font_scale=2)
#     if metric == 'mse':
#         sns.barplot(x='Region', y='Test MSE', hue='Method', data=df, hue_order=order)
#         plt.ylabel('MSE', fontweight='bold')
#     elif metric == 'r2':
#         df['R2'] = df.R2.apply(lambda x: max(.01, x))
#         sns.barplot(x='Region', y='R2', hue='Method', data=df, hue_order=order,
#                     palette=pal)
#         plt.ylabel('$R^2$', fontweight='bold')
#     plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
#                     mode="expand", borderaxespad=0, ncol=4)
#     plt.xlabel('Region', fontweight='bold')


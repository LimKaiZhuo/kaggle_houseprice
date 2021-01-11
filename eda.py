import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from own_package.others import plot_barh, chunks


def eda():
    rawdata = pd.read_csv('./inputs/train.csv')
    plot_barh(rawdata.isnull().sum(), add_count=True, total_count=rawdata.shape[0], figsize=[10,20])
    '''
    Drop:
    1) PoolQC: Because already have pool area and 99.5% nan
    2) MiscFeature: 96.3% nan
    3) Alley: 93.8% nan. Maybe can binary 0 None, 1 have alley access
    4) Fence: 80.8% nan. Maybe can either binary or one hot the whole thing. Try binary first
    5) FireplaceQu: The nan corresponds to the houses with 0 fireplace. Since it is ordinal, can keep as a
     single column and transform to 0 (none), 1 (po, poor), ..., 5 (ex, excellent) 
    6) LotFrontage: Ratio data. Set NaN as 0.
    7) GarageX: NaN is from those GarageType == NaN
    GarageType: one hot
    GarageYrBlt: Usually the same as YearBuilt. Can possibly remove.
    GarageFinish: Ordinal
    GarageArea and GarageCars: Keep GarageArea, remove GarageCars
    GarageQual: Ordinal
    GarageCond: Same as GarageQual, remove
    8) Basement 
    9) Masonary
    10) Electrical: delete that row
    '''
    plot_barh(rawdata['Alley'].fillna('None').value_counts(), add_count=True, total_count=rawdata.shape[0])
    plot_barh(rawdata['Fence'].fillna('None').value_counts(), add_count=True, total_count=rawdata.shape[0])
    plot_barh(rawdata['FireplaceQu'].fillna('None').value_counts(), add_count=True, total_count=rawdata.shape[0])
    plot_barh(rawdata['Fireplaces'].fillna('None').value_counts(), add_count=True, total_count=rawdata.shape[0])


    rawdata.plot.scatter(x='GarageYrBlt', y='YearBuilt', title='Year Built')
    plt.show()
    plt.close()
    rawdata.plot.scatter(x='GarageCars', y='GarageArea', title='Garage Size')
    plt.show()
    plt.close()

    plot_barh(rawdata['MSSubClass'].fillna('None').value_counts(), add_count=True, total_count=rawdata.shape[0])

    # Neighborhood
    neighborhood_prices = rawdata.groupby('Neighborhood').agg({'SalePrice': np.mean}).sort_values('SalePrice')
    bins = 5
    neighborhood_chunks = list(chunks(neighborhood_prices.index.tolist(), int(np.ceil(len(neighborhood_prices.index.tolist())/bins))))



eda()





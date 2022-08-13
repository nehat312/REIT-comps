#%% [markdown]
## 6313 - FTP ##
## NATE EHAT ##

## REIT TRADING COMPARABLES ##

#%% [markdown]
### **ANALYSIS SECTORS:**

# * Analysis evaluates publicly traded REIT tickers, within the commercial RE sectors outlined below:
    #     * Retail - Strip Centers, Malls, Triple-Net Retail (NNN)
    #     * Multifamily - Rental Apartments
    #     * Office - Central Business District (CBD), Suburban (SUB)
    #     * Hospitality - Full-Service Hotels, Limited-Service Hotels
    #     * Industrial - Warehouse, Logistics

#%%
## LIBRARY IMPORTS ##

## BASE PACKAGES ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## SKLEARN ##
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

## SCIPY ##
from scipy import stats as stats
import scipy.stats as st
from scipy import signal
from scipy.stats import chi2

## STATSMODELS ##
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.api import SARIMAX, AutoReg
from statsmodels.stats.outliers_influence import variance_inflation_factor

## TENSORFLOW / KERAS ##
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras import Sequential
from keras import preprocessing
# from keras.preprocessing.sequence import TimeseriesGenerator
# from keras.preprocessing.timeseries import timeseries_dataset_from_array

## SUPPLEMENTAL ##
from numpy import linalg
from numpy import linalg as LA

## WARNINGS ##
import warnings
warnings.filterwarnings('ignore')

## TOOLBOX ##
from toolbox import *

## NOT-IN-USE ##
# import statistics
# import pandas_datareader as web
# import requests
# import json
# import time
# import datetime as dt
# from google.colab import drive

print("\nIMPORT SUCCESS")

#%%
## FOLDER CONFIGURATION ##

# CURRENT FOLDER / PATH
current_folder = '/Users/nehat312/GitHub/Time-Series-Analysis-and-Moldeing/FTP/'

print("\nDIRECTORY CONFIGURED")

#%%
## VISUAL SETTINGS ##
sns.set_style('whitegrid') #ticks

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

print("\nSETTINGS ASSIGNED")

#%%
## ANALYSIS PARAMETERS ##
start_date = '2000-01-01'
end_date = '2022-03-31'

mo_qtr_map = {'01': '1', '02': '1', '03': '1',
              '04': '2', '05': '2', '06': '2',
              '07': '3', '08': '3', '09': '3',
              '10': '4', '11': '4', '12': '4'}

print("\nPARAMETERS ASSIGNED")

#%%
## REAL ESTATE SECTORS / TICKERS ##

apartment = ["EQR",	"AVB", "ESS", "MAA", "UDR",	"CPT", "AIV", "BRG", "APTS"]
office = ["BXP", "VNO",	"KRC", "DEI", "JBGS", "CUZ", "HPP",	"SLG",	"HIW", "OFC", "PGRE", "PDM", "WRE",	"ESRT",	"BDN", "EQC", "VRE"] #"CLI"
hotel = ["HST",	"RHP",	"PK", "APLE", "SHO", "PEB", "RLJ", "DRH", "INN", "HT", "AHT", "BHR"]    #"XHR",
mall = ["SPG", "MAC", "PEI"]    #CBL	TCO	"WPG",
strip_center = ["REG", "FRT",	"KIM",	"BRX", "AKR", "UE", "ROIC", "CDR", "SITC", "BFS"]   #"WRI", "RPAI",
net_lease = ["O", "WPC", "NNN",	"STOR",	"SRC", "PINE", "FCPT", "ADC", "EPRT"]  # "VER",
industrial = ["PLD", "DRE",	"FR", "EGP"]
self_storage = ["EXR",	"CUBE",	"REXR",	"LSI"]
data_center = ["EQIX", "DLR" "AMT"]     #"CONE", "COR"
healthcare = ["WELL", "PEAK", "VTR", "OHI", "HR"]   #"HTA",

sectors = [apartment, office, hotel, mall, strip_center, net_lease, industrial, self_storage, data_center, healthcare]

reit_tickers = ["EQR",	"AVB",	"ESS",	"MAA",	"UDR",	"CPT",	"AIV",	"BRG", "APTS",
               "BXP",	"VNO",	"KRC",	"DEI",	"JBGS",	"CUZ",	"HPP",	"SLG",	"HIW",	"OFC",	"PGRE",	"PDM",	"WRE",	"ESRT",	"BDN", "EQC",
               "HST",	"RHP",	"PK",	"APLE",	"SHO",	"PEB",	"RLJ",	"DRH",	"INN",	"HT",	"AHT",	"BHR",
               "SPG",	"MAC", "PEI", "SKT", "SRG",
               "REG", "FRT",	"KIM",	"BRX",	"AKR",	"UE",	"ROIC",	"CDR",	"SITC",	"BFS",
               "O",	"WPC",	"NNN",	"STOR",	"SRC", "PINE", "FCPT", "ADC", "EPRT",
               "PLD",	"DRE",	"FR",	"EGP",  "GTY",
               "EXR",	"CUBE",	"REXR",	"LSI",
               "EQIX", "DLR", "AMT",
               "WELL",	"PEAK",	"VTR",	"OHI",	"HR"]

print("\nVARIABLES ASSIGNED")

#%%
## INITIALIZE LISTS TO STORE MODEL RESULTS ##
model_name = []
model_mse = []
model_ljb = []
model_error_var = []
model_notes = []

print("\nLISTS INITIALIZED")

#%%
# sector_dict = {'apartment': ["EQR",	"AVB",	"ESS",	"MAA",	"UDR",	"CPT",	"AIV",	"BRG", "APTS"],
#                'office': ["BXP",	"VNO",	"KRC", "DEI", "JBGS",	"CUZ", "HPP",	"SLG",	"HIW", "OFC", "PGRE",	"PDM", "WRE",	"ESRT",	"BDN", "EQC", "VRE"],
#                'hotel': ["HST",	"RHP",	"PK",	"APLE",	"SHO",	"PEB",	"RLJ", "DRH",	"INN", "HT", "AHT",	"BHR"],
#                'mall': ["SPG", "MAC", "PEI"],
#                'strip_center': ["REG", "FRT",	"KIM",	"BRX",	"AKR",	"UE",	"ROIC",	"CDR",	"SITC",	"BFS"],
#                'net_lease': ["O",	"WPC",	"NNN",	"STOR",	"SRC",  "PINE", "FCPT", "ADC", "EPRT"],
#                'industrial': ["PLD", "DRE",	"FR",	"EGP"],
#                'self_storage': ["EXR",	"CUBE",	"REXR",	"LSI"],
#                'data_center': ["EQIX", "DLR" "AMT"],
#                'healthcare': ["WELL",	"PEAK",	"VTR",	"OHI", "HR"]}

#%%
# IMPORT DATA (DATAFRAMES BY RE SECTOR)
all_sectors_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='ALL SECTORS', parse_dates = True, index_col = [0], header=[3])
office_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='OFFICE', parse_dates = True, index_col = [0], header=[2])
residential_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='RESIDENTIAL', parse_dates = True, index_col = [0], header=[2])
lodging_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='LODGING', parse_dates = True, index_col = [0], header=[2])
net_lease_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='NET LEASE', parse_dates = True, index_col = [0], header=[2])
strip_center_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='STRIP CENTER', parse_dates = True, index_col = [0], header=[2])
mall_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='MALL', parse_dates = True, index_col = [0], header=[2])
healthcare_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='HEALTH CARE', parse_dates = True, index_col = [0], header=[2])
industrial_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='INDUSTRIAL', parse_dates = True, index_col = [0], header=[2])
self_storage_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='SELF STORAGE', parse_dates = True, index_col = [0], header=[2])
data_center_import = pd.read_excel(current_folder + 'data/reit_trading_2000_2022.xlsx', sheet_name='DATA CENTER', parse_dates = True, index_col = [0], header=[2])

print("\nIMPORT SUCCESS")

#%%
## SAVE COPIES OF IMPORTS

sector_comps = all_sectors_import
office_comps = office_import
residential_comps = residential_import
lodging_comps = lodging_import
net_lease_comps = net_lease_import
strip_center_comps = strip_center_import
mall_comps = mall_import
healthcare_comps = healthcare_import
industrial_comps = industrial_import
self_storage_comps = self_storage_import
data_center_comps = data_center_import

sector_df_list = [office_comps, residential_comps,  lodging_comps, net_lease_comps, strip_center_comps,
                  mall_comps, healthcare_comps, industrial_comps, self_storage_comps, data_center_comps]

print("\nCOPIES SAVED")


#%%
print(sector_comps.info())
print(sector_comps.columns)

#%%
## FEATURE VARIABLES ##
sector_return_cols = ['ALL_AVG_RETURN_1D', 'OFF_AVG_RETURN_1D', 'RESI_AVG_RETURN_1D', 'HOT_AVG_RETURN_1D',
                      'NL_AVG_RETURN_1D', 'SC_AVG_RETURN_1D', 'MALL_AVG_RETURN_1D', 'HC_AVG_RETURN_1D',
                      'IND_AVG_RETURN_1D', 'SS_AVG_RETURN_1D', 'DC_AVG_RETURN_1D']

sector_close_cols = ['ALL_AVG_CLOSE', 'OFF_AVG_CLOSE', 'RESI_AVG_CLOSE', 'HOT_AVG_CLOSE',
                     'NL_AVG_CLOSE', 'SC_AVG_CLOSE', 'MALL_AVG_CLOSE', 'HC_AVG_CLOSE',
                     'IND_AVG_CLOSE', 'SS_AVG_CLOSE', 'DC_AVG_CLOSE', ]

sector_returns = sector_comps[sector_return_cols]
sector_close = sector_comps[sector_close_cols]

print(sector_returns.info())
# print(sector_returns.columns)

#%%
# REFINE START DATE (ITERATIVE PROCESS)

# new_start_date = '1/1/2004' #'7/1/2009' #'4/1/2009' #'1/2/2009'

# sector_returns = sector_returns[sector_returns.index >= new_start_date]
# print(sector_returns.info())

#%%
## TIME SERIES PLOT ##
fig, axes = plt.subplots(1,1,figsize=(12,8))
# plt.figure(figsize=(10,8))

for i in sector_returns.columns:
    sns.lineplot(x=sector_returns.index, y=sector_returns[i], legend='full') #hue=sector_returns[i], #['ALL_AVG_RETURN_1D']
plt.title('1-DAY PRICE RETURN (%) (1/2000-6/2022)')
plt.xlabel('DATE')
plt.ylabel('1-DAY PRICE RETURN (%)')
plt.tight_layout(pad=1)
#plt.grid()
plt.legend(loc='best')
plt.show()

#%%
## TIME SERIES PLOT ##
fig, axes = plt.subplots(1,1,figsize=(12,8))
sns.lineplot(x=sector_returns.index, y=sector_returns['ALL_AVG_RETURN_1D'], legend='brief') #hue=sector_returns[i], #['ALL_AVG_RETURN_1D']
plt.title('1-DAY PRICE RETURN (%) (1/2000-6/2022)')
plt.xlabel('DATE')
plt.ylabel('1-DAY PRICE RETURN (%)')
plt.tight_layout(pad=1)
#plt.grid()
plt.legend(loc='best')
plt.show()

#%%
plt.figure(figsize=(16,12))
plt.subplot(5,2,1)
sns.lineplot(x=sector_returns.index, y=sector_returns['ALL_AVG_RETURN_1D'], palette='mako')
plt.title('ALL SECTORS: 1-DAY AVG PRICE RETURN (%) (1/2000-6/2022)') #, fontsize=12
plt.xlabel('DATE')
plt.ylabel('1-DAY PRICE RETURN (%)')
plt.tight_layout(pad=1)

plt.subplot(5,2,2)
sns.lineplot(x=sector_returns.index, y=sector_returns['OFF_AVG_RETURN_1D'], palette='flare')
plt.title('OFFICE: 1-DAY AVG PRICE RETURN (%) (1/2000-6/2022)') #, fontsize=12
plt.xlabel('DATE')
plt.ylabel('1-DAY PRICE RETURN (%)')
plt.tight_layout(pad=1)

plt.subplot(5,2,3)
sns.lineplot(x=sector_returns.index, y=sector_returns['RESI_AVG_RETURN_1D'], palette='maroon')
plt.title('RESIDENTIAL: 1-DAY AVG PRICE RETURN (%) (1/2000-6/2022)') #, fontsize=12
plt.xlabel('DATE')
plt.ylabel('1-DAY PRICE RETURN (%)')
plt.tight_layout(pad=1)

plt.subplot(5,2,4)
sns.lineplot(x=sector_returns.index, y=sector_returns['HOT_AVG_RETURN_1D'])
plt.title('LODGING: 1-DAY AVG PRICE RETURN (%) (1/2000-6/2022)') #, fontsize=12
plt.xlabel('DATE')
plt.ylabel('1-DAY PRICE RETURN (%)')
plt.tight_layout(pad=1)

plt.subplot(5,2,5)
sns.lineplot(x=sector_returns.index, y=sector_returns['NL_AVG_RETURN_1D'])
plt.title('NET LEASE: 1-DAY AVG PRICE RETURN (%) (1/2000-6/2022)') #, fontsize=12
plt.xlabel('DATE')
plt.ylabel('1-DAY PRICE RETURN (%)')
plt.tight_layout(pad=1)

plt.subplot(5,2,6)
sns.lineplot(x=sector_returns.index, y=sector_returns['SC_AVG_RETURN_1D'])
plt.title('STRIP CENTER: 1-DAY AVG PRICE RETURN (%) (1/2000-6/2022)') #, fontsize=12
plt.xlabel('DATE')
plt.ylabel('1-DAY PRICE RETURN (%)')
plt.tight_layout(pad=1)

plt.subplot(5,2,7)
sns.lineplot(x=sector_returns.index, y=sector_returns['MALL_AVG_RETURN_1D'])
plt.title('MALL: 1-DAY AVG PRICE RETURN (%) (1/2000-6/2022)') #, fontsize=12
plt.xlabel('DATE')
plt.ylabel('1-DAY PRICE RETURN (%)')
plt.tight_layout(pad=1)

plt.subplot(5,2,8)
sns.lineplot(x=sector_returns.index, y=sector_returns['HC_AVG_RETURN_1D'])
plt.title('HEALTHCARE: 1-DAY AVG PRICE RETURN (%) (1/2000-6/2022)') #, fontsize=12
plt.xlabel('DATE')
plt.ylabel('1-DAY PRICE RETURN (%)')
plt.tight_layout(pad=1)

plt.subplot(5,2,9)
sns.lineplot(x=sector_returns.index, y=sector_returns['IND_AVG_RETURN_1D'])
plt.title('INDUSTRIAL: 1-DAY AVG PRICE RETURN (%) (1/2000-6/2022)') #, fontsize=12
plt.xlabel('DATE')
plt.ylabel('1-DAY PRICE RETURN (%)')
plt.tight_layout(pad=1)


plt.subplot(5,2,10)
sns.lineplot(x=sector_returns.index, y=sector_returns['DC_AVG_RETURN_1D'], palette='mako')
plt.title('DATA CENTER: 1-DAY AVG PRICE RETURN (%) (1/2000-6/2022)') #, fontsize=12
plt.xlabel('DATE')
plt.ylabel('1-DAY PRICE RETURN (%)')
plt.tight_layout(pad=1)

# plt.legend(loc='best')

plt.show()


#%%
## TIME SERIES STATISTICS ##
for i in sector_return_cols:
    print(f'{i}:')
    print(f'MEAN: {sector_returns[i].mean():.4f}')
    print(f'VARIANCE: {sector_returns[i].var():.4f}')
    print(f'STD DEV: {sector_returns[i].std():.4f}')
    print('*' * 25)

#%%
## ROLLING MEAN / VARIANCE ##

# fig, axes = plt.subplots(5, 2, figsize=(12, 8))

for i in sector_returns.columns:
    rolling_mean_var_plots(rolling_mean_var(sector_returns[i]), i)
#rolling_mean_var_plots(rolling_mean_var(df_2[0]), df_2_index)

plt.show()


#%%
## ADF TEST ##
for a in sector_return_cols:
    print(f'ADF TEST - {a}:')
    print(adf_test([a], sector_returns))
    print('*' * 100)

#%%
## KPSS TEST ##
for k in sector_return_cols:
    print(f'KPSS TEST - {k}:')
    print(kpss_test(sector_returns[k]))
    print('*'*150)


#%%
## PRINCIPAL COMPONENT ANALYSIS ##

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
reit_num_cols = sector_returns.select_dtypes(include=numerics)
reit_num_cols.info()
# reit_num_cols = reit_num_cols.dropna(inplace=True)

#%%
X = reit_num_cols[reit_num_cols._get_numeric_data().columns.to_list()[:-1]]
Y = reit_num_cols['ALL_AVG_RETURN_1D']
print(X.describe())

#%%
## SCALING DATA ##
X = StandardScaler().fit_transform(X)

#%%
## PCA STATISTICS ##
pca = PCA(n_components=4, svd_solver='full') # 'mle'
pca.fit(X)
X_PCA = pca.transform(X)

print('ORIGINAL DIMENSIONS:', X.shape)
print('*'*100)
print('TRANSFORMED DIMENSIONS:', X_PCA.shape)
print('*'*100)
print(f'EXPLAINED VARIANCE RATIO: {pca.explained_variance_ratio_}')

#%%
## PCA PLOT ##
x = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1)

plt.figure(figsize=(12,8))
plt.plot(x, np.cumsum(pca.explained_variance_ratio_))
plt.title('PCA - EXPLAINED VARIANCE RATIO', fontsize=18)
plt.xlabel('PCA COMPONENT #', fontsize=16)
plt.ylabel('EXPLAINED VARIANCE %', fontsize=16)
plt.xticks(x)
plt.tight_layout(pad=1)
plt.show()

#%%
## SINGULAR VALUE DECOMPOSITION ANALYSIS [SVD] ##

H = np.matmul(X.T, X)
_, d, _ = np.linalg.svd(H)
print(f'ORIGINAL DATA: SINGULAR VALUES\n {d}')
print('*'*75)
print(f'ORIGINAL DATA: CONDITIONAL NUMBER\n {LA.cond(X)}')

#%%
# TRANSFORMED DATA
H_PCA = np.matmul(X_PCA.T, X_PCA)
_, d_PCA, _ = np.linalg.svd(H_PCA)
print(f'TRANSFORMED DATA: SINGULAR VALUES\n {d_PCA}')
print('*'*75)
print(f'TRANSFORMED DATA: CONDITIONAL NUMBER\n {LA.cond(X_PCA)}')

#%%
# CONSTRUCTION OF REDUCED DIMENSION DATASET

a, b = X_PCA.shape
column = []
#pca_df = pca.explained_variance_ratio_

for i in range(b):
    column.append(f'PRINCIPAL COLUMN {i+1}')

df_PCA = pd.DataFrame(data=X_PCA, columns=column, index=sector_returns.index)
df_PCA = pd.concat([df_PCA, Y], axis=1)

#%%
print(df_PCA.info())
print('*'*50)
print(df_PCA.head())
print('*'*50)
print(df_PCA.describe())

#%%
## PCA PLOT ALTERNATE ##
plt.figure(figsize=(12,8))
sns.lineplot(data=df_PCA, palette='mako', legend='brief')
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
## STL DECOMPOSITION ##
stl_data = sector_returns['ALL_AVG_RETURN_1D'].copy()
# stl_data.index = [i for i in range(stl_data.shape[0])]
stl_data.index = sector_returns.index
stl_res = STL(stl_data, period=12).fit()

#%%
## STL DECOMPOSITION PLOT ##
plt.figure(figsize=(12,8))
fig = stl_res.plot()
plt.xlabel('DATE', fontsize=12)
plt.tight_layout(pad=1)
plt.show()

#%%
## STL DECOMPOSITION T/S/R ##
T = stl_res.trend
S = stl_res.seasonal
R = stl_res.resid

#%%
## STRENGTH OF TREND ##
def strength_of_trend(residual, trend):
    var_resid = np.nanvar(residual)
    var_resid_trend = np.nanvar(np.add(residual, trend))
    return 1 - (var_resid / var_resid_trend)

F = np.maximum(0, 1-np.var(R)/np.var(np.array(T)+np.array(R)))

print(f'STRENGTH OF TREND: {100*F:.3f}% or {strength_of_trend(R, T):.5f}')

#%%
## STRENGTH OF SEASONAL ##
def strength_of_seasonal(residual, seasonal):
    var_resid = np.nanvar(residual)
    var_resid_seasonal = np.nanvar(np.add(residual, seasonal))
    return 1 - (var_resid / var_resid_seasonal)

F = np.maximum(0, 1-np.var(R)/np.var(np.array(S)+np.array(R)))

print(f'STRENGTH OF SEASONALITY: {100*F:.3f}% or {strength_of_seasonal(R, S):.5f}')

#%%
adjusted_seasonal = np.subtract(np.array(sector_returns.ALL_AVG_RETURN_1D), np.array(stl_res.seasonal))
detrended = np.subtract(np.array(sector_returns.ALL_AVG_RETURN_1D), np.array(stl_res.trend))
residual = np.array(stl_res.resid)

#%%
plt.figure(figsize=(12,8))
plt.plot(sector_returns.index, sector_returns.ALL_AVG_RETURN_1D, label='ORIGINAL DATA', color='dodgerblue')
plt.plot(sector_returns.index, adjusted_seasonal, label='ADJUSTED SEASONAL', color='yellow')
# plt.plot(sector_returns.index, detrended, label='DETRENDED', color='gray')
plt.title('SEASONALLY ADJUSTED DATA VS. ORIGINAL')
plt.xlabel('DATE')
plt.ylabel('')
# plt.xticks(sector_returns.index, fontsize=10) #[::4500]
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
plt.figure(figsize=(12,8))
plt.plot(sector_returns.index, sector_returns.ALL_AVG_RETURN_1D, label='ORIGINAL DATA', color='dodgerblue')
plt.plot(sector_returns.index, detrended, label='DETRENDED')

plt.title('DETRENDED VS. ORIGINAL')
plt.xlabel('DATE')
plt.ylabel('')
# plt.xticks(sector_returns.index, fontsize=10) #[::4500]
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
## VARIANCE INFLATION FACTORS ##
X = sector_returns[['OFF_AVG_RETURN_1D', 'RESI_AVG_RETURN_1D',
                    'HOT_AVG_RETURN_1D',
                    'NL_AVG_RETURN_1D', 'SC_AVG_RETURN_1D', 'MALL_AVG_RETURN_1D',
                    'IND_AVG_RETURN_1D',
                    'HC_AVG_RETURN_1D', 'SS_AVG_RETURN_1D',
                     'DC_AVG_RETURN_1D'
                    ]] #'ALL_AVG_RETURN_1D',

# VIF DATAFRAME
VIF_data = pd.DataFrame()
VIF_data['FEATURE'] = X.columns

# CALCULATING VIF FOR EACH FEATURE
VIF_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
VIF_data = VIF_data.sort_values(by='VIF', ascending=False)

print(VIF_data)

#%% [markdown]
## FEATURE SELECTION ##
# To reduce multicollinearity from all forward models, searching for any variables which may hold correlation.
# Variance inflation factor package will assist in identifying signs of multicollinearity in the data set.
# Observations:
    # Office + Residential 1-Day Average Stock Price Returns appear to be exhibit multicollinearity
    # Defensive sectors (Healthcare, Industrial, Self-Storage) share slight multicollinearity
    # Data Center sector appears largely uncorrelated to others - candidate for seasonal?
    # Lodging sector appears largely uncorrelated to others - candidate for seasonal?

#%%
## MULTIPLE LINEAR REGRESSION ##

# ASSIGN TARGET VARIABLES
X_features = sector_returns[['HOT_AVG_RETURN_1D', 'NL_AVG_RETURN_1D', 'SC_AVG_RETURN_1D', 'DC_AVG_RETURN_1D']]
X = X_features
y = sector_returns['ALL_AVG_RETURN_1D']

# X_features = sector_close
# X = sector_close
# y = sector_close['ALL_AVG_CLOSE']

print(f"X:", X.shape)
print(f"y:", y.shape)

#%%
# TRAIN / TEST - SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2, random_state=42)
print(f"X_train:", X_train.shape)
print(f"y_train:", y_train.shape)
print(f"X_test:", X_test.shape)
print(f"y_test:", y_test.shape)

#%%
### OLS MODEL ###
X_train_OLS = sm.add_constant(X_train)
OLS_model = sm.OLS(y_train, X_train_OLS)
OLS_fit = OLS_model.fit()
print(OLS_fit.summary())

#%%
## OLS COEFFICIENTS / AIC / BIC / R^2 ##
OLS_coefficients = OLS_fit.params
initial_aic_bic_rsquared = aic_bic_rsquared_df(OLS_fit)
print(OLS_coefficients)
print('*' * 50)
print(initial_aic_bic_rsquared)

#%%
## ALTERNATE REGRESSION ##
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"LR TRAIN: {round(lr.score(X_train, y_train), 4)}")
print(f"LR TEST: {round(lr.score(X_test, y_test), 4)}")

#%%
## BASE MODEL FUNCTIONS ##
def one_step_average_method(x):
    x = []
    for i in range(1,len(x)):
        m = np.mean(np.array(x[0:i]))
        x.append(m)
    return x

def h_step_average_method(train, test):
    forecast = np.mean(train)
    predictions = []
    for i in range(len(test)):
        predictions.append(forecast)
    return predictions

def one_step_naive_method(x):
    forecast = []
    for i in range(len(x)-1):
        forecast.append(x[i])
    return forecast

def h_step_naive_method(test,train):
    forecast = [test[-1] for i in range (len(train))]
    return forecast

def SES_train(yt,alpha, initial=430):
    prediction = [initial]
    for i in range(1,len(yt)):
        s= alpha*yt[i-1] + (1-alpha)*prediction[i-1]
        prediction.append(s)
    return prediction

def one_step_drift_method(x):
    forecast =[]
    for i in range(1,len(x)-1):
        prediction = x[i]+(x[i]-x[0])/i
        forecast.append(prediction)
    forecast = [x[0]] + forecast
    return forecast

def h_step_drift_method(train,test):
    forecast = []
    prediction = (train[-1] - train[0]) / (len(train)-1)
    for i in range(1,len(test) + 1):
        forecast.append(train[-1]+ i*prediction)
    return forecast

#%%
## TRAIN / TEST - SPLIT ##
split_80 = int(len(sector_returns)*0.8)
split_20 = int(len(sector_returns)-split_80)
df_train = sector_returns[:split_80]
df_test = sector_returns[split_80:]

print(f'DF_TRAIN:', df_train.shape)
print(f'DF_TEST:', df_test.shape)

#%%
## AVERAGE METHOD ##
average_method = h_step_average_method(df_train['ALL_AVG_RETURN_1D'], df_test['ALL_AVG_RETURN_1D'])

#%%
## PLOT TRAIN VS TEST ##
plt.figure()
plt.plot(df_train.index, df_train.ALL_AVG_RETURN_1D, label='TRAIN DATA', color='green')
plt.plot(df_test.index, df_test.ALL_AVG_RETURN_1D, label='TEST DATA', color='dodgerblue')
plt.plot(df_test.index, average_method, label='AVERAGE METHOD', color='yellow')
plt.title('AVERAGE METHOD ON STOCK PRICE RETURN % (1-DAY)')
plt.xlabel('DATE')
plt.ylabel('STOCK PRICE RETURN % (1-DAY)')
plt.xticks(sector_returns.index[::150], fontsize=8, rotation=90)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
## PLOT TEST ##
plt.figure()
plt.plot(df_test.index, df_test.ALL_AVG_RETURN_1D, label='TEST DATA', color='green')
plt.plot(df_test.index, average_method, label='AVERAGE METHOD', color='yellow')
plt.title('AVERAGE METHOD ON STOCK PRICE RETURN % (1-DAY)')
plt.xlabel('DATE')
plt.ylabel('STOCK PRICE RETURN % (1-DAY)')
plt.xticks(df_test.index[::30], fontsize=8, rotation=90)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
prediction, forecast = average_prediction(df_train['ALL_AVG_RETURN_1D'], len(sector_returns))

#%%
## AVERAGE METHOD ERROR ##
one_step_predict = np.array(prediction)
yarray = np.array(df_train.ALL_AVG_RETURN_1D[1:])
avg_yt_error = np.subtract(one_step_predict[2:], yarray)
avg_yf_error = np.array(df_test.ALL_AVG_RETURN_1D) - np.array(average_method)

## AVERAGE METHOD STATISTICS ##
print(f'TRAIN MSE - AVERAGE METHOD: {mse(avg_yt_error).round(4)}')
print(f'TEST MSE - AVERAGE METHOD: {mse(avg_yf_error).round(4)}')

print(f'ERROR VARIANCE - AVERAGE METHOD: {np.var(avg_yt_error)}')
print(f'ERROR MEAN - AVERAGE METHOD: {np.mean(avg_yf_error)}')
print(f'RMSE - AVERAGE METHOD: {mean_squared_error(df_test.ALL_AVG_RETURN_1D, np.array(average_method), squared=False)}')

#%%
## AVERAGE METHOD STEMPLOT ##
# stem_acf('Average-Error-ACF', acf_df(avg_yf_error, 12), len(df_train))

#%%
## RECORD MODEL RESULTS ##
model_name.append('AVERAGE-METHOD')
model_mse.append(mse(avg_yt_error).round(4))
model_ljb.append(sm.stats.acorr_ljungbox(avg_yt_error, lags=[5], boxpierce=True).iat[0,0])
model_error_var.append(np.var(avg_yt_error))
# model_notes.append('Slightly better')

#%%
## NAIVE METHOD ##
naive_predict = h_step_naive_method(df_train.ALL_AVG_RETURN_1D, df_test.ALL_AVG_RETURN_1D)

#%%
## PLOT TRAIN VS TEST ##
plt.figure()
plt.plot(df_train.index, df_train.ALL_AVG_RETURN_1D, label='TRAIN DATA', color='dodgerblue')
plt.plot(df_test.index, df_test.ALL_AVG_RETURN_1D, label='TEST DATA', color='green')
plt.plot(df_test.index, naive_predict, label='NAIVE METHOD', color='yellow')
plt.title('NAIVE METHOD ON STOCK PRICE RETURN % (1-DAY)')
plt.xlabel('DATE')
plt.ylabel('STOCK PRICE RETURN % (1-DAY)')
plt.xticks(sector_returns.index[::150], rotation=90, fontsize=10)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
## PLOT TEST ##
plt.figure()
plt.plot(df_test.index, df_test.ALL_AVG_RETURN_1D, label='TEST DATA', color='green')
plt.plot(df_test.index, naive_predict, label='NAIVE METHOD', color='yellow')
plt.title('NAIVE METHOD ON STOCK PRICE RETURN % (1-DAY)')
plt.xlabel('DATE')
plt.ylabel('STOCK PRICE RETURN % (1-DAY)')
plt.xticks(df_test.index[::30], rotation=90, fontsize=10)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
## NAIVE METHOD ERROR ##
naive_error = np.array(df_train.ALL_AVG_RETURN_1D[1:]) - np.array(one_step_naive_method(df_train.ALL_AVG_RETURN_1D))
print(f'TRAIN MSE - NAIVE METHOD: {mse(naive_error).round(4)}')
N_yf_error = np.array(df_test.ALL_AVG_RETURN_1D) - np.array(naive_predict)
print(f'TEST MSE - NAIVE METHOD: {mse(naive_error).round(4)}')

#%%
## NAIVE METHOD STATISTICS ##
print(f'ERROR VARIANCE - NAIVE METHOD: {np.var(naive_error)}')
print(f'ERROR MEAN - NAIVE METHOD: {np.mean(naive_error)}')
print(f'RMSE - NAIVE METHOD: {mean_squared_error(df_test.ALL_AVG_RETURN_1D, np.array(naive_predict), squared=False)}')

#%%
## NAIVE METHOD STATISTICS ##
print(sm.stats.acorr_ljungbox(N_yf_error, lags=[5], boxpierce=True, return_df=True))
print('PREDICTION ERROR VARIANCE APPEARS LESS THAN FORECAST ERROR VARIANCE')

#%%
## NAIVE METHOD STEMPLOT ##
# stem_acf('Stem-ACF-Naive-Err', acf_df(naive_error, 90), len(df_train))

#%%
## RECORD MODEL RESULTS ##
model_name.append('NAIVE-METHOD')
model_mse.append(mse(naive_error).round(4))
model_ljb.append(sm.stats.acorr_ljungbox(naive_error,lags=[5], boxpierce=True).iat[0,0])
model_error_var.append(np.var(naive_error))
# model_notes.append('SLIGHTLY BETTER')

#%%
## DRIFT METHOD ##
one_step_predict = one_step_drift_method(df_train.ALL_AVG_RETURN_1D)
h_step_predict = h_step_drift_method(df_train.ALL_AVG_RETURN_1D, df_test.ALL_AVG_RETURN_1D)

#%%
## PLOT TRAIN VS TEST ##
plt.figure()
plt.plot(df_train.index, df_train.ALL_AVG_RETURN_1D, label='TRAIN DATA', color='dodgerblue')
plt.plot(df_test.index, df_test.ALL_AVG_RETURN_1D, label='TEST DATA', color='green')
plt.plot(df_test.index, one_step_predict[len(df_train)-len(df_test)-1:], label='DRIFT METHOD', color='yellow')
plt.title('DRIFT METHOD ON STOCK PRICE RETURN % (1-DAY)')
plt.xlabel('DATE')
plt.ylabel('STOCK PRICE RETURN % (1-DAY)')
plt.xticks(sector_returns.index[::150], rotation=90, fontsize=10)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
## PLOT TEST ##
plt.figure()
plt.plot(df_test.index, df_test.ALL_AVG_RETURN_1D, label='TEST DATA', color='green')
plt.plot(df_test.index, one_step_predict[len(df_train)-len(df_test)-1:], label='DRIFT METHOD', color='yellow')
plt.xticks(df_test.index[::30], rotation=90, fontsize=10)
plt.xlabel('DATE')
plt.ylabel('STOCK PRICE RETURN % (1-DAY)')
plt.title('DRIFT METHOD ON STOCK PRICE RETURN % (1-DAY)')
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
## DRIFT METHOD ERROR ##
drift_yt_error = np.subtract(np.array(df_train.ALL_AVG_RETURN_1D[1:]), np.array(one_step_drift_method(df_train.ALL_AVG_RETURN_1D)))
print(f'TRAIN MSE - NAIVE METHOD: {mse(drift_yt_error).round(4)}')

drift_yf_error = np.subtract(np.array(df_test.ALL_AVG_RETURN_1D)[1:], np.array(one_step_drift_method(df_test.ALL_AVG_RETURN_1D)))
print(f'TEST MSE - NAIVE METHOD: {mse(drift_yf_error).round(4)}')

#%%
## DRIFT METHOD STATISTICS ##
print(f'TRAIN ERROR VARIANCE - DRIFT METHOD: {np.var(drift_yt_error)}')
print(f'TEST ERROR VARIANCE - DRIFT METHOD: {np.var(drift_yf_error)}')

print(f'TRAIN ERROR MEAN - DRIFT METHOD: {np.mean(drift_yt_error)}')
print(f'TEST ERROR MEAN - DRIFT METHOD: {np.mean(drift_yf_error)}')

print(f'RMSE - NAIVE METHOD: {mean_squared_error(df_test.ALL_AVG_RETURN_1D, np.array(one_step_predict)[len(df_train)-len(df_test)-1:], squared=False)}')

#%%
## DRIFT METHOD STATISTICS ##
print(sm.stats.acorr_ljungbox(drift_yf_error, lags=[5], boxpierce=True, return_df=True))
print('PREDICTION ERROR VARIANCE APPEARS LESS THAN FORECAST ERROR VARIANCE')

#%%
## DRIFT METHOD STEMPLOT ##
# stem_acf('drift-Stem-ACF-Drift-Err', acf_df(drift_yf_error, 5), len(df_train))

#%%
## RECORD MODEL RESULTS ##
model_name.append('DRIFT-METHOD')
model_mse.append(mse(drift_yt_error).round(4))
model_ljb.append(sm.stats.acorr_ljungbox(drift_yt_error, lags=[5], boxpierce=True).iat[0,0])
model_error_var.append(np.var(drift_yt_error))
# model_notes.append('Slightly better')

#%%
## SEASONAL EXPONENTIAL SMOOTHING ##
holtt = ets.ExponentialSmoothing(df_train.ALL_AVG_RETURN_1D, trend=None, damped_trend=False, seasonal=None).fit(smoothing_level=0.5)
holtf = holtt.forecast(steps=len(df_test))
holtf = pd.DataFrame(holtf)

#%%
## PLOT TRAIN VS TEST ##
plt.figure()
plt.plot(df_train.index, df_train.ALL_AVG_RETURN_1D, label='TRAIN DATA', color='dodgerblue')
plt.plot(df_test.index, df_test.ALL_AVG_RETURN_1D, label='TEST DATA', color='green')
plt.plot(df_test.index, np.array(holtf), label='SES METHOD', color='yellow')
plt.xticks(sector_returns.index[::150], rotation=90, fontsize=10)
plt.xlabel('DATE')
plt.ylabel('STOCK PRICE RETURN % (1-DAY)')
plt.title('SES METHOD ON STOCK PRICE RETURN % (1-DAY)')
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
## PLOT TEST ##
plt.figure()
plt.plot(df_test.index, df_test.ALL_AVG_RETURN_1D, label='TEST DATA', color='green')
plt.plot(df_test.index, np.array(holtf), label='SES METHOD', color='yellow')
plt.xticks(df_test.index[::30], rotation=90, fontsize=10)
plt.xlabel('DATE')
plt.ylabel('STOCK PRICE RETURN % (1-DAY)')
plt.title('SES METHOD ON STOCK PRICE RETURN % (1-DAY)')
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
## SES METHOD ERROR ##
SES_yt_error = np.subtract(np.array(df_train.ALL_AVG_RETURN_1D), SES_train(df_train.ALL_AVG_RETURN_1D, .5))
SES_yf_error = np.subtract(np.array(df_test.ALL_AVG_RETURN_1D), holtf[0])
print(f'TRAIN MSE - SES METHOD: {mse(SES_yt_error).round(4)}')
print(f'TEST MSE - SES METHOD: {mse(SES_yf_error).round(4)}')

#%%
## HOLT FORECAST ##
holtf = holtt.forecast(steps=len(df_test.ALL_AVG_RETURN_1D))
holtf = pd.DataFrame(holtf)

#%%
## HOLT FORECAST ERROR ##
print(f'TRAIN ERROR VARIANCE - DRIFT METHOD: {np.var(SES_yt_error)}')
print(f'TEST ERROR VARIANCE - DRIFT METHOD: {np.var(SES_yf_error)}')

print(f'TRAIN ERROR MEAN - DRIFT METHOD: {np.mean(SES_yf_error)}')
print(f'TEST ERROR MEAN - DRIFT METHOD: {np.mean(SES_yf_error)}')

print(f'RMSE - DRIFT METHOD: {mean_squared_error(df_test.ALL_AVG_RETURN_1D, holtf[0], squared=False)}')

#%%
## ## HOLT FORECAST STATISTICS ##
print(sm.stats.acorr_ljungbox(SES_yf_error, lags=[5], boxpierce=True, return_df=True))
print('PREDICTION ERROR VARIANCE APPEARS LESS THAN FORECAST ERROR VARIANCE')

#%%
## SES METHOD STEMPLOT ##
# stem_acf('Stem-ACF-SES-Err', acf_df(SES_yf_error, 90), len(df_train))

#%%
## RECORD MODEL RESULTS ##
model_name.append('SES-METHOD')
model_mse.append(mse(SES_yt_error).round(4))
model_ljb.append(sm.stats.acorr_ljungbox(SES_yt_error,lags=[5], boxpierce=True).iat[0,0])
model_error_var.append(np.var(SES_yt_error))
# model_notes.append('Slightly better')

#%%
## HOLT-WINTERS ##

# DEFINE FEATURES
HW_features = sector_returns[['HOT_AVG_RETURN_1D', 'NL_AVG_RETURN_1D', 'SC_AVG_RETURN_1D', 'DC_AVG_RETURN_1D', 'ALL_AVG_RETURN_1D']]

# INITIALIZE MODEL
HW_model = ets.ExponentialSmoothing(HW_features.ALL_AVG_RETURN_1D, seasonal_periods=144, trend=None, seasonal='add').fit()

#%%
# FORECAST TRAIN / TEST
HW_train = HW_model.forecast(steps=X_train.shape[0])
HW_train_df = pd.DataFrame(HW_train, columns=['ALL_AVG_RETURN_1D']).set_index(X_train.index)

HW_test = HW_model.forecast(steps=X_test.shape[0])
HW_test_df = pd.DataFrame(HW_test, columns=['ALL_AVG_RETURN_1D']).set_index(X_test.index)

#%%
# HW MODEL ASSESSMENT
HW_train_error = np.array(df_train['ALL_AVG_RETURN_1D'] - HW_train_df['ALL_AVG_RETURN_1D'])
HW_test_error = np.array(df_test['ALL_AVG_RETURN_1D'] - HW_test_df['ALL_AVG_RETURN_1D'])

print(HW_train_error)
print(HW_test_error)

#%%
# TRAIN SET PREDICTION
hw_train_mean_var = rolling_mean_var(HW_train)
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('ROLLING MEAN/VAR OF H-W PREDICTION DATA')
ax1.plot(hw_train_mean_var.index, hw_train_mean_var['ROLLING MEAN'])
ax1.set_ylabel('ROLLING MEAN')
ax2.plot(hw_train_mean_var.index, hw_train_mean_var['ROLLING VARIANCE'])
ax2.set_xlabel('DATE')
ax2.set_ylabel('ROLLING VARIANCE')
plt.show()


#%%
## RECORD MODEL RESULTS ##
model_name.append('HOLT-WINTERS')
model_mse.append(mse(HW_train_error).round(4))
model_ljb.append(sm.stats.acorr_ljungbox(HW_train_error, lags=[5], boxpierce=True).iat[0,0])
model_error_var.append(np.var(HW_train_error))
# model_notes.append('FLAT PREDICTION')

#%%
## MODEL SELECTION DATAFRAME ##
df_models = pd.DataFrame()
df_models['MODELS'] = model_name
df_models['MSE'] = model_mse
df_models['LJB'] = model_ljb
df_models['ERROR_VAR'] = model_error_var
#df_models['NOTES'] = model_notes
print(df_models.head())

#%%
## SAVE MODEL RESULTS ##
df_models.to_csv(current_folder + 'models_results.csv')


#%%


#%%
## REIT TRADING COMPARABLES ##

# * Analysis evaluates publicly traded REIT tickers, within the commercial real estate (CRE) sectors outlined below:
    #     * Retail - Strip Centers, Malls, Triple-Net Retail (NNN)
    #     * Multifamily - Rental Apartments
    #     * Office - Central Business District (CBD), Suburban (SUB)
    #     * Hospitality - Full-Service Hotels, Limited-Service Hotels
    #     * Industrial - Warehouse, Logistics


## LIBRARY IMPORTS ##
import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np

import plotly as ply
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from PIL import Image

import yfinance as yf
import datetime
# from datetime import date

# import tensorflow_technical_indicators as tfti
# from tensorflow_technical_indicators import <indicator>

# from ta.volatility import BollingerBands
# from ta.trend import MACD
# from ta.momentum import RSIIndicator

# import matplotlib.pyplot as plt
# import seaborn as sns
# import dash as dash
# from dash import dash_table
# from dash import dcc
# from dash import html
# from dash.dependencies import Input, Output
# from dash.exceptions import PreventUpdate
# import dash_bootstrap_components as dbc

# import scipy.stats as stats
# import statistics

#%%
## DIRECTORY CONFIGURATION ##
# import os
current_path = r'https://raw.githubusercontent.com/nehat312/REIT-comps/main'
basic_path = 'https://raw.githubusercontent.com/nehat312/REIT-comps/main'

# directory = os.path.dirname(current_path + '/data/')
# if not os.path.exists(directory):
#     os.makedirs(directory)

#%%
## TIME INTERVALS ##

today = datetime.date.today()
before = today - datetime.timedelta(days=1095) #700
start_date = '2000-01-01'
end_date = today #'2022-06-30'  #'2022-03-31'
mrq = '2022-06-30'
mry = '2021-12-31'

#%%
## DATA IMPORT ##
financials_csv = current_path + '/data/reit_historicals_2000_2022.csv'
trading_csv = current_path + '/data/reit_trading_2000_2022.csv'

# trading_excel = current_path + '/data/reit_trading_2000_2022.xlsx'
# financials_excel = current_path + '/data/reit_historicals_2000_2022.xlsx'
reit_financials = pd.read_csv(financials_csv, header=0, index_col='Index', infer_datetime_format=True, parse_dates=True)
# reit_trading = pd.read_csv(trading_csv, header=0, index_col='Index', infer_datetime_format=True) #, index_col='loc_rowid'


## IMAGE IMPORT ##
# jwst_tele_img_1 = Image.open('images/JWST-2.jpg')

#%%
## PRE-PROCESSING ##
reit_financials['reportPeriod'] = pd.to_datetime(reit_financials['reportPeriod'])

# reit_financials.dropna(inplace=True)
# pd.to_numeric(reit_financials['reportPeriod'])
# reit_financials['reportPeriod'].astype(int)

#%%
## REAL ESTATE SECTORS / TICKERS ##
apartment = ["EQR",	"AVB", "ESS", "MAA", "UDR",	"CPT", "AIV", "BRG"] #, "APTS"
office = ["BXP", "VNO",	"KRC", "DEI", "JBGS", "CUZ", "HPP",	"SLG",	"HIW", "OFC", "PGRE", "PDM", "WRE",	"ESRT",	"BDN", "EQC", "VRE"] #"CLI"
hotel = ["HST",	"RHP",	"PK", "APLE", "SHO", "PEB", "RLJ", "DRH", "INN", "HT", "AHT", "BHR"]    #"XHR",
mall = ["SPG", "MAC", "PEI"] #"CBL" "TCO" "WPG"
strip_center = ["REG", "FRT", "KIM", "BRX", "AKR", "UE", "ROIC", "CDR", "SITC", "BFS"]   #"WRI", "RPAI",
net_lease = ["O", "WPC", "NNN",	"STOR",	"SRC", "PINE", "FCPT", "ADC", "EPRT"]  # "VER",
industrial = ["PLD", "DRE",	"FR", "EGP"]
self_storage = ["EXR",	"CUBE",	"REXR",	"LSI"]
data_center = ["EQIX", "DLR", "AMT"] #"CONE", "COR"
healthcare = ["WELL", "PEAK", "VTR", "OHI", "HR"]   #"HTA",

sector_list_of_lists = [apartment, office, hotel, mall, strip_center, net_lease, industrial, self_storage, data_center, healthcare]
sector_list_of_names = ['apartment', 'office', 'hotel', 'mall', 'strip_center', 'net_lease', 'industrial', 'self_storage', 'data_center', 'healthcare']

reit_tickers = ["EQR", "AVB", "ESS", "MAA", "UDR", "CPT", "AIV", "BRG", #"APTS",
               "BXP", "VNO", "KRC", "DEI", "JBGS", "CUZ", "HPP", "SLG",	"HIW", "OFC", "PGRE", "PDM", "WRE", "ESRT",	"BDN", "EQC", "VRE",
               "HST", "RHP", "PK", "APLE",	"SHO",	"PEB",	"RLJ", "DRH", "INN", "HT", "AHT", "BHR",
               "SPG", "MAC", "PEI", #"SKT", "SRG", #CBL, #WPG
               "REG", "FRT", "KIM",	"BRX",	"AKR",	"UE", "ROIC", "CDR", "SITC", "BFS",
               "O", "WPC", "NNN", "STOR", "SRC", "PINE", "FCPT", "ADC", "EPRT",
               "PLD", "DRE", "FR", "EGP", #GTY
               "EXR",	"CUBE",	"REXR",	"LSI",
               "EQIX", "DLR", "AMT",
               "WELL", "PEAK", "VTR", "OHI", "HR"]

sector_dict = {'apartment': ["EQR",	"AVB", "ESS", "MAA", "UDR", "CPT",	"AIV",	"BRG"], #, "APTS"
               'office': ["BXP", "VNO",	"KRC", "DEI", "JBGS", "CUZ", "HPP",	"SLG",	"HIW", "OFC", "PGRE",	"PDM", "WRE",	"ESRT",	"BDN", "EQC", "VRE"],
               'hotel': ["HST",	"RHP",	"PK",	"APLE",	"SHO",	"PEB",	"RLJ", "DRH", "INN", "HT", "AHT",	"BHR"],
               'mall': ["SPG", "MAC", "PEI"],
               'strip_center': ["REG", "FRT",	"KIM",	"BRX",	"AKR",	"UE",	"ROIC",	"CDR",	"SITC",	"BFS"],
               'net_lease': ["O", "WPC", "NNN",	"STOR",	"SRC",  "PINE", "FCPT", "ADC", "EPRT"],
               'industrial': ["PLD", "DRE", "FR", "EGP"],
               'self_storage': ["EXR",	"CUBE",	"REXR",	"LSI"],
               'data_center': ["EQIX", "DLR", "AMT"],
               'healthcare': ["WELL", "PEAK", "VTR", "OHI", "HR"]}

ticker_output_cols = ['reportPeriod', 'ticker', 'company',  'sector', 'city', 'state',
                      'Price_Actual', #'sharePriceAdjustedClose'
                      'shares', # 'weightedAverageShares'
                      'marketCapitalization',
                      'dividendsPerBasicCommonShare', 'dividendYield',
                      'earningBeforeInterestTaxes', 'earningsBeforeInterestTaxesDepreciationAmortization',
                      'assets', 'debt', 'totalLiabilities', 'cashAndEquivalents',
                      'enterpriseValue', 'enterpriseValueOverEBIT', 'enterpriseValueOverEBITDA',
                      'capitalExpenditure', 'investedCapital', 'investments', 'propertyPlantEquipmentNet',
                      'netCashFlow', 'netCashFlowBusinessAcquisitionsDisposals',
                      'profitMargin', 'payoutRatio', 'priceToEarningsRatio',
                      'priceToBookValue', 'tangibleAssetValue',
                      #'shareBasedCompensation', 'sellingGeneralAndAdministrativeExpense',
                      'netIncome', 'netIncomeToNonControllingInterests']

scatter_cols_5x5 = ['Price_Actual', 'netIncome', 'earningBeforeInterestTaxes',
                    'marketCapitalization', 'dividendYield',]

model_cols = ['ticker', 'calendarDate', 'sector', 'company', 'city', 'state',
              'Price_Actual', 'Price_LQ', 'Delta_QoQ', 'Return_QoQ', #'closePrice',
              'operatingIncome', 'operatingExpenses', 'netIncome', 'earningBeforeInterestTaxes',
              'revenues', 'costOfRevenue',
              'assets', 'debt', 'cashAndEquivalents', 'debtToEquityRatio',
              'inventory', 'investments', 'investedCapital', 'propertyPlantEquipmentNet', # 'researchAndDevelopmentExpense',
              'tangibleAssetsBookValuePerShare', 'interestExpense', 'incomeTaxExpense',
              'shares', 'weightedAverageShares',
              'Year', 'Month', 'Quarter'
              ]

mil_cols = ['operatingIncome', 'operatingExpenses', 'netIncome',
            'earningBeforeInterestTaxes',
            'revenues', 'assets', 'debt', 'cashAndEquivalents',
            'inventory', 'investments', 'investedCapital', 'propertyPlantEquipmentNet',
            'interestExpense', 'incomeTaxExpense',
            'shares', 'weightedAverageShares',
            ]

ticker_output_df = reit_financials[ticker_output_cols]

# reit_comps = reit_comps[model_cols]

# mo_qtr_map = {'01': '1', '02': '1', '03': '1',
#               '04': '2', '05': '2', '06': '2',
#               '07': '3', '08': '3', '09': '3',
#               '10': '4', '11': '4', '12': '4'}


## DATA SCRAPE ##

reit_scrape_path = current_path + '/REIT-scrape.py'

# %cd $abspath_util_deep

# %run reit_scrape.py


#%%
# STOCK PRICE TRADING HISTORY
all_reits_trading = yf.download(tickers = reit_tickers,
        period = "max", # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval = "1d", # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        start = start_date, #'2000-01-01'
        end = today,
        group_by = 'column',
        auto_adjust = True,
        prepost = False,
        threads = True,
        proxy = None, #"PROXY_SERVER"
        timeout=12)

apartment_reits_trading = yf.download(tickers = apartment,
        period = "max", # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval = "1d", # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        start = start_date, #'2000-01-01'
        end = today,
        group_by = 'column',
        auto_adjust = True,
        prepost = False,
        threads = True,
        proxy = None,
        timeout=12)


office_reits_trading = yf.download(tickers = office,
        period = "max", # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval = "1d", # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        start = start_date, #'2000-01-01'
        end = today,
        group_by = 'column',
        auto_adjust = True,
        prepost = False,
        threads = True,
        proxy = None,
        timeout=12)

hotel_reits_trading = yf.download(tickers = hotel,
        period = "max", # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval = "1d", # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        start = start_date, #'2000-01-01'
        end = today,
        group_by = 'column',
        auto_adjust = True,
        prepost = False,
        threads = True,
        proxy = None,
        timeout=12)

mall_reits_trading = yf.download(tickers = mall,
        period = "max", # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval = "1d", # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        start = start_date, #'2000-01-01'
        end = today,
        group_by = 'column',
        auto_adjust = True,
        prepost = False,
        threads = True,
        proxy = None,
        timeout=12)

strip_center_reits_trading = yf.download(tickers = strip_center,
        period = "max", # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval = "1d", # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        start = start_date, #'2000-01-01'
        end = today,
        group_by = 'column',
        auto_adjust = True,
        prepost = False,
        threads = True,
        proxy = None,
        timeout=12)

net_lease_reits_trading = yf.download(tickers = net_lease,
        period = "max", # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval = "1d", # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        start = start_date, #'2000-01-01'
        end = today,
        group_by = 'column',
        auto_adjust = True,
        prepost = False,
        threads = True,
        proxy = None,
        timeout=12)

industrial_reits_trading = yf.download(tickers = industrial,
        period = "max", # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval = "1d", # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        start = start_date, #'2000-01-01'
        end = today,
        group_by = 'column',
        auto_adjust = True,
        prepost = False,
        threads = True,
        proxy = None,
        timeout=12)

self_storage_reits_trading = yf.download(tickers = self_storage,
        period = "max", # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval = "1d", # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        start = start_date, #'2000-01-01'
        end = today,
        group_by = 'column',
        auto_adjust = True,
        prepost = False,
        threads = True,
        proxy = None,
        timeout=12)

data_center_reits_trading = yf.download(tickers = data_center,
        period = "max", # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval = "1d", # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        start = start_date, #'2000-01-01'
        end = today,
        group_by = 'column',
        auto_adjust = True,
        prepost = False,
        threads = True,
        proxy = None,
        timeout=12)

healthcare_reits_trading = yf.download(tickers = healthcare,
        period = "max", # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval = "1d", # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        start = start_date, #'2000-01-01'
        end = today,
        group_by = 'column',
        auto_adjust = True,
        prepost = False,
        threads = True,
        proxy = None,
        timeout=12)

#%%
## VARIABLE ASSIGNMENT ##
all_reits_close = all_reits_trading['Close']
all_reits_close_df = pd.DataFrame(all_reits_close)
all_reits_open = all_reits_trading['Open']
all_reits_volume = all_reits_trading['Volume']

apartment_reits_close = apartment_reits_trading['Close']
apartment_reits_close['apartment_avg'] = apartment_reits_close.mean(axis=1)
# apartment_reits_close.mean(axis=1, out=apartment_reits_close['apartment_avg'])
apartment_reits_close_df = pd.DataFrame(apartment_reits_close)
apartment_reits_open = apartment_reits_trading['Open']
apartment_reits_volume = apartment_reits_trading['Volume']
# apartment_reits_close['ticker'] = apartment_reits_close.index
# apartment_reits_close['sector'] = apartment_reits_close['ticker'].map(sector_dict)


office_reits_close = office_reits_trading['Close']
office_reits_close['office_avg'] = office_reits_close.mean(axis=1)
office_reits_close_df = pd.DataFrame(office_reits_close)
office_reits_open = office_reits_trading['Open']
office_reits_volume = office_reits_trading['Volume']

hotel_reits_close = hotel_reits_trading['Close']
hotel_reits_close['hotel_avg'] = hotel_reits_close.mean(axis=1)
hotel_reits_close_df = pd.DataFrame(hotel_reits_close)
hotel_reits_open = hotel_reits_trading['Open']
hotel_reits_volume = hotel_reits_trading['Volume']

mall_reits_close = mall_reits_trading['Close']
mall_reits_close['mall_avg'] = mall_reits_close.mean(axis=1)
mall_reits_close_df = pd.DataFrame(mall_reits_close)
mall_reits_open = mall_reits_trading['Open']
mall_reits_volume = mall_reits_trading['Volume']

strip_center_reits_close = strip_center_reits_trading['Close']
strip_center_reits_close['strip_center_avg'] = strip_center_reits_close.mean(axis=1)
strip_center_reits_close_df = pd.DataFrame(strip_center_reits_close)
strip_center_reits_open = strip_center_reits_trading['Open']
strip_center_reits_volume = strip_center_reits_trading['Volume']

net_lease_reits_close = net_lease_reits_trading['Close']
net_lease_reits_close['net_lease_avg'] = net_lease_reits_close.mean(axis=1)
net_lease_reits_close_df = pd.DataFrame(net_lease_reits_close)
net_lease_reits_open = net_lease_reits_trading['Open']
net_lease_reits_volume = net_lease_reits_trading['Volume']

industrial_reits_close = industrial_reits_trading['Close']
industrial_reits_close['industrial_avg'] = industrial_reits_close.mean(axis=1)
industrial_reits_close_df = pd.DataFrame(industrial_reits_close)
industrial_reits_open = industrial_reits_trading['Open']
industrial_reits_volume = industrial_reits_trading['Volume']

self_storage_reits_close = self_storage_reits_trading['Close']
self_storage_reits_close['self_storage_avg'] = self_storage_reits_close.mean(axis=1)
self_storage_reits_close_df = pd.DataFrame(self_storage_reits_close)
self_storage_reits_open = self_storage_reits_trading['Open']
self_storage_reits_volume = self_storage_reits_trading['Volume']

data_center_reits_close = data_center_reits_trading['Close']
data_center_reits_close['data_center_avg'] = data_center_reits_close.mean(axis=1)
data_center_reits_close_df = pd.DataFrame(data_center_reits_close)
data_center_reits_open = data_center_reits_trading['Open']
data_center_reits_volume = data_center_reits_trading['Volume']

healthcare_reits_close = healthcare_reits_trading['Close']
healthcare_reits_close['healthcare_avg'] = healthcare_reits_close.mean(axis=1)
healthcare_reits_close_df = pd.DataFrame(healthcare_reits_close)
healthcare_reits_open = healthcare_reits_trading['Open']
healthcare_reits_volume = healthcare_reits_trading['Volume']


#%%
all_sectors_close_df = pd.DataFrame([apartment_reits_close['apartment_avg'], office_reits_close['office_avg'], hotel_reits_close['hotel_avg'],
                                mall_reits_close['mall_avg'], strip_center_reits_close['strip_center_avg'], net_lease_reits_close['net_lease_avg'],
                                industrial_reits_close['industrial_avg'], self_storage_reits_close['self_storage_avg'],
                                data_center_reits_close['data_center_avg'], healthcare_reits_close['healthcare_avg']])

    # pd.concat([apartment_reits_close['apartment_avg'], office_reits_close['office_avg']], ignore_index=False, axis=0)
                                # hotel_reits_close['hotel_avg'],
                                # mall_reits_close['mall_avg'], strip_center_reits_close['strip_center_avg'], net_lease_reits_close['net_lease_avg'],
                                # industrial_reits_close['industrial_avg'], self_storage_reits_close['self_storage_avg'],
                                # data_center_reits_close['data_center_avg'], healthcare_reits_close['healthcare_avg']])

all_sectors_close_df = all_sectors_close_df.T


#%%
# all_sectors_close_df.info()

#%%
## SECTOR GROUPBY -- TRADING ##
# office_financials_group = ticker_output_df[ticker_output_df['sector'] == 'OFFICE']
# # all_reits_close_group = all_reits_close.groupby(['sector'], as_index=False)['marketCapitalization'].sum() #, 'reportPeriod'
# all_reits_close_group = all_reits_close.groupby(['sector'], as_index=True).mean() #, 'reportPeriod'
# all_reits_close_group = all_reits_close_group.T
# all_reits_close_group.index = pd.to_datetime(all_reits_close_group.index)
#
# # all_reits_close_group.index = pd.to_datetime(all_reits_close_group.index)
#
# ## SECTOR GROUPBY -- FINANCIALS ##
# office_financials_group = ticker_output_df[ticker_output_df['sector'] == 'OFFICE']
# sector_mkt_cap_group = ticker_output_df.groupby(['sector', 'reportPeriod'], as_index=False)['marketCapitalization'].sum()
# sector_multiples_group = ticker_output_df.groupby(['sector', 'reportPeriod'], as_index=False)['enterpriseValueOverEBIT', 'enterpriseValueOverEBITDA'].sum()
# sector_ratios_group = ticker_output_df.groupby(['sector', 'reportPeriod'], as_index=False)['profitMargin', 'payoutRatio', 'priceToEarningsRatio'].mean()
#
# # print(all_reits_close_group.info())
#
# # print(sector_mkt_cap)
# # print(sector_multiples[:30])
# # print(sector_ratios_group)

#%%
# map_list_all_sectors = []
# for i in all_reits_close.index.values:
#     for k in sector_dict:
#         if i in sector_dict[k]:
#             map_list_all_sectors.append(k)
#             break

# map_list_apartment = []
# for i in apartment_reits_close.index.values:
#     for k in sector_dict:
#         if i in sector_dict[k]:
#             map_list_apartment.append(k)
#             break

#%%
## QUARTERLY BALANCE SHEETS - MRY ##

yf_tickers = []
for i in reit_tickers:
    i = yf.Ticker(f'{i}')
    yf_tickers.append(i)

yf_apartment = []
for i in apartment:
    i = yf.Ticker(f'{i}')
    yf_apartment.append(i)

yf_office = []
for i in office:
    i = yf.Ticker(f'{i}')
    yf_office.append(i)

yf_hotel = []
for i in hotel:
    i = yf.Ticker(f'{i}')
    yf_hotel.append(i)

yf_mall = []
for i in mall:
    i = yf.Ticker(f'{i}')
    yf_mall.append(i)

yf_strip_center = []
for i in strip_center:
    i = yf.Ticker(f'{i}')
    yf_strip_center.append(i)

yf_net_lease = []
for i in net_lease:
    i = yf.Ticker(f'{i}')
    yf_net_lease.append(i)

yf_industrial = []
for i in industrial:
    i = yf.Ticker(f'{i}')
    yf_industrial.append(i)

yf_self_storage = []
for i in self_storage:
    i = yf.Ticker(f'{i}')
    yf_self_storage.append(i)

yf_data_center = []
for i in data_center:
    i = yf.Ticker(f'{i}')
    yf_data_center.append(i)

yf_healthcare = []
for i in healthcare:
    i = yf.Ticker(f'{i}')
    yf_healthcare.append(i)

#     print(str(i).upper())

#%%
yf_ticker_dict = {'yfinance.Ticker object <EQR>':'EQR', 'yfinance.Ticker object <AVB>':'AVB', 'yfinance.Ticker object <ESS>':'ESS', 'yfinance.Ticker object <MAA>':'MAA', 'yfinance.Ticker object <UDR>':'UDR', 'yfinance.Ticker object <CPT>':'CPT', 'yfinance.Ticker object <AIV>':'AIV', 'yfinance.Ticker object <BRG>':'BRG', # 'yfinance.Ticker object <APTS>':'APTS',
                  'yfinance.Ticker object <BXP>':'BXP', 'yfinance.Ticker object <VNO>':'VNO', 'yfinance.Ticker object <KRC>':'KRC', 'yfinance.Ticker object <DEI>':'DEI', 'yfinance.Ticker object <JBGS>':'JBGS', 'yfinance.Ticker object <CUZ>':'CUZ', 'yfinance.Ticker object <HPP>':'HPP', 'yfinance.Ticker object <SLG>':'SLG', 'yfinance.Ticker object <HIW>':'HIW', 'yfinance.Ticker object <OFC>':'OFC', 'yfinance.Ticker object <PGRE>':'PGRE', 'yfinance.Ticker object <PDM>':'PDM', 'yfinance.Ticker object <WRE>':'WRE', 'yfinance.Ticker object <ESRT>':'ESRT', 'yfinance.Ticker object <BDN>':'BDN', 'yfinance.Ticker object <EQC>':'EQC', 'yfinance.Ticker object <VRE>':'VRE',
                  'yfinance.Ticker object <HST>':'HST', 'yfinance.Ticker object <RHP>':'RHP', 'yfinance.Ticker object <PK>':'PK', 'yfinance.Ticker object <APLE>':'APLE', 'yfinance.Ticker object <SHO>':'SHO', 'yfinance.Ticker object <PEB>':'PEB', 'yfinance.Ticker object <RLJ>':'RLJ', 'yfinance.Ticker object <DRH>':'DRH', 'yfinance.Ticker object <INN>':'INN', 'yfinance.Ticker object <HT>':'HT', 'yfinance.Ticker object <AHT>':'AHT', 'yfinance.Ticker object <BHR>':'BHR',
                  'yfinance.Ticker object <SPG>':'SPG', 'yfinance.Ticker object <MAC>':'MAC', 'yfinance.Ticker object <PEI>':'PEI',
                  'yfinance.Ticker object <REG>':'REG', 'yfinance.Ticker object <FRT>':'FRT', 'yfinance.Ticker object <KIM>':'KIM', 'yfinance.Ticker object <BRX>':'BRX', 'yfinance.Ticker object <AKR>':'AKR', 'yfinance.Ticker object <UE>':'UE', 'yfinance.Ticker object <ROIC>':'ROIC', 'yfinance.Ticker object <CDR>':'CDR', 'yfinance.Ticker object <SITC>':'SITC', 'yfinance.Ticker object <BFS>':'BFS',
                  'yfinance.Ticker object <O>':'O', 'yfinance.Ticker object <WPC>':'WPC', 'yfinance.Ticker object <NNN>':'NNN', 'yfinance.Ticker object <STOR>':'STOR', 'yfinance.Ticker object <SRC>':'SRC', 'yfinance.Ticker object <PINE>':'PINE', 'yfinance.Ticker object <FCPT>':'FCPT', 'yfinance.Ticker object <ADC>':'ADC', 'yfinance.Ticker object <EPRT>':'EPRT',
                  'yfinance.Ticker object <PLD>':'PLD', 'yfinance.Ticker object <DRE>':'DRE', 'yfinance.Ticker object <FR>':'FR', 'yfinance.Ticker object <EGP>':'EGP',
                  'yfinance.Ticker object <EXR>':'EXR', 'yfinance.Ticker object <CUBE>':'CUBE', 'yfinance.Ticker object <REXR>':'REXR', 'yfinance.Ticker object <LSI>':'LSI',
                  'yfinance.Ticker object <EQIX>':'EQIX', 'yfinance.Ticker object <DLR>':'DLR', 'yfinance.Ticker object <AMT>':'AMT',
                  'yfinance.Ticker object <WELL>':'WELL', 'yfinance.Ticker object <PEAK>':'PEAK', 'yfinance.Ticker object <VTR>':'VTR', 'yfinance.Ticker object <OHI>':'OHI', 'yfinance.Ticker object <HR>':'HR',
                  }


#%%
# ## APARTMENT ##
# apartment_common_so_temp = pd.DataFrame()
# apartment_assets_temp = pd.DataFrame()
# apartment_liabilities_temp = pd.DataFrame()
# apartment_nci_temp = pd.DataFrame()
# apartment_sh_equity_temp = pd.DataFrame()
# apartment_other_sh_equity_temp = pd.DataFrame()
# apartment_lt_debt_temp = pd.DataFrame()
# apartment_st_lt_debt_temp = pd.DataFrame()
# apartment_cash_temp = pd.DataFrame()
# apartment_net_tangible_temp = pd.DataFrame()
# # apartment_cap_surplus_temp = pd.DataFrame()
# # _df_temp = pd.DataFrame()
#
# for j in yf_apartment:
#     apartment_common_so_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Common Stock', mrq]
#     apartment_assets_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Total Assets', mrq]
#     apartment_liabilities_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Total Liab', mrq]
#     apartment_nci_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Minority Interest', mrq]
#     apartment_sh_equity_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Total Stockholder Equity', mrq]
#         # apartment_other_sh_equity_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Other Stockholder Equity', mrq]
#     apartment_lt_debt_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Long Term Debt', mrq]
#         # apartment_st_lt_debt_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Short Long Term Debt', mrq]
#         # apartment_ttl_debt_temp = j.quarterly_balance_sheet.loc['Long Term Debt', mrq] + j.quarterly_balance_sheet.loc['Short Long Term Debt', mrq]
#         # cap_surplus_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Capital Surplus', mrq]
#     apartment_cash_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Cash', mrq]
#     apartment_net_tangible_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Net Tangible Assets', mrq]
#
#     # _df_temp[f'{j}'] = j.quarterly_balance_sheet.loc['', mrq]
#
# #%%
#
# apartment_common_so_df = apartment_common_so_temp.rename(columns=yf_ticker_dict)
# apartment_assets_df = apartment_assets_temp.rename(columns=yf_ticker_dict)
# apartment_liabilities_df = apartment_liabilities_temp.rename(columns=yf_ticker_dict)
# apartment_nci_df = apartment_nci_temp.rename(columns=yf_ticker_dict)
# apartment_sh_equity_df = apartment_sh_equity_temp.rename(columns=yf_ticker_dict)
# # apartment_other_sh_equity_df = apartment_other_sh_equity_temp.rename(columns=yf_ticker_dict)
# apartment_lt_debt_df = apartment_lt_debt_temp.rename(columns=yf_ticker_dict)
# # apartment_st_lt_debt_df = apartment_st_lt_debt_temp.rename(columns=yf_ticker_dict)
# apartment_cash_df = apartment_cash_temp.rename(columns=yf_ticker_dict)
# apartment_net_tangible_df = apartment_net_tangible_temp.rename(columns=yf_ticker_dict)
#
# ## COMBINE BY SECTOR ##
# apartment_cap_table = pd.concat([apartment_common_so_df, apartment_assets_df, apartment_liabilities_df,
#                                      apartment_nci_df, apartment_sh_equity_df, apartment_lt_debt_df, #apartment_st_lt_debt_df, #apartment_other_sh_equity_df,
#                                      apartment_cash_df, apartment_net_tangible_df],
#                                     keys=['SHARES', 'TTL_ASSETS', 'TTL_LIABILITIES', 'NCI',
#                                           'SH_EQUITY', 'LT_DEBT', 'CASH', 'NET_TBV'])


## CAPITALIZATION TABLE ##

    # PRICE
    # Total Equity Mkt Capitalization
    # Preferred??

    # Total Debt Capitalization
    # Equity Cap + Debt Cap = Total Mkt Cap
    # Enterprise Value == Total Mkt Cap - Cash

## *UNUSED* ##
    # 'Other Current Assets'
    # 'Other Current Liab'
    # 'Total Current Assets'
    # 'Total Current Liabilities'
    # 'Long Term Investments'

# apartment_cap_table_T = apartment_cap_table.T
# # apartment_cap_table_T.rename(columns=['SHARES1', 'SHARES2', 'SHARES3', 'SHARES4',
# #                                       'SHARES5', 'SHARES6', 'SHARES7', 'SHARES8'])
#
# apartment_cap_table_T.columns = apartment_cap_table_T.columns.droplevel(1)
# sector_cap_tables={}
# apt_dict = apartment_cap_table_T.to_dict('sector_cap_tables')

# sector_cap_tables['apt'] = {'apartment':apartment_cap_table_T}


#%%
# print(sector_cap_tables[apartment])
# print(apt_dict)


# print(apartment_cap_table_T.info())
# # print(apartment_cap_table_T.head())
# print(apartment_cap_table_T)

#%%
## STACK
# apartment_stack = apartment_cap_table_T.stack()
# apartment_stack

#%%
## PENDING / WAITLIST ##

## TOTAL RETURN ##
    # returns = {}
    # for stock in apartment_reits_close.columns:
    #     apartment_reits_close[f'{stock}_return'] = apartment_reits_close[stock].dropna().iloc[-1] / apartment_reits_close[stock].dropna().iloc[0]
        # returns[stock] = apartment_reits_close[stock].dropna().iloc[-1] / apartment_reits_close[stock].dropna().iloc[0]
        # print(returns)

## EXPORT HISTORICAL TRADING DATA ##
## IMPORT DATA (UNIQUE DATAFRAMES FOR EACH CRE SECTOR??) ##
## SAVE COPIES OF IMPORTS ##
## TOOLBOX FUNCTIONS ##


#%%
##################
# FORMAT / STYLE #
##################

## COLOR SCALES ##
Sunset = px.colors.sequential.Sunset
Sunsetdark = px.colors.sequential.Sunsetdark
Tropic = px.colors.diverging.Tropic
Temps = px.colors.diverging.Temps
Tealrose = px.colors.diverging.Tealrose
Blackbody = px.colors.sequential.Blackbody
Ice = px.colors.sequential.ice
Ice_r = px.colors.sequential.ice_r
Dense = px.colors.sequential.dense
# YlOrRd = px.colors.sequential.YlOrRd
# Mint = px.colors.sequential.Mint
# Electric = px.colors.sequential.Electric

## SECTOR COLORS ##

sector_colors = {'apartment':'#FFDF00',
                 'office':'#29609C',
                 'hotel':'#E9EDED',
                 'mall':'#D5FF0A',
                 'strip_center':'#46D8BF',
                 'net_lease':'#EEFCF7',
                 'industrial':'#535865',
                 'self_storage':'#5F8C95',
                 'data_center':'#3AA5C3',
                 'healthcare':'#FF3363',

                 'apartment_avg':'#FFDF00',
                 'office_avg':'#29609C',
                 'hotel_avg':'#E9EDED',
                 'mall_avg':'#D5FF0A',
                 'strip_center_avg':'#46D8BF',
                 'net_lease_avg':'#EEFCF7',
                 'industrial_avg':'#535865',
                 'self_storage_avg':'#5F8C95',
                 'data_center_avg':'#3AA5C3',
                 'healthcare_avg':'#FF3363',
                 }

## VISUALIZATION LABELS ##
chart_labels = {'apartment':'APARTMENT',
                'office':'OFFICE',
                'hotel':'LODGING',
                'strip_center':'STRIP_CENTER',
                'net_lease':'NET LEASE',
                'mall':'MALL',
                'industrial':'INDUSTRIAL',
                'self_storage':'SELF-STORAGE',
                'data_center':'DATA CENTER',
                'healthcare':'HEALTHCARE',

                'apartment_avg':'APARTMENT',
                'office_avg':'OFFICE',
                'hotel_avg':'LODGING',
                'strip_center_avg':'STRIP_CENTER',
                'net_lease_avg':'NET LEASE',
                'mall_avg':'MALL',
                'industrial_avg':'INDUSTRIAL',
                'self_storage_avg':'SELF-STORAGE',
                'data_center_avg':'DATA CENTER',
                'healthcare_avg':'HEALTHCARE',
                # 'value':'SHARE PRICE ($)',

                'reportPeriod':'REPORT PERIOD',
                'ticker':'TICKER',
                'company':'COMPANY',
                'city':'CITY',
                'state':'STATE',

                'Price_Actual':'SHARE PRICE ($)',
                'sharePriceAdjustedClose':'ADJ. CLOSE PRICE ($)',
                'shares':'S/O',
                'weightedAverageShares':'S/O (WTD AVG)',
                'marketCapitalization':'MARKET CAP.',
                'dividendsPerBasicCommonShare':'DIV./SHARE ($)',
                'dividendYield':'DIV. YIELD (%)',
                'earningBeforeInterestTaxes':'EBIT',
                'earningsBeforeInterestTaxesDepreciationAmortization':'EBITDA',
                'assets':'ASSETS',
                'debt':'DEBT',
                'totalLiabilities':'LIABILITIES',
                'cashAndEquivalents':'CASH',
                'enterpriseValue':'EV',
                'enterpriseValueOverEBIT':'EV/EBIT',
                'enterpriseValueOverEBITDA':'EV/EBITDA',

                'capitalExpenditure':'CAPEX',
                'investedCapital':'CAPITAL INVESTED',
                'investments':'INVESTMENTS',
                'propertyPlantEquipmentNet':'NET PP&E',

                'netCashFlow':'NET CASH FLOW',
                'netCashFlowBusinessAcquisitionsDisposals':'NET ACQ./DISP.',
                'profitMargin':'PROFIT MARGIN (%)',
                'payoutRatio':'PAYOUT RATIO (%)',
                'priceToEarningsRatio':'P/E RATIO',
                'priceToBookValue':'PRICE/BV',
                'tangibleAssetValue':'TANGIBLE ASSET VALUE',
                'shareBasedCompensation':'EQUITY-BASED COMP.',
                'sellingGeneralAndAdministrativeExpense':'SG&A EXP.',
                'netIncome':'NET INCOME',
                'netIncomeToNonControllingInterests':'NCI',
                }

#%%
## VISUALIZATIONS ##
reit_scatter_matrix = px.scatter_matrix(ticker_output_df,
                                     dimensions=scatter_cols_5x5,
                                     color=ticker_output_df['sector'],
                                     # color_continuous_scale=Temps,
                                     # color_discrete_sequence=Temps,
                                     color_discrete_map=sector_colors,
                                     hover_name=ticker_output_df['company'],
                                     hover_data=ticker_output_df[['sector','reportPeriod',]],
                                     title='REIT COMPS SCATTER MATRIX',
                                     labels=chart_labels,
                                 height=1000,
                                 width=1000,
                                 )

sector_market_cap_line = px.line(ticker_output_df,
                                 x=ticker_output_df['reportPeriod'],
                                 y=ticker_output_df['marketCapitalization'],
                                 color=ticker_output_df['sector'],
                                 # color_continuous_scale=Ice_r,
                                 color_discrete_sequence=Ice_r,
                                 color_discrete_map=sector_colors,
                                 hover_name=ticker_output_df['company'],
                                 hover_data=ticker_output_df[['sector','reportPeriod']],
                                 title='REIT SECTORS MARKET CAPITALIZATION',
                                 labels=chart_labels,
                                 height=1000,
                                 width=1000,
                                 )

# ticker_price_line = px.line(ticker_output_df[ticker_input],
#                                  x=ticker_output_df['reportPeriod'],
#                                  y=ticker_output_df['marketCapitalization'],
#                                      color=ticker_output_df['sector'],
#                                      # color_continuous_scale=Electric,
#                                      color_discrete_sequence=Electric,
#                                      hover_name=ticker_output_df['company'],
#                                      hover_data=ticker_output_df[['sector','reportPeriod']],
#                                      title='REIT SECTORS MARKET CAPITALIZATION',
#                                      labels=chart_labels,
#                                  height=800,
#                                  # width=600,
#                                  )


# ticker_price_line = px.line(ticker_output_df[ticker_input],
#                                  x=ticker_output_df['reportPeriod'],
#                                  y=ticker_output_df['marketCapitalization'],
#                                      color=ticker_output_df['sector'],
#                                      # color_continuous_scale=Electric,
#                                      color_discrete_sequence=Electric,
#                                      hover_name=ticker_output_df['company'],
#                                      hover_data=ticker_output_df[['sector','reportPeriod']],
#                                      title='REIT SECTORS MARKET CAPITALIZATION',
#                                      labels=chart_labels,
#                                  height=800,
#                                  # width=600,
#                                  )

# sector_heatmap = px.density_heatmap(
#                                  x=all_reits_volume.index,
#                                  y=all_reits_close,
#                                     z=all_reits_volume,
#                                      # color=all_reits_volume['sector'],
#                                      # color_continuous_scale=Electric,
#                                      color_discrete_sequence=Electric,
#                                      hover_name=ticker_output_df['company'],
#                                      hover_data=ticker_output_df[['sector','reportPeriod']],
#                                      title='REIT SECTORS MARKET CAPITALIZATION',
#                                      labels=chart_labels,
#                                  height=800,
#                                  # width=600,
#                                  )

# reit_density_map = px.density_contour(ticker_output_df,
#                                    title='REIT ',
#                                    labels=chart_labels,
#                                    )


# scatter_3d_1 = px.scatter_3d(reit_financials,
#                              x=reit_financials['ra'],
#                              y=reit_financials['dec'],
#                              z=reit_financials['sy_distance_pc'],
#                              color=reit_financials['st_temp_eff_k'],
#                              color_discrete_sequence=Ice_r,
#                              color_continuous_scale=Ice_r,
#                              color_continuous_midpoint=5000,
#                              size=reit_financials['pl_rade'],
#                              size_max=50,
#                              # symbol=exo_drop_na['disc_year'],
#                              hover_name=reit_financials['pl_name'],
#                              hover_data=reit_financials[['host_name', 'disc_facility', 'disc_telescope']],
#                              title='EXOPLANET POPULATION -- RIGHT ASCENSION / DECLINATION / DISTANCE',
#                              labels=chart_labels,
#                              # range_x=[0,360],
#                              # range_y=[-50,50],
#                              range_z=[0,2500],
#                              # range_color=Sunsetdark,
#                              opacity=.8,
#                              height=800,
#                              width=1600,
#                              )

# disc_info_1 = px.histogram(disc_facility_filter,
#                            y=disc_facility_filter['disc_facility'],
#                            color=disc_facility_filter['disc_method'],
#                            color_discrete_sequence=Ice_r,
#                            hover_name=disc_facility_filter['pl_name'],
#                            hover_data=disc_facility_filter[['host_name', 'disc_facility', 'disc_telescope', 'sy_star_count', 'sy_planet_count']],
#                            # animation_frame=disc_facility_filter['disc_year'],
#                            # animation_group=disc_facility_filter['disc_facility'],
#                            title='EXOPLANET DISCOVERY FACILITY (BY DISCOVERY METHOD)',
#                            labels=chart_labels,
#                            range_x=[0,2500],
#                            height=1000,
#                            # width=800,
#                            )


#%%
#####################
### STREAMLIT APP ###
#####################

## CONFIGURATION ##
st.set_page_config(page_title='REIT PUBLIC TRADING COMPS',
                   layout='wide',
                   initial_sidebar_state='auto') #, page_icon=":emoji:"

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """

st.markdown(hide_menu_style, unsafe_allow_html=True)

## CSS CUSTOMIZATION ##
th_props = [('font-size', '12px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('color', '#EBEDE9'), #6d6d6d #29609C
            ('background-color', '#29609C') #f7f7f9
            ]

td_props = [('font-size', '12px'),
            # ('text-align', 'center'),
            # ('font-weight', 'bold'),
            # ('color', '#EBEDE9'), #6d6d6d #29609C
            # ('background-color', '#29609C') #f7f7f9
            ]

df_styles = [dict(selector="th", props=th_props),
             dict(selector="td", props=td_props)]


col_format_dict = {'profitMargin': "{:.1%}", 'payoutRatio': "{:.1%}", 'dividendYield': "{:.1%}",
                   'dividendsPerBasicCommonShare': "${:.2}", #'Price_Actual': "${:.2}",
                   'priceToEarningsRatio': "{:.1}x", 'priceToBookValue': "{:.1}x",
                   'enterpriseValueOverEBIT': "{:.1}x", 'enterpriseValueOverEBITDA': "{:.1}x",
                   'shares': "{:,}",
                   'marketCapitalization': "${:,}",
                   'earningBeforeInterestTaxes': "${:,}",
                   'earningsBeforeInterestTaxesDepreciationAmortization': "${:,}",
                   'assets': "${:,}", 'debt': "${:,}", 'totalLiabilities': "${:,}",
                   'cashAndEquivalents': "${:,}",
                   'netIncome': "${:,}", 'netIncomeToNonControllingInterests': "${:,}",
                   'enterpriseValue': "${:,}", 'netCashFlow': "${:,}",
                   'capitalExpenditure': "${:,}", 'netCashFlowBusinessAcquisitionsDisposals': "${:,}",
                   'investedCapital': "${:,}", 'investments': "${:,}",
                   'propertyPlantEquipmentNet': "${:,}", 'tangibleAssetValue': "${:,}",
                   }

## SIDEBAR (WIP) ##
# sector_sidebar_select = st.sidebar.selectbox('SECTOR', (sector_list_of_names), help='SELECT CRE SECTOR')
# ticker_sidebar_select = st.sidebar.selectbox('TICKER', (sector_dict['apartment'])) #sector_sidebar_select
sidebar_header = st.sidebar.subheader('VISUALIZATION TIMEFRAME:')
sidebar_start = st.sidebar.date_input('START DATE', before)
sidebar_end = st.sidebar.date_input('END DATE', today)
if sidebar_start < sidebar_end:
    st.sidebar.success('START DATE: `%s`\n\nEND DATE: `%s`' % (sidebar_start, sidebar_end))
else:
    st.sidebar.error('ERROR: END DATE BEFORE START DATE')

## HEADER ##
st.container()

## EXTERNAL LINKS ##
github_link = '[GITHUB REPOSITORY](https://github.com/nehat312/REIT-comps/)'
propswap_link = '[PROP/SWAP](<TBU>)'
tbu_link = '[TBU](<TBU>)'

link_col_1, link_col_2, link_col_3 = st.columns(3)
ext_link_1 = link_col_1.markdown(github_link, unsafe_allow_html=True)
ext_link_2 = link_col_2.markdown(propswap_link, unsafe_allow_html=True)
ext_link_3 = link_col_3.markdown(tbu_link, unsafe_allow_html=True)

st.title('REIT PUBLIC MARKET TRADING COMPARABLES')
# st.write('*TBU*')

@st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)

def display_ticker_stats(ticker_input):
    display_ticker_df = ticker_output_df.loc[ticker_output_df['ticker'] == ticker_input]
    st.dataframe(display_ticker_df.style.format(col_format_dict).set_table_styles(df_styles))

# def display_ticker_charts(ticker_input):
#     x = all_reits_close.index
#     y = all_reits_close[ticker_input]
#     px.line(x, y,
#         #all_reits_close[{ticker_input}],
#             # x=ticker_output_df['reportPeriod'],
#             # y=ticker_output_df['marketCapitalization'],
#             # color=ticker_output_df['sector'],
#             # # color_continuous_scale=Electric,
#             # color_discrete_sequence=Electric,
#             # color_discrete_map=sector_colors,
#             # hover_name=ticker_output_df['company'],
#             # hover_data=ticker_output_df[['sector', 'reportPeriod']],
#             # title=f'{ticker_input} SHARE PRICE',
#             labels=chart_labels,
#             height=600,
#             width=600,
#             )

def display_sector_comps(df):
    # display_sector_comps_df = pd.DataFrame(apartment_cap_table_T) #f'{sector_hardcode}_stack'
    st.dataframe(df.style.set_table_styles(df_styles))


def display_sector_stats(sector_input1):
    display_sector_df = ticker_output_df.loc[ticker_output_df['sector'] == sector_input1]
    # display_sector_df.drop(columns=display_ticker_df, inplace=True)
    st.dataframe(display_sector_df.style.format(col_format_dict).set_table_styles(df_styles))
    # .applymap(color_negative_red, subset=[''])
    # .highlight_max(subset=[''])
    # .set_caption(f'CUSTOM CAPTION')

def display_sector_charts(sector_input2):
    x = all_reits_close.index
    y = all_reits_close[ticker_input]
    px.line(x, y,
        #all_reits_close[{ticker_input}],
            # x=ticker_output_df['reportPeriod'],
            # y=ticker_output_df['marketCapitalization'],
            # color=ticker_output_df['sector'],
            # # color_continuous_scale=Electric,
            # color_discrete_sequence=Electric,
            # color_discrete_map=sector_colors,
            # hover_name=ticker_output_df['company'],
            # hover_data=ticker_output_df[['sector', 'reportPeriod']],
            # title=f'{ticker_input} SHARE PRICE',
            labels=chart_labels,
            height=1000,
            width=1000,
            )

## SECTOR TABS ##
tab_0, tab_1, tab_2, tab_3, tab_4, tab_5, tab_6, tab_7, tab_8, tab_9, tab_10 = st.tabs(['ALL SECTORS', 'APARTMENT', 'OFFICE', 'HOTEL', 'MALL', 'STRIP CENTER', 'NET LEASE', 'INDUSTRIAL', 'SELF-STORAGE', 'DATA CENTER', 'HEALTHCARE'])
with tab_0:
    st.subheader('ALL SECTORS')

    # st.dataframe(apartment_cap_table_T)
    # st.dataframe(display_sector_comps('apartment'))

    st.plotly_chart(px.line(all_sectors_close_df,
                            # line_group=all_reits_close['sector'],
                            # color=all_reits_close_group.columns,
                            # color_continuous_scale=Electric,
                            color_discrete_sequence=Ice_r,
                            color_discrete_map=sector_colors,
                            title=f'HISTORICAL SHARE PRICE ($)',
                            # symbol='*',
                            labels=chart_labels,
                            range_x=[sidebar_start, sidebar_end],
                            range_y=[0, 500],
                            height=600,
                            width=1200,
                            ))

    st.subheader('ALL REITS')
    # all_sectors_x = all_reits_close.columns,
    # mask = df.continent.isin(continents)
    st.plotly_chart(px.line(all_reits_close_df,
                            # line_group=all_reits_close['sector'],
                            # color=all_reits_close_group.columns,
                            # color_continuous_scale=Electric,
                            color_discrete_sequence=Ice_r,
                            color_discrete_map=sector_colors,
                            title=f'HISTORICAL SHARE PRICE ($)',
                            # symbol='*',
                            labels=chart_labels,
                            range_x=[sidebar_start, sidebar_end],
                            # range_y=[0, 400],
                            height=600,
                            width=1200,
                            ))

with tab_1:
    st.subheader('APARTMENT')
    # current_sector_reits =
    # st.dataframe(display_sector_comps(apartment_cap_table_T))
    # st.dataframe(apartment_stack)

    st.plotly_chart(px.line(apartment_reits_close_df,
                            # ['apartment_avg']
                            # color=apartment_reits_close_df.columns,
                            # color_continuous_scale=Electric,
                            color_discrete_sequence=Ice_r,
                            color_discrete_map=sector_colors,
                            title=f'HISTORICAL SHARE PRICE ($)',
                            # symbol='*',
                            labels=chart_labels,
                            range_x=[sidebar_start, sidebar_end],
                            range_y=[0, 300],
                            height=600,
                            width=1200,
                            ))

     # .style.format(col_format_dict).set_table_styles(df_styles))

    # returns = {}
    # for stock in apartment_reits_close.columns:
    #     returns[stock] = apartment_reits_close[stock].dropna().iloc[sidebar_start] / apartment_reits_close[stock].dropna().iloc[sidebar_end]
    #     st.dataframe(returns)

    with st.form('APARTMENT TICKER METRICS'):
        ticker_prompt = st.subheader('SELECT TICKER:')
        ticker_input = st.selectbox('TICKER', (apartment))
        ticker_submit = st.form_submit_button('TICKER METRICS')
        if ticker_submit:

            display_ticker_stats(ticker_input)
            # display_ticker_charts(ticker_input)
        # display_sector_stats('RESIDENTIAL')


with tab_2:
    st.subheader('OFFICE REITS')

with tab_3:
    st.subheader('HOTEL REITS')

with tab_4:
    st.subheader('MALL REITS')

with tab_5:
    st.subheader('STRIP CENTER REITS')

with tab_6:
    st.subheader('NET LEASE REITS')

with tab_7:
    st.subheader('INDUSTRIAL REITS')

with tab_8:
    st.subheader('SELF-STORAGE REITS')

with tab_9:
    st.subheader('DATA CENTER REITS')

with tab_10:
    st.subheader('HEALTHCARE REITS')



## APP TERMINATION ##
st.stop()



## SELECTION FORMS -- SECTOR / TICKER -- OLD ##
# @st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)

# def display_sector_stats(sector_input):
#     display_sector_df = ticker_output_df.loc[ticker_output_df['sector'] == sector_input]
#     # display_sector_df.drop(columns=display_ticker_df, inplace=True)
#     st.dataframe(display_sector_df.style.format(col_format_dict).set_table_styles(df_styles))
#     # .applymap(color_negative_red, subset=[''])
#     # .highlight_max(subset=[''])
#     # .set_caption(f'CUSTOM CAPTION')
#
#
# with st.form('SECTOR METRICS'):
#     sector_prompt = st.subheader('SELECT SECTOR:')
#     sector_input = st.selectbox('SECTOR', (ticker_output_df['sector'].unique())) #'EXOPLANETS:'
#     sector_submit = st.form_submit_button('SECTOR METRICS')
#     if sector_submit:
#         display_sector_stats(sector_input)

# def display_ticker_stats(ticker_input):
#     display_ticker_df = ticker_output_df.loc[ticker_output_df['ticker'] == ticker_input]
#     st.dataframe(display_ticker_df)
#
# with st.form('TICKER METRICS'):
#     ticker_prompt = st.subheader('SELECT TICKER:')
#     ticker_input = st.selectbox('TICKER', (reit_tickers))
#     ticker_submit = st.form_submit_button('TICKER METRICS')
#     if ticker_submit:
#         display_ticker_stats(ticker_input)




## IMAGES ## -- ## WILLARD SPONSOR? ##
    # tele_col_1, tele_col_2, tele_col_3, tele_col_4 = st.columns(4)
    # tele_col_1.image(jwst_tele_img_1, caption='JAMES WEBB SPACE TELESCOPE (JWST)', width=200)
    # tele_col_2.image(tess_tele_img_1, caption='TRANSITING EXOPLANET SURVEY SATELLITE (TESS)', width=200)
    # tele_col_3.image(kepler_tele_img_1, caption='KEPLER SPACE TELESCOPE', width=200)
    # tele_col_4.image(hubble_tele_img_1, caption='HUBBLE SPACE TELESCOPE', width=200)

# st.plotly_chart(ticker_input_line, use_container_width=False, sharing="streamlit")

## REIT SCATTER MATRIX ##
# st.plotly_chart(reit_scatter_matrix, use_container_width=False, sharing="streamlit")

## SECTOR HEATMAP ##


## MARKET CAP LINE CHART ##
# st.plotly_chart(sector_market_cap_line, use_container_width=False, sharing="streamlit")

## 3D SCATTER ##
# st.plotly_chart(scatter_3d_1, use_container_width=False, sharing="streamlit")

## DISCOVERY INFORMATION ##
# left_col_1, right_col_1 = st.columns(2)
# st.plotly_chart(disc_info_1.update_yaxes(categoryorder='total ascending'), use_container_width=True, sharing="streamlit")

## DENSITY MAP ##
# st.plotly_chart(density_map_1, use_container_width=False, sharing="streamlit")

## SUBPLOTS ##
# subplots = make_subplots(rows=1, cols=2)
    # subplots.add_trace(scatter_3d_1, row=1, col=1)
    # subplots.add_trace(scatter_3d_1, row=1, col=2)

        ## DATAFRAME STYLING ##

        # def df_style_map(val):
        #     if val == 'United States':
        #         color = 'black'
        #     else:
        #         color = 'pink'
        #         return f'background-color: {color}'
        #
        # st.dataframe(buyer_rec_df.style.applymap(df_style_map, subset=['COUNTRY']))

## GALAXY IMAGES ##
# img_col_1, img_col_2, img_col_3 = st.columns(3)
# img_col_1.image(jwst_carina_img_1, caption='CARINA NEBULA (JWST)', width=400)
# img_col_2.image(jwst_phantom_img_1, caption='PHANTOM GALAXY (JWST)', width=400)
# img_col_3.image(jwst_infra_img_1, caption='INFRARED PANORAMIC (JWST)', width=400)



### INTERPRETATION ###



## STREAMLIT COMPONENTS ##


## CONFIG / LAYOUT / PADDING ##

# st.set_page_config(
#     page_title="4M",
#     page_icon="chart_with_upwards_trend",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     menu_items={
#         'Get Help': 'https://www.extremelycoolapp.com/help',
#         'Report a bug': "https://www.extremelycoolapp.com/bug",
#         'About': "# > Creator: Gordon D. Pisciotta  ·  4M  ·  [modern.millennial.market.mapping]",
#     }
# )
# st.markdown(
#     f"""
#     <style>
#     #.reportview-container .main .block-container{{
#         padding-top: {1.3}rem;
#         padding-right: {2.5}rem;
#         padding-left: {3.4}rem;
#         padding-bottom: {3.4}rem;
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True)

## CHART SIZING ##
# col1, col2, col3 = st.columns(3)
# with col1:
#     ticker = st.text_input('TICKER (👇ℹ️)', value='ℹ️')
# with col2:
#     ticker_dx = st.slider('HORIZONTAL OFFSET', min_value=-50, max_value=50, step=1, value=0)  #-30 #30
# with col3:
#     ticker_dy = st.slider('VERTICAL OFFSET', min_value=-50, max_value=50, step=1, value=-10)



### USER AUTHENTICATION ###

# pip install streamlit-authenticator

# import streamlit_authenticator as stauth

# credentials:
#   usernames:
#     jsmith:
#       email: jsmith@gmail.com
#       name: John Smith
#       password: '123' # To be replaced with hashed password
#     rbriggs:
#       email: rbriggs@gmail.com
#       name: Rebecca Briggs
#       password: '456' # To be replaced with hashed password
# cookie:
#   expiry_days: 30
#   key: some_signature_key
#   name: some_cookie_name
# preauthorized:
#   emails:
#   - melsby@gmail.com
#
# hashed_passwords = stauth.Hasher(['123', '456']).generate()
#
# with open('../config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)
#
# authenticator = Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )
#
# name, authentication_status, username = authenticator.login('Login', 'main')
#
# if authentication_status:
#     authenticator.logout('Logout', 'main')
#     st.write(f'Welcome *{name}*')
#     st.title('Some content')
# elif authentication_status == False:
#     st.error('Username/password is incorrect')
# elif authentication_status == None:
#     st.warning('Please enter your username and password')
#
# if st.session_state["authentication_status"]:
#     authenticator.logout('Logout', 'main')
#     st.write(f'Welcome *{st.session_state["name"]}*')
#     st.title('Some content')
# elif st.session_state["authentication_status"] == False:
#     st.error('Username/password is incorrect')
# elif st.session_state["authentication_status"] == None:
#     st.warning('Please enter your username and password')


### SCRATCH NOTES ###

## DATE EXTRACTION ##
# df.index.dt.year
# df.index.dt.month
# df.index.dt.day
# df.index.dt.hour
# df.index.dt.minute


## EXCEL SAVE WORKAROUND ##

# def to_excel(df):
#     output = BytesIO()
#     writer = pd.ExcelWriter(output, engine='xlsxwriter')
#     df.to_excel(writer, sheet_name='Sheet1')
#     writer.save()
#     processed_data = output.getvalue()
#     return processed_data
#
# def get_table_download_link(df):
#     """Generates a link allowing the data in a given panda dataframe to be downloaded
#     in:  dataframe
#     out: href string
#     """
#     val = to_excel(df)
#     b64 = base64.b64encode(val)  # val looks like b'...'
#     return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="download.xlsx">Download excel file</a>' # decode b'abc' => abc
#
# st.markdown(get_table_download_link(df), unsafe_allow_html=True)


## POLYGON DATA ##

# from polygon import RESTClient

# client = RESTClient()
# financials = client.get_ticker_details("NFLX")
# print(financials)

# for (i, n) in enumerate(client.list_ticker_news("INTC", limit=5)):
#     print(i, n)
#     if i == 5:
#         break


# client = RESTClient()
# aggs = client.get_aggs("AAPL", 1, "day", "2022-04-04", "2022-04-04")
# print(aggs)

# TENSORFLOW INDICATORS #

# candles = [...]
# c = tfti.features.close(candles)
# rsi = tfti.rsi(candles=c, window_size=7, method='ema')

# you can also pass multidimensional tensors with (time step, features) where features = open, close to calculate some indicator for both open and close
# result = tfti.indicator(candles, ..params..)


## FONTS ##

# t = st.radio("Toggle to see font change", [True, False])
#
# if t:
#     st.markdown(
#         """
#         <style>
# @font-face {
#   font-family: 'Tangerine';
#   font-style: normal;
#   font-weight: 400;
#   src: url(https://fonts.gstatic.com/s/tangerine/v12/IurY6Y5j_oScZZow4VOxCZZM.woff2) format('woff2');
#   unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD;
# }
#
#     html, body, [class*="css"]  {
#     font-family: 'Tangerine';
#     font-size: 48px;
#     }
#     </style>
#
#     """,
#         unsafe_allow_html=True,
#     )
#
# "# Hello"
#
# """This font will look different, based on your choice of radio button"""

# CONFIG TEMPLATE
    # st.set_page_config(page_title="CSS hacks", page_icon=":smirk:")
    #
    # c1 = st.container()
    # st.markdown("---")
    # c2 = st.container()
    # with c1:
    #     st.markdown("Hello")
    #     st.slider("World", 0, 10, key="1")
    # with c2:
    #     st.markdown("Hello")
    #     st.slider("World", 0, 10, key="2")

# STYLE WITH CSS THROUGH MARKDOWN
    # st.markdown("""
    # <style>
    # div[data-testid="stBlock"] {
    #     padding: 1em 0;
    #     border: thick double #32a1ce;
    # }
    # </style>
    # """, unsafe_allow_html=True)


# STYLE WITH JS THROUGH HTML IFRAME
    # components.html("""
    # <script>
    # const elements = window.parent.document.querySelectorAll('div[data-testid="stBlock"]')
    # console.log(elements)
    # elements[0].style.backgroundColor = 'paleturquoise'
    # elements[1].style.backgroundColor = 'lightgreen'
    # </script>
    # """, height=0, width=0)


# st.markdown("""
#             <style>
#             div[data-testid="stBlock"] {padding: 1em 0; border: thick double #32a1ce; color: blue}
#             </style>
#             """,
#             unsafe_allow_html=True)

# style={'textAlign': 'Center', 'backgroundColor': 'rgb(223,187,133)',
#                                            'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
#                                            'border': '4px solid black', 'font-family': 'Arial'}),

#pattern_shape = "nation", pattern_shape_sequence = [".", "x", "+"]

            # fig = px.bar(df, x="sex", y="total_bill", color="smoker", barmode="group", facet_row="time", facet_col="day",
            #        category_orders={"day": ["Thur", "Fri", "Sat", "Sun"], "time": ["Lunch", "Dinner"]})

            # fig = px.scatter_matrix(df, dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"], color="species")

            # fig = px.parallel_categories(df, color="size", color_continuous_scale=px.colors.sequential.Inferno)

            # fig = px.parallel_coordinates(df, color="species_id", labels={"species_id": "Species",
            #                   "sepal_width": "Sepal Width", "sepal_length": "Sepal Length",
            #                   "petal_width": "Petal Width", "petal_length": "Petal Length", },
            #                     color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)



# ['Intangible Assets', 'Capital Surplus', 'Total Liab',
#        'Total Stockholder Equity', 'Minority Interest', 'Other Current Liab',
#        'Total Assets', 'Common Stock', 'Other Current Assets', 'Other Liab',
#        'Gains Losses Not Affecting Retained Earnings', 'Other Assets', 'Cash',
#        'Total Current Liabilities', 'Short Long Term Debt',
#        'Other Stockholder Equity', 'Property Plant Equipment',
#        'Total Current Assets', 'Long Term Investments', 'Net Tangible Assets',
#        'Net Receivables', 'Long Term Debt', 'Accounts Payable',
#        'Deferred Long Term Asset Charges']

#%%
## BALANCE SHEET ##
# ALL REITS #
# common_so_df_temp = pd.DataFrame()
# assets_df_temp = pd.DataFrame()
# liabilities_df_temp = pd.DataFrame()
# nci_df_temp = pd.DataFrame()
# sh_equity_df_temp = pd.DataFrame()
# other_sh_equity_df_temp = pd.DataFrame()
# lt_debt_df_temp = pd.DataFrame()
# st_lt_debt_df_temp = pd.DataFrame()
# cash_df_temp = pd.DataFrame()
# net_tangible_df_temp = pd.DataFrame()
# # cap_surplus_df_temp = pd.DataFrame()
# # _df_temp = pd.DataFrame()
#
# for j in yf_tickers:
#     common_so_df_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Common Stock', mrq]
#     assets_df_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Total Assets', mrq]
#     liabilities_df_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Total Liab', mrq]
#     # nci_df_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Minority Interest', mrq]
#     sh_equity_df_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Total Stockholder Equity', mrq]
#     # lt_debt_df_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Long Term Debt', mrq]
#     cash_df_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Cash', mrq]
#     net_tangible_df_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Net Tangible Assets', mrq]
#     # _df_temp[f'{j}'] = j.quarterly_balance_sheet.loc['', mrq]
#
# ## CAPITALIZATION TABLE ##
#     # PRICE
#     # Total Equity Mkt Capitalization
#     # Preferred??
#     # Total Debt Capitalization
#     # Equity Cap + Debt Cap = Total Mkt Cap
#     # Enterprise Value == Total Mkt Cap - Cash
#
# # print(bs_dict.keys())
#
#
# all_reits_common_so_df = common_so_df_temp.rename(columns=yf_ticker_dict)
# all_reits_assets_df = assets_df_temp.rename(columns=yf_ticker_dict)
# all_reits_liabilities_df = liabilities_df_temp.rename(columns=yf_ticker_dict)
# # all_reits_nci_df = nci_df_temp.rename(columns=yf_ticker_dict)
# all_reits_sh_equity_df = sh_equity_df_temp.rename(columns=yf_ticker_dict)
# # all_reits_lt_debt_df = lt_debt_df_temp.rename(columns=yf_ticker_dict)
# all_reits_cash_df = cash_df_temp.rename(columns=yf_ticker_dict)
# all_reits_net_tangible_df = net_tangible_df_temp.rename(columns=yf_ticker_dict)
#
# ## COMBINE BY SECTOR ##
# all_reits_cap_table = pd.concat([all_reits_common_so_df, all_reits_assets_df, all_reits_liabilities_df,
#                                  #all_reits_nci_df, #all_reits_lt_debt_df,
#                                  all_reits_sh_equity_df,
#                                  all_reits_cash_df, all_reits_net_tangible_df],
#                                 keys=['S/O', 'TTL ASSETS', 'TTL LIABILITIES', #'NCI',
#                                       'S.H. EQUITY',  'CASH', 'NET TBV']) #'LT DEBT',
#
# all_reits_cap_table_T = all_reits_cap_table.T
#
# #%%
#
# print(all_reits_cap_table_T.head())
# print(all_reits_cap_table_T.info())
# print(all_reits_cap_table_T[-20:])


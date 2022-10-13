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
import requests
from bs4 import BeautifulSoup
import time
import r'https://raw.githubusercontent.com/nehat312/REIT-comps/main/REIT-scrape.py'


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


#%%
## DATA SCRAPE IMPORT ##
# reit_scrape_path = current_path + '/REIT-scrape.py'
# %run reit_scrape_path
# %cd $abspath_util_deep

#%%
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
## REAL ESTATE SECTORS / TICKERS ##
apartment = ["EQR",	"AVB", "ESS", "MAA", "UDR",	"CPT", "AIV",] #, "APTS"  "BRG"
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

reit_tickers = ["EQR", "AVB", "ESS", "MAA", "UDR", "CPT", "AIV", #"BRG", #"APTS",
               "BXP", "VNO", "KRC", "DEI", "JBGS", "CUZ", "HPP", "SLG",	"HIW", "OFC", "PGRE", "PDM", "WRE", "ESRT",	"BDN", "EQC", "VRE",
               "HST", "RHP", "PK", "APLE",	"SHO",	"PEB",	"RLJ", "DRH", "INN", "HT", "AHT", "BHR",
               "SPG", "MAC", "PEI", #"SKT", "SRG", #CBL, #WPG
               "REG", "FRT", "KIM",	"BRX",	"AKR",	"UE", "ROIC", "CDR", "SITC", "BFS",
               "O", "WPC", "NNN", "STOR", "SRC", "PINE", "FCPT", "ADC", "EPRT",
               "PLD", "DRE", "FR", "EGP", #GTY
               "EXR",	"CUBE",	"REXR",	"LSI",
               "EQIX", "DLR", "AMT",
               "WELL", "PEAK", "VTR", "OHI", "HR"]

sector_dict = {'apartment': ["EQR",	"AVB", "ESS", "MAA", "UDR", "CPT",	"AIV",	], #, "APTS" "BRG"
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



#%%

clean_yahoo_index = ['Market Cap (intraday) ', 'Enterprise Value ', 'Shares Outstanding 5', 'Float 8', #'Implied Shares Outstanding 6':'',
                    'Forward P/E ', 'Trailing P/E ', # 'PEG Ratio (5 yr expected) ',
                    'Price/Sales (ttm)', 'Price/Book (mrq)',
                    'Enterprise Value/Revenue ', 'Enterprise Value/EBITDA ',
                    '52 Week High 3', '52 Week Low 3', #'52-Week Change 3':'52-WEEK %',
                    # 'Beta (5Y Monthly) ':'',#'S&P500 52-Week Change 3':'',
                    '50-Day Moving Average 3', '200-Day Moving Average 3',
                    'Avg Vol (3 month) 3', 'Avg Vol (10 day) 3',
                    '% Held by Insiders 1', '% Held by Institutions 1',
                    'Shares Short (Jul 28, 2022) 4', 'Short Ratio (Jul 28, 2022) 4',
                    'Forward Annual Dividend Rate 4', 'Trailing Annual Dividend Rate 3',
                    'Forward Annual Dividend Yield 4', 'Trailing Annual Dividend Yield 3',
                    '5 Year Average Dividend Yield 4',
                    'Payout Ratio 4',
                    #'Dividend Date 3', 'Ex-Dividend Date 4',
                    # 'Last Split Factor 2', 'Last Split Date 3',
                    # #'Fiscal Year Ends ', 'Most Recent Quarter (mrq)',
                    'Profit Margin ', 'Operating Margin (ttm)',
                    'Return on Assets (ttm)', 'Return on Equity (ttm)',
                    'Revenue (ttm)', #'Revenue Per Share (ttm)',
                    'Gross Profit (ttm)', 'EBITDA ',
                    'Quarterly Revenue Growth (yoy)', 'Quarterly Earnings Growth (yoy)',
                    'Total Cash (mrq)', 'Book Value Per Share (mrq)',
                     'Total Debt (mrq)', 'Total Debt/Equity (mrq)',
                    'Operating Cash Flow (ttm)', 'Levered Free Cash Flow (ttm)']

cap_stack = ['Market Cap (intraday) ',

             ]

working_sector_dict = {'Market Cap (intraday) ':'MARKET CAPITALIZATION',
                       'Enterprise Value ':'ENTERPRISE VALUE',
                       'Shares Outstanding 5':'SHARES OUTSTANDING', 'Float 8':'PUBLIC FLOAT', #'Implied Shares Outstanding 6':'',
                       'Forward P/E ':'FORWARD PRICE/EARNINGS', 'Trailing P/E ':'TRAILING PRICE/EARNINGS', # 'PEG Ratio (5 yr expected) ',
                       'Price/Sales (ttm)':'PRICE/SALES RATIO (TTM)', 'Price/Book (mrq)':'PRICE/BOOK (MRQ)',
                       'Enterprise Value/Revenue ':'EV/REVENUE', 'Enterprise Value/EBITDA ':'EV/EBITDA',
                       '52 Week High 3':'52-WEEK HIGH', '52 Week Low 3':'52-WEEK LOW', #'52-Week Change 3':'52-WEEK %',
                       # 'Beta (5Y Monthly) ':'',#'S&P500 52-Week Change 3':'',
                       '50-Day Moving Average 3':'50-MA', '200-Day Moving Average 3':'200-MA',
                       'Avg Vol (3 month) 3':'AVG. VOLUME (3 MONTH)', 'Avg Vol (10 day) 3':'AVG. VOLUME (10 DAY)',
                       '% Held by Insiders 1':'INSIDERS %', '% Held by Institutions 1':'INSTITUTIONAL %',
                       'Shares Short (Jul 28, 2022) 4':'SHARES SHORT', 'Short Ratio (Jul 28, 2022) 4':'SHORT RATIO',
                       #'Short % of Float (Jul 28, 2022) 4':'', 'Short % of Shares Outstanding (Jul 28, 2022) 4':'',
                       #'Shares Short (prior month Jun 29, 2022) 4':'',
                       'Forward Annual Dividend Rate 4':'FORWARD ANN. DIVIDEND/SHARE', 'Trailing Annual Dividend Rate 3':'TRAILING ANN. DIVIDEND/SHARE',
                       'Forward Annual Dividend Yield 4':'FORWARD ANN. DIVIDEND YIELD', 'Trailing Annual Dividend Yield 3':'TRAILING ANN. DIVIDEND YIELD',
                       '5 Year Average Dividend Yield 4':'5-YEAR AVG. DIVIDEND YIELD',
                       'Payout Ratio 4':'DIVIDEND PAYOUT RATIO',
                       #'Dividend Date 3', 'Ex-Dividend Date 4',
                       # 'Last Split Factor 2', 'Last Split Date 3',
                       #'Fiscal Year Ends ', 'Most Recent Quarter (mrq)',
                       'Profit Margin ':'PROFIT MARGIN', 'Operating Margin (ttm)':'OPERATING MARGIN (TTM)',
                       'Return on Assets (ttm)':'RETURN ON ASSETS (TTM)', 'Return on Equity (ttm)':'RETURN ON EQUITY (TTM)',
                       'Revenue (ttm)':'REVENUE (TTM)', #'Revenue Per Share (ttm)',
                       'Gross Profit (ttm)':'GROSS PROFIT (TTM)', 'EBITDA ':'EBITDA',
                       'Quarterly Revenue Growth (yoy)':'QTR. REVENUE GROWTH (YoY)', 'Quarterly Earnings Growth (yoy)':'QTR. EARNINGS GROWTH (YoY)',
                       'Total Cash (mrq)':'CASH (MRQ)', #'Total Cash Per Share (mrq)':'',
                       'Book Value Per Share (mrq)':'BV PER SHARE (MRQ)',

                       'Total Debt (mrq)':'TOTAL DEBT (MRQ)',
                       'Total Debt/Equity (mrq)':'TOTAL DEBT/EQUITY (MRQ)', # 'Current Ratio (mrq)':'',
                       'Operating Cash Flow (ttm)':'OPERATING CF (MRQ)',
                       'Levered Free Cash Flow (ttm)':'LEVERED FCF (TTM)'}

                        #'Net Income Avi to Common (ttm)':'', 'Diluted EPS (ttm)':'',

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


### SCRATCH NOTES ###

## DATE EXTRACTION ##
# df.index.dt.year
# df.index.dt.month
# df.index.dt.day
# df.index.dt.hour
# df.index.dt.minute


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



#%%

## DATAFRAME STYLING ##

# def df_style_map(val):
#     if val == 'United States':
#         color = 'black'
#     else:
#         color = 'pink'
#         return f'background-color: {color}'
#
# st.dataframe(buyer_rec_df.style.applymap(df_style_map, subset=['COUNTRY']))

#%%


#%%
## REIT TRADING COMPARABLES -- DATA SCRAPE ##

## LIBRARY IMPORTS ##
import time

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

#%%
## TIME INTERVALS ##
today = datetime.date.today()
before = today - datetime.timedelta(days=1095) #700
start_date = '2000-01-01'
end_date = today #'2022-06-30'  #'2022-03-31'
mrq = '2022-06-30'
mrq_prior = '2022-03-31'
mry = '2021-12-31'

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

#%%
## YAHOO FINANCE ##
headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'}
base_yahoo_url = 'https://finance.yahoo.com/quote/' #https://finance.yahoo.com/quote/AVB/key-statistics?p=AVB
ext_yahoo_url = 'key-statistics?p='

#%%
##  YAHOOOOOOOO ####
yahoo_cols = ['METRICS', 'CURRENT', '06-30-2022', '03-31-2022', '12-31-2021', '09-30-2021', '06-30-2021']
yahoo_data_dict = {i : pd.DataFrame() for i in apartment} # reit_tickers


#%%
## ONLY NEED CURRENT QTR ##
for ticker in apartment: # reit_tickers
    yahoo_key_stats = requests.get(base_yahoo_url + f'{ticker}/' + ext_yahoo_url + f'{ticker}', headers=headers)
    soup = BeautifulSoup(yahoo_key_stats.text, 'html.parser')   #r.content,'lxml'     #.text,'html.parser'
    section = soup.find_all('section')[0]  # [0] == 
    div = soup.find_all('div')[0]  # [0] ==
    table0 = soup.find_all('table')[0] # [0] == Valuation Measures
    table1 = soup.find_all('table')[1] # [1] == Stock Price History
    table2 = soup.find_all('table')[2] # [2] == Share Statistics
    table3 = soup.find_all('table')[3] # [3] == Dividends & Splits
        # table = soup.find("table", class_="W(100%) Bdcl(c)  M(0) Whs(n) BdEnd Bdc($seperatorColor) D(itb)")
        # table = soup.find('table')
    tables = [table0, table1, table2, table3]
    for x in tables:
        cols = [each.text for each in x.find_all('th')]
        rows = x.find_all('tr')
    for y in section:
        cols = [each.text for each in y.find_all('th')]
        rows = y.find_all('tr')
    for z in div:
        cols = [each.text for each in y.find_all('th')]
        rows = y.find_all('tr')
    # cols0 = [each.text for each in table0.find_all('th')]
    # rows0 = table0.find_all('tr')
    # cols1 = [each.text for each in table1.find_all('th')]
    # rows1 = table1.find_all('tr')
    # cols2 = [each.text for each in table2.find_all('th')]
    # rows2 = table2.find_all('tr')
    # cols3 = [each.text for each in table3.find_all('th')]
    # rows3 = table3.find_all('tr')

        for row in rows:
            data = [each.text for each in row.find_all('td')]
            temp_df = pd.DataFrame([data])
            yahoo_data_dict[ticker] = yahoo_data_dict[ticker].append(temp_df, sort=True).reset_index(drop=True)
            # yahoo_data_dict[ticker] = yahoo_data_dict[ticker].dropna() #.fillna()

#%%
for reit in yahoo_data_dict:
    yahoo_data_dict[reit].columns = yahoo_cols
    yahoo_data_dict[reit].index = yahoo_data_dict[reit]['METRICS']
    # yahoo_data_dict[ticker][ticker] = yahoo_data_dict[ticker]['2021']
    # yahoo_data_dict[reit].drop(columns=['METRICS', '6/30/2022', '3/31/2022', '12/31/2021', '9/30/2021', '6/30/2021'], inplace=True)

    # return yahoo_data_dict[ticker]




#%%
yahoo_data_dict_copy = yahoo_data_dict

#%%
print(yahoo_data_dict_copy['EQR'])

#%%
print(section)

#%%
print(yahoo_data_dict_copy)


#%%
## GROUP BY SECTOR ##
for i in apartment:
    yahoo_apartment_data = pd.concat([])
    print(yahoo_data_dict_copy[i])

#%%

## CAPITALIZATION TABLE ##

    # PRICE
    # Total Equity Mkt Capitalization
    # Preferred??

    # Total Debt Capitalization
    # Equity Cap + Debt Cap = Total Mkt Cap
    # Enterprise Value == Total Mkt Cap - Cash




#%%



#%%
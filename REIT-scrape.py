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

# INITIALIZE DICTIONARY #
yahoo_data_dict = {i : pd.DataFrame() for i in apartment} # reit_tickers

# og_yahoo_cols = ['METRICS', 'CURRENT', '06-30-2022', '03-31-2022', '12-31-2021', '09-30-2021', '06-30-2021']


#%%
## APARTMENT ##
for ticker in apartment: # reit_tickers
    yahoo_key_stats = requests.get(base_yahoo_url + f'{ticker}/' + ext_yahoo_url + f'{ticker}', headers=headers)
    soup = BeautifulSoup(yahoo_key_stats.text, 'html.parser')   #r.content,'lxml'     #.text,'html.parser'
    div0 = soup.find_all('div')[0]  # [0] ==
    for z in div0:
        div0_cols = [each.text for each in z.find_all('th')]
        div0_rows = z.find_all('tr')
        for row in div0_rows:
            div0_data = [each.text for each in row.find_all('td')]
            temp_df = pd.DataFrame([div0_data], columns=['CURRENT METRICS', f'{ticker}'])
            yahoo_data_dict[ticker] = yahoo_data_dict[ticker].append(temp_df, sort=True).reset_index(drop=True)
        yahoo_data_dict[ticker].index = yahoo_data_dict[ticker]['CURRENT METRICS']
        yahoo_data_dict[ticker].drop(columns=['CURRENT METRICS'], inplace=True)



    # table0 = soup.find_all('table')[0] # [0] == Valuation Measures
    # table1 = soup.find_all('table')[1] # [1] == Stock Price History
    # table2 = soup.find_all('table')[2] # [2] == Share Statistics
    # table3 = soup.find_all('table')[3] # [3] == Dividends & Splits
    # tables = [table0, table1, table2, table3]
    # for x in tables:
    #     tables_cols = [each.text for each in x.find_all('th')]
    #     tables_rows = x.find_all('tr')
    #     for row in tables_rows:
    #         tables_data = [each.text for each in row.find_all('td')]
    #         temp_df = pd.DataFrame([tables_data])
    #         yahoo_data_dict[ticker] = yahoo_data_dict[ticker].append(temp_df, sort=True).reset_index(drop=True)
        # for row in tables_rows:
        #     tables_data = [each.text for each in row.find_all('td')]
        #
        #     temp_df = pd.DataFrame([tables_data])
        #     yahoo_data_dict[ticker] = yahoo_data_dict[ticker].append(temp_df, sort=True).reset_index(drop=True)
            # yahoo_data_dict[ticker] = yahoo_data_dict[ticker].dropna() #.fillna()

#%%
yahoo_data_dict_copy = yahoo_data_dict
print(yahoo_data_dict_copy['EQR'][:65])

#%%
## GROUP BY SECTOR ##
yahoo_apartment_data = pd.DataFrame()
for i in apartment:
    # temp_df = pd.DataFrame(yahoo_data_dict[i], columns=['CURRENT METRICS', f'{i}'])
    # yahoo_apartment_data = yahoo_data_dict[i].append(temp_df, sort=True).reset_index(drop=True)
    yahoo_apartment_data[i] = yahoo_data_dict[i]

#%%
print(yahoo_apartment_data.info())
print(yahoo_apartment_data.index)

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
                       'Book Value Per Share (mrq)':'BV/SHARE (MRQ)',

                       'Total Debt (mrq)':'TOTAL DEBT (MRQ)',
                       'Total Debt/Equity (mrq)':'TOTAL DEBT/EQUITY (MRQ)', # 'Current Ratio (mrq)':'',
                       'Operating Cash Flow (ttm)':'OPERATING CF (MRQ)',
                       'Levered Free Cash Flow (ttm)':'LEVERED FCF (TTM)'}

                        #'Net Income Avi to Common (ttm)':'', 'Diluted EPS (ttm)':'',


#%%
yahoo_apartment_data_new = pd.DataFrame(index=clean_yahoo_index, data=yahoo_apartment_data)
# yahoo_apartment_data_new.index = working_sector_dict
#yahoo_apartment_data.index = yahoo_apartment_data.index.map(working_sector_dict)
# print(yahoo_apartment_data[:10])
#%%
print(yahoo_apartment_data_new.info())

print(yahoo_apartment_data_new[:20])

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
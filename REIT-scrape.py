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


#%%
## YAHOO FINANCE ##
headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'}

# soup = bs.BeautifulSoup(r.content,'lxml')

base_yahoo_url = 'https://finance.yahoo.com/quote/' #https://finance.yahoo.com/quote/AVB/key-statistics?p=AVB
ext_yahoo_url = 'key-statistics?p='


#%%
##  YAHOOOOOOOO ####
yahoo_cols = ['METRICS', 'CURRENT', '6/30/2022', '3/31/2022','12/31/2021', '9/30/2021', '6/30/2021']
yahoo_link_dict = {link : pd.DataFrame() for link in reit_tickers}
yahoo_df = pd.DataFrame()


#%%
yahoo_key_stats = requests.get(base_yahoo_url + f'{i}/' + ext_yahoo_url + f'{i}', headers=headers)
soup = BeautifulSoup(yahoo_key_stats.text, 'html.parser') #r.content,'lxml' # .text,'html.parser'
# table = soup.find("table", class_="W(100%) Bdcl(c)  M(0) Whs(n) BdEnd Bdc($seperatorColor) D(itb)")
table = soup.find_all('table')[0]
# table = soup.find('table')
cols = yahoo_cols
# cols = [each.text for each in table.find_all('th')]
rows = table.find_all('tr')

for row in rows:
    data = [each.text for each in row.find_all('td')]
    temp_df = pd.DataFrame([data])
    yahoo_df = yahoo_df.append(temp_df, sort=True).reset_index(drop=True)



#%%

##  WORTH ITERATING OVER ALL TICKERS IN ONE GIANT DATAFRAME?? ##

## ONLY NEED CURRENT QTR ##
for i in reit_tickers:
    yahoo_key_stats = requests.get(base_yahoo_url + f'{i}/' + ext_yahoo_url + f'{i}', headers=headers)
    soup = BeautifulSoup(yahoo_key_stats.text, 'html.parser') #r.content,'lxml' # .text,'html.parser'
    # table = soup.find("table", class_="W(100%) Bdcl(c)  M(0) Whs(n) BdEnd Bdc($seperatorColor) D(itb)")
    table = soup.find_all('table')[0]
    # table = soup.find('table')
    cols = yahoo_cols
    # cols = [each.text for each in table.find_all('th')]
    rows = table.find_all('tr')

    for row in rows:
        data = [each.text for each in row.find_all('td')]
        temp_df = pd.DataFrame([data])
        yahoo_df = yahoo_df.append(temp_df, sort=True).reset_index(drop=True)

#https://stackoverflow.com/questions/71207430/scraping-from-yahoo-finance-with-beautifulsoup-resulting-in-status-code-404

#%%
# print(soup)
# print(table)

print(yahoo_df)
print(yahoo_df.info())


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
## APARTMENT ##
apartment_common_so_temp = pd.DataFrame()
apartment_assets_temp = pd.DataFrame()
apartment_liabilities_temp = pd.DataFrame()
apartment_nci_temp = pd.DataFrame()
apartment_sh_equity_temp = pd.DataFrame()
apartment_other_sh_equity_temp = pd.DataFrame()
apartment_lt_debt_temp = pd.DataFrame()
apartment_st_lt_debt_temp = pd.DataFrame()
apartment_cash_temp = pd.DataFrame()
apartment_net_tangible_temp = pd.DataFrame()
# apartment_cap_surplus_temp = pd.DataFrame()
# _df_temp = pd.DataFrame()

for j in yf_apartment:
    apartment_common_so_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Common Stock', mrq]
    apartment_assets_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Total Assets', mrq]
    apartment_liabilities_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Total Liab', mrq]
    apartment_nci_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Minority Interest', mrq]
    apartment_sh_equity_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Total Stockholder Equity', mrq]
        # apartment_other_sh_equity_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Other Stockholder Equity', mrq]
    apartment_lt_debt_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Long Term Debt', mrq]
        # apartment_st_lt_debt_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Short Long Term Debt', mrq]
        # apartment_ttl_debt_temp = j.quarterly_balance_sheet.loc['Long Term Debt', mrq] + j.quarterly_balance_sheet.loc['Short Long Term Debt', mrq]
        # cap_surplus_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Capital Surplus', mrq]
    apartment_cash_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Cash', mrq]
    apartment_net_tangible_temp[f'{j}'] = j.quarterly_balance_sheet.loc['Net Tangible Assets', mrq]

    # _df_temp[f'{j}'] = j.quarterly_balance_sheet.loc['', mrq]

#%%

apartment_common_so_df = apartment_common_so_temp.rename(columns=yf_ticker_dict)
apartment_assets_df = apartment_assets_temp.rename(columns=yf_ticker_dict)
apartment_liabilities_df = apartment_liabilities_temp.rename(columns=yf_ticker_dict)
apartment_nci_df = apartment_nci_temp.rename(columns=yf_ticker_dict)
apartment_sh_equity_df = apartment_sh_equity_temp.rename(columns=yf_ticker_dict)
# apartment_other_sh_equity_df = apartment_other_sh_equity_temp.rename(columns=yf_ticker_dict)
apartment_lt_debt_df = apartment_lt_debt_temp.rename(columns=yf_ticker_dict)
# apartment_st_lt_debt_df = apartment_st_lt_debt_temp.rename(columns=yf_ticker_dict)
apartment_cash_df = apartment_cash_temp.rename(columns=yf_ticker_dict)
apartment_net_tangible_df = apartment_net_tangible_temp.rename(columns=yf_ticker_dict)

## COMBINE BY SECTOR ##
apartment_cap_table = pd.concat([apartment_common_so_df, apartment_assets_df, apartment_liabilities_df,
                                     apartment_nci_df, apartment_sh_equity_df, apartment_lt_debt_df, #apartment_st_lt_debt_df, #apartment_other_sh_equity_df,
                                     apartment_cash_df, apartment_net_tangible_df],
                                    keys=['SHARES', 'TTL_ASSETS', 'TTL_LIABILITIES', 'NCI',
                                          'SH_EQUITY', 'LT_DEBT', 'CASH', 'NET_TBV'])



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



apartment_cap_table_T = apartment_cap_table.T
# apartment_cap_table_T.rename(columns=['SHARES1', 'SHARES2', 'SHARES3', 'SHARES4',
#                                       'SHARES5', 'SHARES6', 'SHARES7', 'SHARES8'])

apartment_cap_table_T.columns = apartment_cap_table_T.columns.droplevel(1)
# sector_cap_tables={}
# apt_dict = apartment_cap_table_T.to_dict('sector_cap_tables')

# sector_cap_tables['apt'] = {'apartment':apartment_cap_table_T}


#%%
# print(sector_cap_tables[apartment])
# print(apt_dict)

print(apartment_cap_table_T.info())
# # print(apartment_cap_table_T.head())
# print(apartment_cap_table_T)



#%%



#%%
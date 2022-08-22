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
before = today - datetime.timedelta(days=700)
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
apartment = ["EQR",	"AVB", "ESS", "MAA", "UDR",	"CPT", "AIV", "BRG", "APTS"]
office = ["BXP", "VNO",	"KRC", "DEI", "JBGS", "CUZ", "HPP",	"SLG",	"HIW", "OFC", "PGRE", "PDM", "WRE",	"ESRT",	"BDN", "EQC", "VRE"] #"CLI"
hotel = ["HST",	"RHP",	"PK", "APLE", "SHO", "PEB", "RLJ", "DRH", "INN", "HT", "AHT", "BHR"]    #"XHR",
mall = ["SPG", "MAC", "PEI"] #"CBL" "TCO" "WPG"
strip_center = ["REG", "FRT",	"KIM",	"BRX", "AKR", "UE", "ROIC", "CDR", "SITC", "BFS"]   #"WRI", "RPAI",
net_lease = ["O", "WPC", "NNN",	"STOR",	"SRC", "PINE", "FCPT", "ADC", "EPRT"]  # "VER",
industrial = ["PLD", "DRE",	"FR", "EGP"]
self_storage = ["EXR",	"CUBE",	"REXR",	"LSI"]
data_center = ["EQIX", "DLR" "AMT"] #"CONE", "COR"
healthcare = ["WELL", "PEAK", "VTR", "OHI", "HR"]   #"HTA",

sector_list_of_lists = [apartment, office, hotel, mall, strip_center, net_lease, industrial, self_storage, data_center, healthcare]
sector_list_of_names = ['apartment', 'office', 'hotel', 'mall', 'strip_center', 'net_lease', 'industrial', 'self_storage', 'data_center', 'healthcare']

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

sector_dict = {'apartment': ["EQR",	"AVB", "ESS", "MAA", "UDR", "CPT",	"AIV",	"BRG", "APTS"],
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

#%%
## VARIABLE ASSIGNMENT ##
all_reits_close = all_reits_trading.Close
all_reits_open = all_reits_trading.Open
all_reits_volume = all_reits_trading.Volume

office_reits_close = office_reits_trading.Close
office_reits_open = office_reits_trading.Open
office_reits_volume = office_reits_trading.Volume

apartment_reits_close = apartment_reits_trading.Close
apartment_reits_open = apartment_reits_trading.Open
apartment_reits_volume = apartment_reits_trading.Volume

ticker_list = all_reits_close.columns

## DETERMINE START / END DATES ##
# print(f'START DATE: {all_reits_close.index.min()}')
# print('*'*50)
# print(f'END DATE: {all_reits_close.index.max()}')

## PRE-PROCESSING ##
## FILTER DATA ##
# disc_facility_filter = exoplanets[exoplanets['facility_count'] > 1]
# facility_filtered = disc_facility_filter['disc_facility'].unique()


#%%
## EXPORT HISTORICAL TRADING DATA ##
# all_reits_close.to_csv(basic_path + '/data/reit_trading_test1.csv')
# all_reits_close.to_excel(current_path + f'/data/reit_trading_2000_{today}.xlsx', index=True, header=[0]) #, index = False


## IMPORT DATA (UNIQUE DATAFRAMES FOR EACH CRE SECTOR??) ##
# all_sectors_import = pd.read_excel(current_path + '/data/reit_trading_2000_2022.xlsx', sheet_name='ALL SECTORS', parse_dates = True, index_col = [0], header=[3])

## SAVE COPIES OF IMPORTS ##
# sector_comps = all_sectors_import
# office_comps = office_import
# residential_comps = residential_import
# lodging_comps = lodging_import
# net_lease_comps = net_lease_import
# strip_center_comps = strip_center_import
# mall_comps = mall_import
# healthcare_comps = healthcare_import
# industrial_comps = industrial_import
# self_storage_comps = self_storage_import
# data_center_comps = data_center_import

# sector_df_list = [office_comps, residential_comps,  lodging_comps, net_lease_comps, strip_center_comps,
#                   mall_comps, healthcare_comps, industrial_comps, self_storage_comps, data_center_comps]

#%%
## MAP SECTORS (??) ##
# sector_map_df = all_reits_close
# sector_map_df['sector'] = pd.DataFrame.from_dict(sector_dict)
# sector_map_df['sector'] = sector_map_df['sector'].map(sector_dict)
# print(sector_map_df)

## TOOLBOX FUNCTIONS ##

#%%
##################
# FORMAT / STYLE #
##################

## COLOR SCALES ##
YlOrRd = px.colors.sequential.YlOrRd
Mint = px.colors.sequential.Mint
Electric = px.colors.sequential.Electric
Sunsetdark = px.colors.sequential.Sunsetdark
Sunset = px.colors.sequential.Sunset
Tropic = px.colors.diverging.Tropic
Temps = px.colors.diverging.Temps
Tealrose = px.colors.diverging.Tealrose
Blackbody = px.colors.sequential.Blackbody
Ice = px.colors.sequential.ice
Ice_r = px.colors.sequential.ice_r
Dense = px.colors.sequential.dense

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
                 'healthcare':'#FF3363'
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
## GROUPBY SECTOR ##
office_financials_group = ticker_output_df[ticker_output_df['sector'] == 'OFFICE']
sector_mkt_cap_group = ticker_output_df.groupby(['sector', 'reportPeriod'], as_index=False)['marketCapitalization'].sum()
sector_multiples_group = ticker_output_df.groupby(['sector', 'reportPeriod'], as_index=False)['enterpriseValueOverEBIT', 'enterpriseValueOverEBITDA'].sum()
sector_ratios_group = ticker_output_df.groupby(['sector', 'reportPeriod'], as_index=False)['profitMargin', 'payoutRatio', 'priceToEarningsRatio'].mean()

# print(sector_mkt_cap)
# print(sector_multiples[:30])
# print(sector_ratios_group)


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
                                 # color_continuous_scale=Electric,
                                 color_discrete_sequence=Electric,
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
#                                    x=ticker_output_df['ra'],
#                                    y=ticker_output_df['dec'],
#                                    z=ticker_output_df['sy_distance_pc'],
#                                    color=ticker_output_df['disc_method'],
#                                    color_discrete_sequence=Temps,
#                                    hover_name=ticker_output_df['company'],
#                                    hover_data=ticker_output_df[['ticker', 'sector']],
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
st.set_page_config(page_title='REIT PUBLIC MARKET TRADING COMPARABLES', layout='wide', initial_sidebar_state='auto') #, page_icon=":smirk:"

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

## WILLARD SPONSOR? ##

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

def display_ticker_charts(ticker_input):
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

def display_sector_stats(sector_input):
    display_sector_df = ticker_output_df.loc[ticker_output_df['sector'] == sector_input]
    # display_sector_df.drop(columns=display_ticker_df, inplace=True)
    st.dataframe(display_sector_df.style.format(col_format_dict).set_table_styles(df_styles))
    # .applymap(color_negative_red, subset=[''])
    # .highlight_max(subset=[''])
    # .set_caption(f'CUSTOM CAPTION')

def display_sector_charts(sector_input):
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
tab_1, tab_2, tab_3, tab_4, tab_5, tab_6, tab_7, tab_8, tab_9, tab_10 = st.tabs(['APARTMENT', 'OFFICE', 'HOTEL', 'MALL', 'STRIP CENTER', 'NET LEASE', 'INDUSTRIAL', 'SELF-STORAGE', 'DATA CENTER', 'HEALTHCARE'])
with tab_1:
    st.header('APARTMENT')
    x = apartment_reits_close.index
    # y = apartment_reits_close[ticker_input]
    st.plotly_chart(px.line(apartment_reits_close,
                            # color=ticker_output_df.columns,
                            # color_continuous_scale=Electric,
                            color_discrete_sequence=Tropic,
                            color_discrete_map=sector_colors,
                            # hover_name=ticker_output_df['company'],
                            # hover_data=ticker_output_df[['sector', 'reportPeriod']],
                            title=f'HISTORICAL SHARE PRICE ($)',
                            labels=chart_labels,
                            height=800,
                            width=800,
                            ))

    with st.form('APARTMENT TICKER METRICS'):
        ticker_prompt = st.subheader('SELECT TICKER:')
        ticker_input = st.selectbox('TICKER', (apartment))
        ticker_submit = st.form_submit_button('TICKER METRICS')
        if ticker_submit:
            # display_ticker_stats(ticker_input)
            date_x = all_reits_close.index
            price_y = all_reits_close[ticker_input]
            st.plotly_chart(px.line(x=date_x, y=price_y,
                    # all_reits_close[{ticker_input}],
                    # x=ticker_output_df['reportPeriod'],
                    # y=ticker_output_df['marketCapitalization'],
                    # color=ticker_output_df['sector'],
                    # # color_continuous_scale=Electric,
                    # color_discrete_sequence=Electric,
                    # color_discrete_map=sector_colors,
                    # hover_name=ticker_output_df['company'],
                    # hover_data=ticker_output_df[['sector', 'reportPeriod']],
                    title=f'{ticker_input} SHARE PRICE',
                    labels=chart_labels,
                                    # range_x=TBU,
                                    # range_y=TBU,
                    height=600,
                    width=600,
                    ))
            # display_ticker_charts(ticker_input)
        # display_sector_stats('RESIDENTIAL')


with tab_2:
    st.header('OFFICE')

with tab_3:
    st.header('HOTEL')

with tab_4:
    st.header('MALL')

with tab_5:
    st.header('STRIP CENTER')

with tab_6:
    st.header('NET LEASE')

with tab_7:
    st.header('INDUSTRIAL')

with tab_8:
    st.header('SELF-STORAGE')

with tab_9:
    st.header('DATA CENTER')

with tab_10:
    st.header('HEALTHCARE')

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




## SPONSOR IMAGES ##
    # tele_col_1, tele_col_2, tele_col_3, tele_col_4 = st.columns(4)
    # tele_col_1.image(jwst_tele_img_1, caption='JAMES WEBB SPACE TELESCOPE (JWST)', width=200)
    # tele_col_2.image(tess_tele_img_1, caption='TRANSITING EXOPLANET SURVEY SATELLITE (TESS)', width=200)
    # tele_col_3.image(kepler_tele_img_1, caption='KEPLER SPACE TELESCOPE', width=200)
    # tele_col_4.image(hubble_tele_img_1, caption='HUBBLE SPACE TELESCOPE', width=200)

# st.plotly_chart(ticker_input_line, use_container_width=False, sharing="streamlit")

## REIT SCATTER MATRIX ##
st.plotly_chart(reit_scatter_matrix, use_container_width=False, sharing="streamlit")

## HEATMAP ##

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


## SCRIPT TERMINATION ##
st.stop()




### INTERPRETATION ###



## STREAMLIT COMPONENTS ##

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



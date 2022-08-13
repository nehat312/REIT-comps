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

## DIRECTORY CONFIGURATION ##
abs_path = r'https://raw.githubusercontent.com/nehat312/exoplanet-explorer/main'
exoplanet_path = abs_path + '/data/NASA_Exoplanets-8-7-22.csv'

## DATA IMPORT ##
exoplanets = pd.read_csv(exoplanet_path, header=0, index_col='loc_rowid') #, header=0, index_col='pl_name'#,
exoplanets.sort_values(by='disc_year', inplace=True)

## IMAGE IMPORT ##
jwst_tele_img_1 = Image.open('images/JWST-2.jpg')
tess_tele_img_1 = Image.open('images/TESS-1.jpg')
kepler_tele_img_1 = Image.open('images/Kepler-1.jpg')
hubble_tele_img_1 = Image.open('images/Hubble-1.jpg')
jwst_carina_img_1 = Image.open('images/JW-Carina-1.jpg')
jwst_phantom_img_1 = Image.open('images/JW-Phantom.jpg')
jwst_infra_img_1 = Image.open('images/JW-Infrared.jpg')


# exoplanets.dropna(inplace=True)

# print(exoplanets.info())
# print(exoplanets.columns)
# print(exoplanets.head())

# pd.to_numeric(exoplanets['disc_year'])
# exoplanets['disc_year'].astype(int)

## FORMAT / STYLE ##

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

# pd.options.display.float_format = '${:,.2f}'.format
# pd.set_option('display.max_colwidth', 200)

## VISUALIATION LABELS ##

chart_labels = {'pl_name':'PL. NAME',
                'host_name':'ST. NAME',
                'sy_star_count':'SYS. STARS (#)',
                'sy_planet_count':'SYS. PLANETS (#)',
                'disc_method':'DISC. METHOD',
                'disc_year':'DISC. YEAR',
                'disc_facility':'DISC. FACILITY',
                'disc_telescope':'DISC. TELESCOPE',
                'disc_instrument':'DISC. INSTRUMENT',
                'pl_orbper':'ORB. PERIOD',
                'pl_orbeccen':'ORB. ECCENTRICITY',
                'pl_orbsmax':'ORB. SPEED',
                'pl_rade':'PL. RADIUS (E)',
                'pl_radj':'PL. RADIUS (J)',
                'pl_bmasse':'PL. MASS (E)',
                'pl_bmassj':'PL. MASS (J)',
                'st_temp_eff_k':'ST. TEMP. (K)',
                'st_radius':'ST. RADIUS',
                'st_mass':'ST. MASS',
                'st_metallicity':'METALLICITY',
                'st_surf_gravity':'SURFACE GRAVITY',
                'sy_distance_pc':'ST. DISTANCE (PC)',
                'ra':'RIGHT ASCENSION',
                'dec':'DECLINATION',
                'glon':'GALACTIC LONGITUDE',
                'glat':'GALACTIC LATITUDE',
                }

## FEATURED VARIABLES ##

exo_planet_list = list(exoplanets['pl_name'])
exo_star_list = list(exoplanets['host_name'])
disc_telescope_list = list(exoplanets['disc_telescope'])
disc_method_list = list(exoplanets['disc_method'])
disc_facility_list = list(exoplanets['disc_facility'])
disc_year_list = list(exoplanets['disc_year'])

## PRE-PROCESSING ##
exo_drop_na = exoplanets.dropna()
exo_with_temp = exoplanets[['st_temp_eff_k']].dropna()
exo_with_dist = exoplanets[['sy_distance_pc']].dropna()


## FILTER DATA ##
disc_facility_filter = exoplanets[exoplanets['facility_count'] > 1]
facility_filtered = disc_facility_filter['disc_facility'].unique()
# print(disc_facility_filter)
# print(facility_filtered)


## VISUALIZATIONS ##

scatter_3d_1 = px.scatter_3d(exo_drop_na,
                             x=exo_drop_na['ra'],
                             y=exo_drop_na['dec'],
                             z=exo_drop_na['sy_distance_pc'],
                             color=exo_drop_na['st_temp_eff_k'],
                             color_discrete_sequence=Ice_r,
                             color_continuous_scale=Ice_r,
                             color_continuous_midpoint=5000,
                             size=exo_drop_na['pl_rade'],
                             size_max=50,
                             # symbol=exo_drop_na['disc_year'],
                             hover_name=exo_drop_na['pl_name'],
                             hover_data=exo_drop_na[['host_name', 'disc_facility', 'disc_telescope']],
                             title='EXOPLANET POPULATION -- RIGHT ASCENSION / DECLINATION / DISTANCE',
                             labels=chart_labels,
                             # range_x=[0,360],
                             # range_y=[-50,50],
                             range_z=[0,2500],
                             # range_color=Sunsetdark,
                             opacity=.8,
                             height=800,
                             width=1600,
                             )

disc_info_1 = px.histogram(disc_facility_filter,
                           y=disc_facility_filter['disc_facility'],
                           color=disc_facility_filter['disc_method'],
                           color_discrete_sequence=Ice_r,
                           hover_name=disc_facility_filter['pl_name'],
                           hover_data=disc_facility_filter[['host_name', 'disc_facility', 'disc_telescope', 'sy_star_count', 'sy_planet_count']],
                           # animation_frame=disc_facility_filter['disc_year'],
                           # animation_group=disc_facility_filter['disc_facility'],
                           title='EXOPLANET DISCOVERY FACILITY (BY DISCOVERY METHOD)',
                           labels=chart_labels,
                           range_x=[0,2500],
                           height=1000,
                           # width=800,
                           )

density_map_1 = px.density_contour(exoplanets,
                                   x=exoplanets['ra'],
                                   y=exoplanets['dec'],
                                   z=exoplanets['sy_distance_pc'],
                                   color=exoplanets['disc_method'],
                                   color_discrete_sequence=Temps,
                                   hover_name=exoplanets['pl_name'],
                                   hover_data=exoplanets[['host_name', 'disc_facility', 'disc_telescope', 'sy_star_count', 'sy_planet_count']],
                                   title='EXOPLANET RIGHT ASCENSION / DECLINATION',
                                   labels=chart_labels,
                                   )

exo_matrix_1 = px.scatter_matrix(exoplanets,
                                     dimensions=['pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbeccen'], #, 'pl_orbsmax'
                                     color=exoplanets['st_temp_eff_k'],
                                     color_continuous_scale=Ice_r,
                                     color_discrete_sequence=Ice_r,
                                     hover_name=exoplanets['pl_name'],
                                     hover_data=exoplanets[['host_name', 'sy_star_count', 'sy_planet_count']],
                                     title='EXOPLANET ATTRIBUTES',
                                     labels=chart_labels,
                                 height=850,
                                 # width=800,
                                 )


#####################
### STREAMLIT APP ###
#####################

## CONFIGURATION ##
st.set_page_config(page_title='EXOPLANET EXPLORER', layout='wide', initial_sidebar_state='auto') #, page_icon=":smirk:"

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """

st.markdown(hide_menu_style, unsafe_allow_html=True)


## SIDEBAR ##
# st.sidebar.xyz


## HEADER ##
st.container()

## EXTERNAL LINKS ##

github_link = '[GITHUB REPOSITORY](https://github.com/nehat312/exoplanet-explorer/)'
nasa_exo_link = '[NASA EXOPLANETS](https://exoplanets.nasa.gov/)'
nasa_caltech_link = '[NASA ARCHIVE](https://exoplanetarchive.ipac.caltech.edu/)'

link_col_1, link_col_2, link_col_3 = st.columns(3)
ext_link_1 = link_col_1.markdown(github_link, unsafe_allow_html=True)
ext_link_2 = link_col_2.markdown(nasa_exo_link, unsafe_allow_html=True)
ext_link_3 = link_col_3.markdown(nasa_caltech_link, unsafe_allow_html=True)

st.title('EXOPLANET EXPLORER')
st.write('*Sourced from NASA-CalTECH mission archives*')

## TELESCOPE IMAGES ##
tele_col_1, tele_col_2, tele_col_3, tele_col_4 = st.columns(4)
tele_col_1.image(jwst_tele_img_1, caption='JAMES WEBB SPACE TELESCOPE (JWST)', width=250)
tele_col_2.image(tess_tele_img_1, caption='TRANSITING EXOPLANET SURVEY SATELLITE (TESS)', width=250)
tele_col_3.image(kepler_tele_img_1, caption='KEPLER SPACE TELESCOPE', width=250)
tele_col_4.image(hubble_tele_img_1, caption='HUBBLE SPACE TELESCOPE', width=250)

## 3D SCATTER ##
st.plotly_chart(scatter_3d_1, use_container_width=False, sharing="streamlit")

## SELECTION FORM ##
exo_drop_cols = ['pl_controv_flag', 'pl_bmassprov', 'ttv_flag',
                 'st_temp_eff_k1', 'st_temp_eff_k2',
                 'decstr', 'rastr',
                 'sy_vmag', 'sy_kmag', 'sy_gaiamag']


## EXOPLANET SELECTION ##
@st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)
def display_planet_stats(exo_input):
    exo_df = exoplanets.loc[exoplanets['pl_name'] == exo_input] #'K2-398 b'
    exo_df.drop(columns=exo_drop_cols, inplace=True)
    st.dataframe(exo_df)

with st.form('EXOPLANET SELECTION'):
    exoplanet_prompt = st.subheader('SELECT AN EXOPLANET:')
    exo_input = st.selectbox('', (exo_planet_list)) #'EXOPLANETS:'
    exo_submit = st.form_submit_button('EXO-STATS')
    if exo_submit:
        display_planet_stats(exo_input)


## DISCOVERY INFORMATION ##
st.plotly_chart(disc_info_1.update_yaxes(categoryorder='total ascending'), use_container_width=True, sharing="streamlit")

## SCATTER MATRIX ##
left_col_1, right_col_1 = st.columns(2)
left_col_1.plotly_chart(exo_matrix_1, use_container_width=False, sharing="streamlit")
right_col_1.plotly_chart(star_matrix_1, use_container_width=False, sharing="streamlit")

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
img_col_1, img_col_2, img_col_3 = st.columns(3)
img_col_1.image(jwst_carina_img_1, caption='CARINA NEBULA (JWST)', width=400)
img_col_2.image(jwst_phantom_img_1, caption='PHANTOM GALAXY (JWST)', width=400)
img_col_3.image(jwst_infra_img_1, caption='INFRARED PANORAMIC (JWST)', width=400)


## SCRIPT TERMINATION ##
st.stop()




### INTERPRETATION ###





### SCRATCH NOTES ###






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

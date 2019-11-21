#!/usr/bin/env python
# coding: utf-8

# This code is for the Data Analytics Project by team WeRAnalysers.
# It is the final code after cleaning.The NA value replacement has already been done and it's code can be found on our github.It is not being included in the final
# code as it takes almost 1hr to execute. The dataset after the preprocessing(NA replacement as well as dropping of negative values) has been uploaded and is used i.e.**india_noNA.csv**. 
# This dataset has been created from the following three initial datasets: 2016_india.csv,2017_india.csv and 2018_india.csv 
# This code contains all the major visualizations as well as the SARIMA model.
# For the initial visualisations please refer the Stocktaking.ipynb file on github.
# Other datasets on kaggle have also been used.
# Link : https://www.kaggle.com/rstogi896/rainfall-in-india#Sub_Division_IMD_2017.csv
# 
# The AQI standards followed are those implemented by http://www.indiaenvironmentportal.org.in/files/file/Air%20Quality%20Index.pdf
# 

# In[1]:


#Importing libraries
import numpy as np
import pandas as pd 
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from ipywidgets import widgets
import seaborn as sns
import folium
from folium.plugins import TimestampedGeoJson
from datetime import date
import calendar
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


# In[2]:


#Reading files
df=pd.read_csv("/kaggle/input/air-pollution-dataset-india20162018/india_noNA.csv")
df_rainfall = pd.read_csv("../input/rainfall-in-india/Sub_Division_IMD_2017.csv")
df_temp = pd.read_csv("/kaggle/input/montly-temperature-india-19012017/Mean_Temp_IMD_2017.csv")


# In[3]:


#Check for NA values even after the cleaning process, if any drop them
print("NA values even after cleaning :",df.latitude.isna().sum())
df=df.dropna(axis=0)
df=df.reset_index(drop=True)


# **--Simple Pie chart **

# In[4]:


values=[]
par=df.parameter.value_counts()
for i in range(len(par)):
    values.append(par[i])
labels=['no2','co','pm25','o3','so2','pm10']
colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen','red','black']
fig_pie = go.Figure(data=[go.Pie(labels=labels, 
                             values=values)])
fig_pie.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig_pie.show()


# **--Base Graph/Dropdown to be used to identify patterns to be further investigated--**

# In[5]:


#---MAIN DROPDOWN
#Creating a new dataframe where I have removed the time part of utc because we will be working with daily data and not hourly
data_dropdown=df

#Dropping unwanted attributes and stripping the time from the datetime format
data_dropdown=data_dropdown.drop(['local','country','attribution','location'],axis=1)
data_dropdown['utc'] = data_dropdown['utc'].map(lambda x: str(x)[:-14])
#Converting to datetime
data_dropdown['utc']=pd.to_datetime(data_dropdown['utc'],format="%Y-%m-%d")

#Creating an aggregated dataframe
agg_data_dropdown=data_dropdown
#Grouping by the 3 columns to get unique values
grouped_dropdown=agg_data_dropdown.groupby(['utc','city','parameter'])
#Getting mean of each column
grouped_dropdown=grouped_dropdown.mean()


# In[6]:


#So basically now what we get is a heirarchically indexed dataframe, so the three columns together form the index and are no more
#available as separate columns
grouped_dropdown


# In[7]:


#Inserting those columns from the index again so that we can make accessing easier
grouped_dropdown.insert(3,'date',pd.to_datetime('2016-01-03'))
grouped_dropdown.insert(4,'parameter',0)
grouped_dropdown.insert(5,'city',0)


# In[8]:


#Copying the values from index to the newly added row
for i in grouped_dropdown.index:
    grouped_dropdown.date[i]=i[0]
    grouped_dropdown.parameter[i]=i[2]
    grouped_dropdown.city[i]=i[1]
    #print(grouped_dropdown.date[i])


# In[9]:


#The grouped dropdown
grouped_dropdown.head()


# In[10]:


#Initial content for dropdown i.e city=Delhi and pollutant = 'pm25'
filter_list = [i and j for i, j in
        zip(grouped_dropdown['city'] == 'Delhi', grouped_dropdown['parameter'] == 'pm25')]
#Creating a temporary dataframe holding only the filtered rows
temp_df = grouped_dropdown[filter_list]


# In[11]:


#Making the dropdown as an Interactive Widget
city = widgets.Dropdown(
    description='City:   ',
    value='Delhi',
    options=grouped_dropdown['city'].unique().tolist()
)
parameter = widgets.Dropdown(
    options=list(grouped_dropdown['parameter'].unique()),
    value='pm25',
    description='Parameter:   ',
)
trace1 = go.Scatter(x=temp_df['date'], y=temp_df['value'], mode='markers')

g = go.FigureWidget(data=[trace1],
                    layout=go.Layout(
                        title=dict(
                            text='Pollutant concentrations for different cities'
                        )
                    ))

#Function to keep track of the inputs of the user
def response(change):
        filter_list = [i and j for i, j in
        zip(grouped_dropdown['city'] == city.value, grouped_dropdown['parameter'] == parameter.value)]
        temp_df = grouped_dropdown[filter_list]
        x1 = temp_df['date']
        y1 = temp_df['value']
        with g.batch_update():
            g.data[0].x = x1
            g.data[0].y = y1
            g.layout.xaxis.title = 'Date'
            g.layout.yaxis.title = 'Pollutant levels'


city.observe(response, names="value")
parameter.observe(response, names="value")
container2 = widgets.HBox([city, parameter])
widgets.VBox([container2,g])


# * A general trend that can be observed in most of the cities is that the levels of pm10 during the monsoon months of July-October are considerably low as compared to the neighboring seasons. This is due to wet deposition and air scrubbing by rainfall.
# * But another striking feature is that we observe that in Delhi the plot of pm10 shows a U shape with the peaks occuring between November - January. This trend can be observed for pm25 values for Kolkata as well. We will try to understand this trend better as we move forward.
# * Bengaluru seems to have more or less a stable pollutant concentration. There appears to be no trend in general. This seems to strengthen the fact that "Namma bengaluru Garden city".<br>
# Finer details will be investigated as we move through the study.

#------------------MAP----------------
#Define a dictionary consisting of the pollutant and its categories of severity as specified by the  http://www.indiaenvironmentportal.org.in/files/file/Air%20Quality%20Index.pdf
pollutants = {
    'so2': {
        'notation' : 'SO2',
        'name' :'Sulphur dioxide',
        'bin_edges' : np.array([15,30,45,60,80,100,125,165,250])
    },
    'pm10': {
        'notation' : 'PM10',
        'name' :'Particulate matter < 10 µm',
        'bin_edges' : np.array([10,20,30,40,50,70,100,150,200])
    },
    'o3': {'notation' : 'O3',
        'name' :'Ozone',
        'bin_edges' : np.array([30,50,70,90,110,145,180,240,360])
    },
    'no2': {'notation' : 'NO2',
        'name' :'Nitrogen dioxide',
        'bin_edges' : np.array([25,45,60,80,110,150,200,270,400])
    },
    'co': {'notation' : 'CO',
        'name' :'Carbon monoxide',
         'bin_edges' : np.array([1.4,2.1,2.8,3.6,4.5,5.2,6.6,8.4,13.7])
    },
    'pm25': {
        'notation' : 'PM25',
        'name' :'Particulate matter < 25 µm',
        'bin_edges' : np.array([10,20,30,40,50,70,100,150,200])
    }
    
}


# In[14]:


#Defining volor scale for the map
color_scale = np.array(['#10ff00','#99ff00','#ccff00','#ffff00','#ffee00','#FFCC00','#ff9900','#ff6600','#ff0000','#960018'])
sns.palplot(sns.color_palette(color_scale))


# In[15]:


import warnings
warnings.filterwarnings("ignore")
#Functions to load data, color code the date and create geojson  features
def load_data(pollutant_ID,grouped_monthly_dropdown):
   
    agg_ts = grouped_monthly_dropdown[grouped_monthly_dropdown['parameter']==pollutant_ID]
    return agg_ts

def color_coding(poll, bin_edges):    
    idx = np.digitize(poll, bin_edges, right=True)
    return color_scale[idx]


def prepare_data(df, pollutant_ID):
    
    df['color'] = df.value.apply(color_coding, bin_edges=pollutants[pollutant_ID]['bin_edges'])
    return df

def create_geojson_features(df):
  
    
    features = []
    for _, row in df.iterrows():
        feature = {
            'type': 'Feature',
            'geometry': {
                'type':'Point', 
                'coordinates':[row['longitude'],row['latitude']]
            },
            'properties': {
                'time': row['date'].date().__str__(),
                'style': {'color' : row['color']},
                'icon': 'circle',
                'iconstyle':{
                    'fillColor': row['color'],
                    'fillOpacity': 0.8,
                    'stroke': 'true',
                    'radius': 7
                }
            }
        }
        features.append(feature)
    return features

def make_map(features):
    
    coords_delhi=[28.65381,77.22897]
    pollution_map = folium.Map(location=coords_delhi, control_scale=True, zoom_start=8)

    TimestampedGeoJson(
        {'type': 'FeatureCollection',
        'features': features}
        , period='P1M'
        , add_last_point=True
        , auto_play=False
        , loop=False
        , max_speed=1
        , loop_button=True
        , date_options='YYYY/MM'
        , time_slider_drag_update=True
    ).add_to(pollution_map)
    print('> Done.')
    return pollution_map

def plot_pollutant(pollutant_ID):
    print('Mapping {} pollution in India in 2016-2018'.format(pollutants[pollutant_ID]['name']))
    pollutant_map_df = load_data(pollutant_ID,grouped_monthly_dropdown)
    pollutant_map_df = prepare_data(pollutant_map_df, pollutant_ID)
    features = create_geojson_features(pollutant_map_df)
    return make_map(features), pollutant_map_df


# In[16]:


#MAP for No2, slide the slider to see it varying over time
pollution_map_no2, pollutant_map_df_no2 = plot_pollutant('no2')
pollution_map_no2.save('/kaggle/input/pollution_no2.html')
pollution_map_no2


# The NO2 concentrations appears to be constant through the 2 year time period. Towards the end of 2016, there seemed to be a slight increase in the pollutant levels but appears to have reduced thereafter. <br>There are no striking values that can be observed. 

# In[17]:


#Map for CO
pollution_map_co, pollutant_map_df_co = plot_pollutant('co')
pollution_map_co.save('/kaggle/input/pollution_co.html')
pollution_map_co


# The CO levels appear to be extremely high throughout the duration. Pollution from vehicles is a major contributor to high carbon monoxide levels. <br>However, researchers believe winds carry the CO produced by biomass burning in Africa and Southeast Asian countries to the Indian subcontinent, thus adding to the already high levels of the gas in the country's atmosphere. <br>
# This appears to be a plausible explanation but we cannot be sure since the data can be misrepresentative.

# In[18]:


#Map for o3
pollution_map_o3, pollutant_map_df_o3 = plot_pollutant('o3')
pollution_map_o3.save('/kaggle/input/pollution_o3.html')
pollution_map_o3


# The Ozone levels also appear to be stationary and there seems to be no sudden change.

# In[19]:


#Map for pm10
pollution_map_pm10, pollutant_map_df_pm10 = plot_pollutant('pm10')
pollution_map_pm10.save('/kaggle/input/pollution_pm10.html')
pollution_map_pm10


# The general trend that can be observed is the the concentration of pm10 has been increasing. There appeared to be a phase in between when the levels had gone down but they shot back up right after.

# In[20]:


#Map for pm25
pollution_map_pm25, pollutant_map_df_pm25 = plot_pollutant('pm25')
pollution_map_pm25.save('/kaggle/input/pollution_pm25.html')
pollution_map_pm25


# The pm25 values initially seemed to be improving as they moved to the greener side but they steadily move towards the red palette indicating deteriorating air quality.

# **--AQI Calculations**

# In[28]:


#AQI Code
#Formula for calculating AQI using linear segmengted formula
# p= [{(IHI - ILO)/ (BHI -BLO)} * (Cp-BLO)] + ILO
# 
# BHI= Breakpoint concentration greater or equal to given conc.
# BLO= Breakpoint concentration smaller or equal to given conc.
# 
# IHI = AQI value corresponding to BHI
# 
# ILO = AQI value corresponding to BLO


# In[29]:


#Defining breakpoints for different pollutants as per http://www.indiaenvironmentportal.org.in/files/file/Air%20Quality%20Index.pdf
co_bp=[1000,2000,10000,17000,34000]
no2_bp=[40,80,180,280,400]
pm10_bp=[50,100,250,350,430]
pm25_bp=[30,60,90,150,250]
o3_bp=[50,100,200,265,748]
so2_bp=[40,80,380,800,1600]

#order in dictionary PM10 PM2.5 NO2 O3 CO SO2
dict_aqi_mapping_pm10={'0-50':[0,50],'51-100':[51,100],'101-200':[101,250],'201-300':[251,350],'301-400':[351,430],'401-500':[430,600]}
dict_aqi_mapping_pm25={'0-50':[0,30],'51-100':[31,60],'101-200':[61,90],'201-300':[91,120],'301-400':[121,250],'401-500':[250,400]}
dict_aqi_mapping_no2={'0-50':[0,40],'51-100':[41,80],'101-200':[81,180],'201-300':[181,280],'301-400':[281,400],'401-500':[400,600]}
dict_aqi_mapping_o3={'0-50':[0,50],'51-100':[51,100],'101-200':[101,168],'201-300':[169,208],'301-400':[209,748],'401-500':[748,900]}
dict_aqi_mapping_co={'0-50':[0,1000],'51-100':[1001,2000],'101-200':[2001,10000],'201-300':[10001,17000],'301-400':[17001,34000],'401-500':[34000,40000]}
dict_aqi_mapping_so2={'0-50':[0,40],'51-100':[41,80],'101-200':[81,380],'201-300':[381,800],'301-400':[801,1600],'401-500':[1600,2000]}

breakpoint_pm10=[0,50,100,250,350,430,600]
breakpoint_pm25=[0,30,60,90,150,250,400]
breakpoint_no2=[0,40,80,180,280,400,600]
breakpoint_o3=[0,50,100,200,265,748,900]
breakpoint_co=[0,1000,2000,10000,17000,34000,40000]
breakpoint_so2=[0,40,80,380,800,1600,2000]


# In[30]:


#Function to calculate the AQI given the pollutant
def calculateAQI(cp,pollutant):
    if(pollutant=='pm10'):
        pollutant_bp=breakpoint_pm10
        pollutant_aqi_mapping=dict_aqi_mapping_pm10
    elif(pollutant=='pm25'):
        pollutant_bp=breakpoint_pm25
        pollutant_aqi_mapping=dict_aqi_mapping_pm25
    elif(pollutant=='no2'):
        pollutant_bp=breakpoint_no2
        pollutant_aqi_mapping=dict_aqi_mapping_no2
    elif(pollutant=='so2'):
        pollutant_bp=breakpoint_so2
        pollutant_aqi_mapping=dict_aqi_mapping_so2
    elif(pollutant=='o3'):
        pollutant_bp=breakpoint_o3
        pollutant_aqi_mapping=dict_aqi_mapping_o3
    elif(pollutant=='co'):
        pollutant_bp=breakpoint_co
        pollutant_aqi_mapping=dict_aqi_mapping_co
    if(cp==0):
        return 0
    flag=0
    j=pollutant_bp[0]
    for i in pollutant_bp:
        if(i>=cp and flag==0):
            bhi=i
            blo=j
            flag=1
        j=i
        
#--Arbitary value 600(Greater than max AQI possible)
    if(flag==0):
        return 600
    else:
        found_ihi=0
        found_ilo=0
        for i in pollutant_aqi_mapping.keys():
            if(pollutant_aqi_mapping[i][1]>=bhi and found_ihi==0):

                ihilo=i.split('-')
                ihi=float(ihilo[1])
                ilo=float(ihilo[0])
                found_ihi=1

        ip=(((ihi-ilo)/(bhi-blo))*(cp-blo))+ilo
        return ip


# In[31]:


#Grouped on day,city,parameter
#Have to sort the df first
#Monthly not daily as map does not work for daily due to excessive amounts of data points
grouped_daily_AQI=grouped_monthly_dropdown
grouped_daily_AQI=grouped_daily_AQI.reset_index(drop=True)
grouped_daily_AQI=grouped_daily_AQI.sort_values(by=['date','city'])

city_l=[]
date_l=[]
val_l=[]
long_l=[]
lat_l=[]
temp_date=str(grouped_daily_AQI['date'][0])
temp_city=grouped_daily_AQI['city'][0]

l=[]
for i in grouped_daily_AQI.index:
    if str(grouped_daily_AQI.loc[i,'date'])==temp_date:
        
        if grouped_daily_AQI.loc[i,'city']!=temp_city:
            max_val=max(l)
            date_l.append(temp_date)
            city_l.append(temp_city)
            val_l.append(max_val)
            lat_l.append(grouped_daily_AQI.loc[i,'latitude'])
            long_l.append(grouped_daily_AQI.loc[i,'longitude'])
            temp_city=grouped_daily_AQI.loc[i,'city']
            l=[]
            l.append(calculateAQI(grouped_daily_AQI.loc[i,'value'],grouped_daily_AQI.loc[i,'parameter']))
        else:
            l.append(calculateAQI(grouped_daily_AQI.loc[i,'value'],grouped_daily_AQI.loc[i,'parameter']))
    else:
        l=[]
        temp_date=str(grouped_daily_AQI.loc[i,'date'])
        temp_city=grouped_daily_AQI.loc[i,'city']
        l.append(calculateAQI(grouped_daily_AQI.loc[i,'value'],grouped_daily_AQI.loc[i,'parameter']))
        
#Create new dataframe by going through dict_AQI

data_df = {'date':date_l,
        'city':city_l,
          'value_AQI':val_l,
          'latitude':lat_l,
          'longitude':long_l}
 
# Create DataFrame
df_AQI = pd.DataFrame(data_df)


# In[32]:


#Create a new dataframe with the monthly data per city per pollutant along with its Textual Interpretation
l_text_AQI=[]
dict_aqi_mapping={50:'Good',100:'Satisfactory',200:'Moderately Polluted',300:'Poor',400:'Very Poor',500:'Severe',1000:'Extreme'}
for i in df_AQI.index:
    flag=0
    temp=df_AQI.loc[i,'value_AQI']
    for j,val in dict_aqi_mapping.items():
        if(flag!=0):
            break
        if(j>temp):
            l_text_AQI.append(val)
            flag=1
        
df_AQI['text_AQI']=l_text_AQI


# In[33]:


df_AQI.head()


# **--MAP for AQI--**

# In[34]:


color_scale_AQI = np.array(['#10ff00','#ccff00','#ffff00','#FFCC00','#ff9900','#ff6600','#ff0000'])
sns.palplot(sns.color_palette(color_scale_AQI))


# In[35]:


#----AQI MAP
pollutants_AQI={
    'value_AQI': {
        'notation' : 'AQI',
        'name' :'Air Quality Index(AQI)',
        'bin_edges' : np.array([50,100,200,300,400,500,1000])
    }
}


def color_coding_AQI(poll, bin_edges):    
    idx = np.digitize(poll, bin_edges, right=True)
    return color_scale_AQI[idx]


def prepare_data_AQI(df, pollutant_ID):
    print('> Preparing data...')
    #df = df.reset_index().merge(meta, how='inner', on='city').set_index('DatetimeBegin')
    #df = df.loc[:, ['SamplingPoint','Latitude', 'Longitude', 'AirPollutionLevel']]
    #df = df.groupby('SamplingPoint', group_keys=False).resample(rule='M').last().reset_index()
    df['color'] = df.value_AQI.apply(color_coding_AQI, bin_edges=pollutants_AQI[pollutant_ID]['bin_edges'])
    return df
def create_geojson_features_AQI(df):
    print('> Creating GeoJSON features...')
    features = []
    for _, row in df.iterrows():
        feature = {
            'type': 'Feature',
            'geometry': {
                'type':'Point', 
                'coordinates':[row['longitude'],row['latitude']]
            },
            'properties': {
                'time': row['date'],
                'style': {'color' : row['color']},
                'icon': 'circle',
                'iconstyle':{
                    'fillColor': row['color'],
                    'fillOpacity': 0.8,
                    'stroke': 'true',
                    'radius': 7
                }
            }
        }
        features.append(feature)
    return features


def make_map_AQI(features):
    #print('> Making map...')
    coords_india=[28.65381,77.22897]
    pollution_map = folium.Map(location=coords_india, control_scale=True, zoom_start=4)

    TimestampedGeoJson(
        {'type': 'FeatureCollection',
        'features': features}
        , period='P1M'
        , add_last_point=True
        , auto_play=False
        , loop=False
        , max_speed=1
        , loop_button=True
        , date_options='YYYY/MM/DD'
        , time_slider_drag_update=True
    ).add_to(pollution_map)
    #print('> Done.')
    return pollution_map

def plot_pollutant_AQI(pollutant_ID):
    print('Mapping {} levels in India during 2016-2018'.format(pollutants_AQI[pollutant_ID]['name']))
    #pollutant_map_df = load_data_AQI(pollutant_ID,df_AQI)
    #df = clean_data(df)
    pollutant_map_df_AQI = prepare_data_AQI(df_AQI, pollutant_ID)
    features = create_geojson_features_AQI(pollutant_map_df_AQI)
    return make_map_AQI(features), pollutant_map_df_AQI

#Change the name below to any polllutant you want i.e pollution_map, df = plot_pollutant('no2')
pollution_map_AQI, pollutant_map_df_AQI = plot_pollutant_AQI('value_AQI')
pollution_map_AQI.save('/kaggle/input/pollution_AQI.html')
pollution_map_AQI


# We observe that the general trend in AQI is that it seems to be increasing which implies worsening air quality. <br> The last month i.e 2018/04 appears to be a sudden shift and we believe it is something wrong with the data itself.

# **Effects of Crop Burning **

# In[36]:


delhi_pm10=grouped_dropdown[grouped_dropdown['city']=='Delhi']
delhi_pm10=delhi_pm10[delhi_pm10['parameter']=='pm10']
delhi_pm10 = delhi_pm10[['date','value']]


bengaluru_pm10=grouped_dropdown[grouped_dropdown['city']=='Bengaluru']
bengaluru_pm10=bengaluru_pm10[bengaluru_pm10['parameter']=='pm10']
bengaluru_pm10 = bengaluru_pm10[['date','value']]
import plotly.express as px
fig_delhi = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="pm10 values")
    ))
fig_delhi.add_trace(go.Scatter(x=delhi_pm10['date'], y=delhi_pm10['value'],
                    mode='markers',
                    name='Delhi',))
fig_delhi.add_trace(go.Scatter(x=bengaluru_pm10['date'], y=bengaluru_pm10['value'],
                    mode='markers',
                    name='Bangalore'))

fig_delhi.show()


# We observe that in Delhi the plot of pm10 shows a U shape with the peaks occuring between October - January.<br>
# This trend has stayed to be the same for both the years.
# What is so peculiar about these months?<br>
# Well, on actually spending some time analysing the data and reading up a lot of articles we'd like to present to you :<br>
# "**India's burning issue of crop burning**"<br>
# Farmers  in India find buring off the residuals an easier way to get rid of the previous harvest
# and to make their plot ready for the next season as quickly as possible.<br>
# Deprived of equally cheap and easy alternatives of preparing the fields, farmers have continued to flout the law by burning the harvest remains.<br>
# When asked, this is what our kisaan has to say "I can not afford to buy a machine and even to rent it is 10,000 rupees , maybe more. To burn it is just 1,000 rupees and the next day it is done.”
# <br>
# <br>
# To make this fact more promising we can observe the striking difference in the trend as compared to Bangalore which does not have a lot of agriculture and is mostly a commercial city.<br>
# This strengthens the claim that the crop burning is actually impacting the lives of Indians in a bad way.
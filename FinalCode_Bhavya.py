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

**--Diwali--**

# In[21]:


#data for diwali of the three years
#2016 - Sun 30 Oct
#2017 - Thu 19 Oct
#2018 - Wed 7 Nov

#2016 dates - 28,29,30,31,1 Oct-Nov
#2017 dates - 17,18,19,20,21 Oct
#2018 dates - 5,6,7,8,9 Nov

#2016 -> 20,21,22,23,24,25,26,27,28,29, 30 ,31,1,2,3,4,5,6,7,8,9
#2017 -> 9,10,11,12,13,14,15,16,17,18, 19 ,20,21,22,23,24,25,26,27,28,29

#bursting of crackers - ozone, no2, so2
#burning of fuels - pm, co, no2, so2


# In[22]:


#grouping the data to get one value per city, per pollutant, per day
data_daily_diwali=df
data_daily_diwali=data_daily_diwali.drop(['utc','country','attribution','location','unit'],axis=1)
data_daily_diwali['local'] = data_daily_diwali['local'].map(lambda x: str(x)[:10])
data_daily_diwali['local']=pd.to_datetime(data_daily_diwali['local'],format="%Y-%m-%d")

agg_daily_diwali=data_daily_diwali
grouped_daily_diwali=agg_daily_diwali.groupby(['local','city','parameter'])
grouped_daily_diwali=grouped_daily_diwali.mean()

grouped_daily_diwali.insert(3,'date',pd.to_datetime('2016-01'))
grouped_daily_diwali.insert(4,'city',0)
grouped_daily_diwali.insert(5,'parameter',0)

for i in grouped_daily_diwali.index:
    grouped_daily_diwali.date[i]=i[0]
    grouped_daily_diwali.city[i]=i[1]
    grouped_daily_diwali.parameter[i]=i[2]

grouped_daily=grouped_daily_diwali


# In[23]:


#getting dates for diwali days
y16_oct = list(range(20,32))
y16_nov = list(range(1,10))
y17 = list(range(9,30))

y_comb = [y16_oct, y16_nov, y17]
year_month = ["2016-10-", "2016-11-", "2017-10-"]
dat = []
for i in range(len(y_comb)):
    for j in range(len(y_comb[i])):
        if (y_comb[i][j] < 10):
            y_comb[i][j] = '0' + str(y_comb[i][j])
        else:
            y_comb[i][j] = str(y_comb[i][j])

#print("Diwali dates across the years = ", y_comb)


# We create a dropdown for choosing the city and pollutant to see the trend for the five days of diwali as compared to 10 days before and after

# In[24]:



#-----------------------------------------creating temp dataframe to hold the values of the diwali dates--------------------------
#Initial for Delhi and No2
df_sub = grouped_daily_diwali.loc[grouped_daily_diwali['city'] == 'Delhi']
df_sub = df_sub.loc[df_sub['parameter'] == 'no2']

years_data = []
for ym in range(len(year_month)):
    y = []
    for d in y_comb[ym]:
        c = year_month[ym] + d
        df1_sub = df_sub[df_sub['date'].astype(str).str.contains(c)]
        yd = df1_sub.value.mean()
        y.append(yd)
    years_data.append(y)

y16 = []
for i in years_data[0]:
    y16.append(i)
for i in years_data[1]:
    y16.append(i)

y17_data = []
for i in years_data[2]:
    y17_data.append(i)

#--------------------------------------- Create a dropdown ---------------------------------------------------

city_diwali = widgets.Dropdown(
    description='City:   ',
    value='Delhi',
    options=grouped_daily_diwali['city'].unique().tolist()
)
parameter_diwali = widgets.Dropdown(
    options=list(grouped_daily_diwali['parameter'].unique()),
    value='no2',
    description='Parameter:   ',
)

trace1_diwali = go.Scatter(y=y16, mode='lines', name='year 2016', line=dict(color='rgb(255, 133, 133)'))
trace2_diwali = go.Scatter(y=y17_data, mode='lines', name='Year 2017', line=dict(color='rgb(16, 216, 235)'))
trace3_diwali = go.Scatter(y=y16[8:13], x = list(range(8,13)), name = "Diwali day 2016", line=dict(color='rgb(250, 5, 5)'))
trace4_diwali = go.Scatter(y=y17_data[8:13], x = list(range(8,13)), name = "Diwali day 2017", line=dict(color='rgb(12, 10, 128)'))

g_diwali = go.FigureWidget(data=[trace1_diwali, trace2_diwali, trace3_diwali, trace4_diwali],
                    layout=go.Layout(
                        title=dict(
                            text='Pollutant levels during diwali days for 2016,2017'
                        )
                    ))

def response(change):
        #-----------------------------------------creating temp dataframe--------------------------
        df_sub = grouped_daily_diwali.loc[grouped_daily_diwali['city'] == city_diwali.value]
        df_sub = df_sub.loc[df_sub['parameter'] == parameter_diwali.value]

        years_data = []
        for ym in range(len(year_month)):
            y = []
            for d in y_comb[ym]:
                c = year_month[ym] + d
                df1_sub = df_sub[df_sub['date'].astype(str).str.contains(c)]
                yd = df1_sub.value.mean()
                y.append(yd)
            years_data.append(y)

        y16 = []
        for i in years_data[0]:
            y16.append(i)
        for i in years_data[1]:
            y16.append(i)

        y17_data = []
        for i in years_data[2]:
            y17_data.append(i)        
        #------------------------------------------------------------------------------------------
        y0 = y16
        y1 = y17_data
        y2 = y16[8:13]
        x2 = list(range(8,13))
        y3 = y17_data[8:13]
        x3 = list(range(8,13))
        with g_diwali.batch_update():
            g_diwali.data[0].y = y0
            g_diwali.data[1].y = y1
            g_diwali.data[2].y = y2
            g_diwali.data[3].y = y3
            g_diwali.data[2].x = x2
            g_diwali.data[3].x = x3
            g_diwali.layout.xaxis.title = 'Date'
            g_diwali.layout.yaxis.title = 'Pollutant levels'


city_diwali.observe(response, names="value")
parameter_diwali.observe(response, names="value")
container2_diwali = widgets.HBox([city_diwali, parameter_diwali])
widgets.VBox([container2_diwali,g_diwali])


# Since people burst a lot of crackers during diwali, we expect an increase in the pollutant levels.
# * We notice that there is a sharp rise in the NO2 levels in 2016, in Delhi, for the second day of Diwali as compared to the days before and after. This is in accordance with what we expected. Also, the pollutant levels are higher in 2016 than in 2017 which could suggest that people are becoming more sensitive towards the environment now.
# * In Mumbai as well, we notice that 2016 NO2 values are much higher than 2017 ones. And for 2017, the graph peaks on third day of Diwali. This is again in accordance with the expectation. And pm2.5 and pm10 values are much higher during the five days than the remaining.
# * In Bengaluru 2017, pm10 rises to very high levels during Diwali
# * Jodhpur also shows very high trends of pm10 during those days for both the years.

# **--Weekend vs Weekday--**

# In[25]:


#function to get the day from the given date
def get_day(my_date):
    my_date = str(my_date)
    y = int(my_date[:4])
    m = int(my_date[5:7])
    d = int(my_date[8:10])

    my_day = date(y, m, d).weekday()
    return calendar.day_name[my_day]


# In[26]:


#Dropdown For Weekend vs Weekday
#-----------------------------------------creating temp dataframe--------------------------
weekday = pd.DataFrame()
weekend = pd.DataFrame()
df_no2 = grouped_daily_diwali.loc[grouped_daily_diwali['parameter'] == 'no2']
df_no2 = df_no2.loc[df_no2['city'] == 'Delhi']
df_no2 = df_no2[df_no2['date'].astype(str).str.contains("2016")]

for i in range(len(df_no2)):
    if ((get_day(df_no2.iloc[i,3]) == 'Sunday') or (get_day(df_no2.iloc[i,3]) == 'Saturday')):
        weekend = weekend.append(df_no2.iloc[i,:])
    else:
        weekday = weekday.append(df_no2.iloc[i,:])

#----------------------------------- Create the dropdown -------------------------------------------------------

city_ww = widgets.Dropdown(
    description='City:   ',
    value='Delhi',
    options=grouped_daily_diwali['city'].unique().tolist()
)

parameter_ww = widgets.Dropdown(
    options=list(grouped_daily_diwali['parameter'].unique()),
    value='no2',
    description='Parameter:   ',
)

year_ww = widgets.Dropdown(
    options=["2016","2017","2018"],
    value='2016',
    description='Year:   ',
)

trace1_ww = go.Bar(x=weekend['date'], y=weekend['value'], name='Weekends')
trace2_ww = go.Bar(x=weekday['date'], y=weekday['value'], name='Weekdays')
g_ww = go.FigureWidget(data=[trace1_ww, trace2_ww],
                    layout=go.Layout(
                        title=dict(
                            text='Pollutant levels on weekends vs weekdays'
                        )
                    ))
def response(change):

        #-----------------------------------------creating temp dataframe--------------------------
        weekday = pd.DataFrame()
        weekend = pd.DataFrame()
        df_no2 = grouped_daily_diwali.loc[grouped_daily_diwali['parameter'] == parameter_ww.value]
        df_no2 = df_no2.loc[df_no2['city'] == city_ww.value]
        df_no2 = df_no2[df_no2['date'].astype(str).str.contains(year_ww.value)]

        for i in range(len(df_no2)):
            if ((get_day(df_no2.iloc[i,3]) == 'Sunday') or (get_day(df_no2.iloc[i,3]) == 'Saturday')):
                weekend = weekend.append(df_no2.iloc[i,:])
            else:
                weekday = weekday.append(df_no2.iloc[i,:])
        #------------------------------------------------------------------------------------------

        x0 = weekend['date']
        y0 = weekend['value']
        x1 = weekday['date']
        y1 = weekday['value']
        with g_ww.batch_update():
            g_ww.data[0].x = x0
            g_ww.data[0].y = y0
            g_ww.data[1].x = x1
            g_ww.data[1].y = y1
            g_ww.layout.xaxis.title = 'Date'
            g_ww.layout.yaxis.title = 'Pollutant levels of ' + parameter_ww.value

city_ww.observe(response, names="value")
parameter_ww.observe(response, names="value")
year_ww.observe(response, names="value")
container2_ww = widgets.HBox([city_ww, parameter_ww,year_ww])
widgets.VBox([container2_ww,g_ww])


# From this plot we can gain the following insights - 
# * The pm2.5 values for Delhi in 2016 are extremely high on 6th November. If we look up what happened on that day, we notice that on this day the Indian government declared levels of air pollution in Delhi as an emergency situation, closing schools and construction sites. Pm2.5 particles are very harmful because they can reach deep into the lungs and breach the blood-brian barrier. And on this day the pm2.5 levels had reached values more thann 16 times the safe limit. This is very alarming.
# * The concentration of CO on 3rd December, 2017 was tremendously high compared to the neighbouring values. And when we search for what happened on this day as well, we found that there was a cricket match organinsed between India and Sri Lanka. And the players were forced to wear masks and many complained of breathing problems. This news had made headlines. Thus our observation from the data is validated by true events that have occured on those days

# In[27]:


#Box plot for weekend vs weekday
#-----------------------------------------creating temp dataframe--------------------------
weekday_box = pd.DataFrame()
weekend_box = pd.DataFrame()
df_no2_box = grouped_daily_diwali.loc[grouped_daily_diwali['parameter'] == 'no2']
df_no2_box = df_no2_box.loc[df_no2_box['city'] == 'Delhi']
df_no2_box = df_no2_box[df_no2_box['date'].astype(str).str.contains("2016")]

for i in range(len(df_no2_box)):
    if ((get_day(df_no2_box.iloc[i,3]) == 'Sunday') or (get_day(df_no2_box.iloc[i,3]) == 'Saturday')):
        weekend_box = weekend_box.append(df_no2_box.iloc[i,:])
    else:
        weekday_box = weekday_box.append(df_no2_box.iloc[i,:])
#------------------------------------------------------------------------------------------

city_ww_box = widgets.Dropdown(
    description='City:   ',
    value='Delhi',
    options=grouped_daily_diwali['city'].unique().tolist()
)

parameter_ww_box = widgets.Dropdown(
    options=list(grouped_daily_diwali['parameter'].unique()),
    value='no2',
    description='Parameter:   ',
)

year_ww_box = widgets.Dropdown(
    options=["2016","2017","2018"],
    value='2016',
    description='Year:   ',
)

trace0_ww_box = go.Box(y=weekday_box['value'], name = 'Weekdays')
trace1_ww_box = go.Box(y=weekend_box['value'], name = 'Weekends')
g_ww_box = go.FigureWidget(data=[trace0_ww_box, trace1_ww_box],
                    layout=go.Layout(
                        title=dict(
                            text='Pollutant levels on weekends vs weekdays'
                        )
                    ))
def response(change):

        #-----------------------------------------creating temp dataframe--------------------------
        weekday_box = pd.DataFrame()
        weekend_box = pd.DataFrame()
        df_no2_box = grouped_daily_diwali.loc[grouped_daily_diwali['parameter'] == parameter_ww_box.value]
        df_no2_box = df_no2_box.loc[df_no2_box['city'] == city_ww_box.value]
        df_no2_box = df_no2_box[df_no2_box['date'].astype(str).str.contains(year_ww_box.value)]

        for i in range(len(df_no2_box)):
            if ((get_day(df_no2_box.iloc[i,3]) == 'Sunday') or (get_day(df_no2_box.iloc[i,3]) == 'Saturday')):
                weekend_box = weekend_box.append(df_no2_box.iloc[i,:])
            else:
                weekday_box = weekday_box.append(df_no2_box.iloc[i,:])
        #------------------------------------------------------------------------------------------

        y0 = weekday_box['value']
        y1 = weekend_box['value']
        with g_ww_box.batch_update():
            g_ww_box.data[0].y = y0
            g_ww_box.data[1].y = y1
            g_ww_box.layout.xaxis.title = 'Date'
            g_ww_box.layout.yaxis.title = 'Pollutant levels of ' + parameter_box.value


city_ww_box.observe(response, names="value")
parameter_ww_box.observe(response, names="value")
year_ww_box.observe(response, names="value")
container2_ww_box = widgets.HBox([city_ww_box, parameter_ww_box,year_ww_box])
widgets.VBox([container2_ww_box,g_ww_box])


# We also wanted to see if there was any difference in pollutant levels between weekends and weekdays. The graph shows that there isn’t any significant difference between the two. The boxplot also shows the same. The values on weekdends is very slighty lesser than the weekady



# Now we are trying to see how the rainfall in a region affects the AQI values. We use the monthly AQI data along with the monthly rainfall in varios sub divisions of the country.

# In[40]:


#Haryana Delhi & Chandigarh = subdivision, delhi = city
df_rainfall = pd.read_csv("../input/rainfall-in-india/Sub_Division_IMD_2017.csv")
df_rain_16 = df_rainfall.loc[df_rainfall['YEAR'] == 2016]
df_rain_17 = df_rainfall.loc[df_rainfall['YEAR'] == 2017]

df_temp_16 = df_temp.loc[df_temp['YEAR'] == 2016]
df_temp_17 = df_temp.loc[df_temp['YEAR'] == 2017]

df_rain_16 = df_rain_16.loc[df_rain_16['SUBDIVISION'] == 'Haryana Delhi & Chandigarh']
df_rain_17 = df_rain_17.loc[df_rain_17['SUBDIVISION'] == 'Haryana Delhi & Chandigarh']


#getting the data for delhi, added extra columns for rainfall and month for easy retrieval from the rainfall dataset
#its called df_beng but has values for delhi itself
df_beng = df_AQI.loc[df_AQI['city'] == 'Delhi']
df_beng['year'] = df_beng['date'].map(lambda x: x[:4])
df_beng['date'] = df_beng['date'].map(lambda x: x[:7])
df_beng = df_beng.loc[(df_beng['year'] == '2016') | (df_beng['year'] == '2017')]
df_beng['month'] = df_beng['date'].map(lambda x: int(x[5:7]))
df_beng_16 = df_beng.loc[df_beng['year'] == '2016']
df_beng_17 = df_beng.loc[df_beng['year'] == '2017']
df_beng_16

#adding value of rainfall
rain = []
for i in range(len(df_beng_16)):
    
    month = int(df_beng_16.iloc[i][8]) + 1
    print("filling in = ", df_rain_16.iloc[0][month])
    rain.append(df_rain_16.iloc[0][month])
df_beng_16['rain'] = rain

rain = []
for i in range(len(df_beng_17)):
    #print("index = ", i, "month = ", int(df_beng_17.iloc[i][7]))
    month = int(df_beng_17.iloc[i][8]) + 1
    #print("filling in = ", df_rain_17.iloc[0][month])
    rain.append(df_rain_17.iloc[0][month])
df_beng_17['rain'] = rain

#normalizing the data for scatter plot
df_beng_16_norm = df_beng_16
df_beng_17_norm = df_beng_17
df_beng_16_norm['value_AQI'] = (df_beng_16_norm['value_AQI'] - min(df_beng_16_norm['value_AQI'])) / ( max(df_beng_16_norm['value_AQI']) - min(df_beng_16_norm['value_AQI']) )
df_beng_17_norm['value_AQI'] = (df_beng_17_norm['value_AQI'] - min(df_beng_17_norm['value_AQI'])) / ( max(df_beng_17_norm['value_AQI']) - min(df_beng_17_norm['value_AQI']) )

df_beng_16_norm['rain'] = (df_beng_16_norm['rain'] - min(df_beng_16_norm['rain'])) / ( max(df_beng_16_norm['rain']) - min(df_beng_16_norm['rain']) )
df_beng_17_norm['rain'] = (df_beng_17_norm['rain'] - min(df_beng_17_norm['rain'])) / ( max(df_beng_17_norm['rain']) - min(df_beng_17_norm['rain']) )

#concatenating the data frames, plotting scatterplot
df_beng_norm = df_beng_16_norm.append(df_beng_17_norm, ignore_index = True)

import plotly.express as px
fig = px.scatter(df_beng_norm, x="value_AQI", y="rain", color="year",size='value_AQI', hover_data=['value_AQI', 'rain'])
fig.update_layout(
    title="AQI vs Rainfall",
    xaxis_title="AQI (Normalised)",
    yaxis_title="Rainfall (Normalised)"
)
fig.show()


# The cluster of points around high AQI and low rainfall areas of the plot show that regions that have less rainfall have a higher AQI. This could suggest that rainfall helps clear out the air thus reducing the AQI. <br> As it rains, the rain brings down the gaseous pollutants and thus reducing the pollution level in the air but this would actually lead to the ground water to get contaminated.

# In[41]:


#rainfall, temperature, AQI all together
#-------------------------------------------------aqi ----------------------------------------
#removing location and taking only the monthly avg aqi values for all over india 
df_aqi_monthly=df_AQI.groupby('date')
df_aqi_monthly
df_aqi_monthly=df_aqi_monthly.mean()
df_aqi_monthly.insert(3,'date',pd.to_datetime('2016-01'))
for i in df_aqi_monthly.index:
    #print("i = ", i)
    df_aqi_monthly.date[i]=i
df_aqi_monthly['month'] = df_aqi_monthly['date'].map(lambda x: int(str(x)[5:7]))
df_aqi_monthly['year'] = df_aqi_monthly['date'].map(lambda x: str(x)[:4])

#-------------------------------------------------rain-----------------------------------------
#getting the monthly data for all locations for rainfall data
df_rain_16_monthly = df_rain_16.mean()
df_rain_17_monthly = df_rain_17.mean()

df_all = df_aqi_monthly
df_all_16 = df_all.loc[df_aqi_monthly['year'] == '2016']
df_all_17 = df_all.loc[df_aqi_monthly['year'] == '2017']

rain = []
for i in range(len(df_all_16)):
    #print("index = ", i, "month = ", df_all_16.iloc[i][4])
    month = df_all_16.iloc[i][4]
    #print("filling in = ", df_rain_16_monthly[month])
    rain.append(df_rain_16_monthly[month])
df_all_16['rain'] = rain

rain = []
for i in range(len(df_all_17)):
    #print("index = ", i, "month = ", df_all_17.iloc[i][4])
    month = df_all_17.iloc[i][4]
    #print("filling in = ", df_rain_17_monthly[month])
    rain.append(df_rain_17_monthly[month])
df_all_17['rain'] = rain

#-------------------------------------------------------temp----------------------------------------------------
temp = []
for i in range(len(df_all_16)):
    #print("index = ", i, "month = ", df_all_16.iloc[i][4])
    month = df_all_16.iloc[i][4]
    #print("filling in = ", df_temp_16.iloc[0][month])
    temp.append(df_temp_16.iloc[0][month])
df_all_16['temp'] = temp

temp = []
for i in range(len(df_all_17)):
    #print("index = ", i, "month = ", df_all_17.iloc[i][4])
    month = df_all_17.iloc[i][4]
    #print("filling in = ", df_temp_17.iloc[0][month])
    temp.append(df_temp_17.iloc[0][month])
df_all_17['temp'] = temp


#-----------------------------------------------normalize data-----------------------------------------------
df_all_16_norm = df_all_16
df_all_17_norm = df_all_17
df_all_16_norm['value_AQI'] = (df_all_16_norm['value_AQI'] - min(df_all_16_norm['value_AQI'])) / ( max(df_all_16_norm['value_AQI']) - min(df_all_16_norm['value_AQI']) )
df_all_17_norm['value_AQI'] = (df_all_17_norm['value_AQI'] - min(df_all_17_norm['value_AQI'])) / ( max(df_all_17_norm['value_AQI']) - min(df_all_17_norm['value_AQI']) )

df_all_16_norm['rain'] = (df_all_16_norm['rain'] - min(df_all_16_norm['rain'])) / ( max(df_all_16_norm['rain']) - min(df_all_16_norm['rain']) )
df_all_17_norm['rain'] = (df_all_17_norm['rain'] - min(df_all_17_norm['rain'])) / ( max(df_all_17_norm['rain']) - min(df_all_17_norm['rain']) )

df_all_16_norm['temp'] = (df_all_16_norm['temp'] - min(df_all_16_norm['temp'])) / ( max(df_all_16_norm['temp']) - min(df_all_16_norm['temp']) )
df_all_17_norm['temp'] = (df_all_17_norm['temp'] - min(df_all_17_norm['temp'])) / ( max(df_all_17_norm['temp']) - min(df_all_17_norm['temp']) )

#----------------------------------------------scatter plot----------------------------------------------------
df_all_norm = df_all_16_norm.append(df_all_17_norm, ignore_index = True)

import plotly.express as px
fig = px.scatter_3d(df_all_norm, x="value_AQI", y="rain", z="temp", color="year",size='value_AQI', hover_data=['value_AQI', 'rain', 'temp'])
fig.update_layout(
    title="AQI vs Rainfall vs Temperature",
    xaxis_title="AQI (Normalised)",
    yaxis_title="Rainfall (Normalised)"
    
)
fig.show()


# We observe that during the cold months of winter, the AQI value is higher because the cooler and denser air traps pollutants.<br> And it is usually during the winter months we see a lot of smog forming and brings in great trouble for breathing.

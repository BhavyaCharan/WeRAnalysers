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

#Creating a new dataframe consisting of monthly aggregated data i.e per month per city per pollutant
data_monthly_dropdown=df
data_monthly_dropdown=data_monthly_dropdown.drop(['local','country','attribution','location','unit'],axis=1)
data_monthly_dropdown['utc'] = data_monthly_dropdown['utc'].map(lambda x: str(x)[:-17])
#Converting to datetime
data_monthly_dropdown['utc']=pd.to_datetime(data_monthly_dropdown['utc'])

#Creating an aggregated dataframe
agg_monthly_dropdown=data_monthly_dropdown
#Grouping by the 3 columns to get unique values
grouped_monthly_dropdown=agg_monthly_dropdown.groupby(['utc','city','parameter'])
#Getting mean of each column
grouped_monthly_dropdown=grouped_monthly_dropdown.mean()


#Adding new columns for ease of work
grouped_monthly_dropdown.insert(3,'date',pd.to_datetime('2016-01'))
grouped_monthly_dropdown.insert(4,'city',0)
grouped_monthly_dropdown.insert(5,'parameter',0)

for i in grouped_monthly_dropdown.index:
    grouped_monthly_dropdown.date[i]=i[0]
    grouped_monthly_dropdown.city[i]=i[1]
    grouped_monthly_dropdown.parameter[i]=i[2]





# We now try to observe if certain days decalred as "**Bandh**" actually result in inactivity in the country or if it's just a big misconception.

# In[37]:


#Bharat bandh
# 28th Nov 2016 - demonetization
# 2nd Sept 2016 - trade union strike, bengal, up, kerala
# 2nd April 2018 - SC/ST act updates,several cities of Rajasthan, Gujarat, MP, Punjab, UP, Bihar, Delhi, Odisha,Jharkhand
# 10th sept 2018 - fuel prices - no data at all


# In[38]:


#getting dates for bandh days

#day1 demonetization
y16_nov_b = list(range(23,32))
y16_dec_b = list(range(1, 3))

#day2 trade union strike 
y16_aug_b = list(range(28,32))
y16_sep_b = list(range(1,8))

#day3  SC/ST act 
y18_mar_b = list(range(28,32))
y18_apr_b = list(range(1,8))

y_comb_b = [y16_nov_b, y16_dec_b, y16_aug_b, y16_sep_b, y18_mar_b, y18_apr_b]
year_month_b = ["2016-11-", "2016-12-", "2016-08-", "2016-09-", "2018-03-", "2018-04-"]
dat_b = []
for i in range(len(y_comb_b)):
    for j in range(len(y_comb_b[i])):
        if (y_comb_b[i][j] < 10):
            y_comb_b[i][j] = '0' + str(y_comb_b[i][j])
        else:
            y_comb_b[i][j] = str(y_comb_b[i][j])


# In[39]:


from ipywidgets import widgets

#-----------------------------------------creating temp dataframe--------------------------
df_part = grouped_daily_diwali.loc[grouped_daily_diwali['city'] == 'Delhi']
df_part = df_part.loc[df_part['parameter'] == 'no2']

years_data_b = []
for ym_b in range(len(year_month_b)):
    y_b = []
    for d_b in y_comb_b[ym_b]:
        c_b = year_month_b[ym_b] + d_b
        df1_part = df_part[df_part['date'].astype(str).str.contains(c_b)]
        yd_b = df1_part.value.mean()
        y_b.append(yd_b)
    years_data_b.append(y_b)

day1 = []
for i in years_data_b[0]:
    day1.append(i)
for i in years_data_b[1]:
    day1.append(i)

day2 = []
for i in years_data_b[2]:
    day2.append(i)
for i in years_data_b[3]:
    day2.append(i)

day3 = []
for i in years_data_b[4]:
    day3.append(i)
for i in years_data_b[5]:
    day3.append(i)

#------------------------------------------------------------------------------------------

city_bandh = widgets.Dropdown(
    description='City:   ',
    value='Delhi',
    options=grouped_daily_diwali['city'].unique().tolist()
)
parameter_bandh = widgets.Dropdown(
    options=list(grouped_daily_diwali['parameter'].unique()),
    value='no2',
    description='Parameter:   ',
)

trace1_bandh = go.Scatter(y=day1, mode='lines', name='Day1 Demonetization', line=dict(color='rgb(255, 133, 133)'))
pt1_bandh = go.Scatter(y=day1[5:6], x = list(range(5,6)), name = "Day1 Demonetization", line=dict(color='rgb(250, 5, 5)'))

trace2_bandh = go.Scatter(y=day2, mode='lines', name='Day2 Trade union strike', line=dict(color='rgb(16, 216, 235)'))
pt2_bandh = go.Scatter(y=day2[5:6], x = list(range(5,6)), name = "Day2 trade union strike", line=dict(color='rgb(12, 10, 128)'))

trace3_bandh = go.Scatter(y=day3, mode='lines', name='Day3 SC/ST act', line=dict(color='rgb(51, 204, 51)'))
pt3_bandh = go.Scatter(y=day3[5:6], x = list(range(5,6)), name = "Day3 SC/ST act", line=dict(color='rgb(0, 128, 0)'))

g_bandh = go.FigureWidget(data=[trace1_bandh, pt1_bandh, trace2_bandh, pt2_bandh, trace3_bandh, pt3_bandh],
                    layout=go.Layout(
                        title=dict(
                            text='Pollutant levels during bandhs for 2016,2017'
                        )
                    ))

def response(change):
        #-----------------------------------------creating temp dataframe--------------------------
        df_part = grouped_daily_diwali.loc[grouped_daily_diwali['city'] == city_bandh.value]
        df_part = df_part.loc[df_part['parameter'] == parameter_bandh.value]
        years_data_b = []
        for ym_b in range(len(year_month_b)):
            y_b = []
            for d_b in y_comb_b[ym_b]:
                c_b = year_month_b[ym_b] + d_b
                df1_part = df_part[df_part['date'].astype(str).str.contains(c_b)]
                if (len(df1_part) == 0):
                    continue
                    #print("got empty data")
                else:
                    yd_b = df1_part.value.mean()
                    y_b.append(yd_b)
            years_data_b.append(y_b)

       # print(years_data_b)
        
        day1 = []
        for i in years_data_b[0]:
            day1.append(i)
        for i in years_data_b[1]:
            day1.append(i)

        day2 = []
        for i in years_data_b[2]:
            day2.append(i)
        for i in years_data_b[3]:
            day2.append(i)

        day3 = []
        for i in years_data_b[4]:
            day3.append(i)
        for i in years_data_b[5]:
            day3.append(i)
        #------------------------------------------------------------------------------------------
        with g_bandh.batch_update():
            #print("calling the batch update funtion")
            g_bandh.data[0].y = day1
            g_bandh.data[1].y = day1[5:6]
            g_bandh.data[1].x = list(range(5,6))
            g_bandh.data[2].y = day2
            g_bandh.data[3].y = day2[5:6]
            g_bandh.data[3].x = list(range(5,6))
            g_bandh.data[4].y = day3
            g_bandh.data[5].y = day3[5:6]
            g_bandh.data[5].x = list(range(5,6))
            
            g_bandh.layout.xaxis.title = 'Date'
            g_bandh.layout.yaxis.title = 'Pollutant levels'


city_bandh.observe(response, names="value")
parameter_bandh.observe(response, names="value")
container2_bandh = widgets.HBox([city_bandh, parameter_bandh])
widgets.VBox([container2_bandh, g_bandh])


# The dots in the graph represent the day of the bandh and the line graph is drawn to give the viewer a perspective of the neighboring values. <br>
# We observe that Bandh days dont have any effect on the pollutant concentration. Most of us have an idea that "Bandh" implies 0 or almost no activity and its absolute silence in the country but the data does not seem to showcase the same fact.<br>
# Hence Bandh or no-bandh does not affect air pollution levels.

#Moving on to forecasting AQI for Delhi
#for forecasting, getting the daily aqi values
grouped_daily_AQI=grouped_daily_diwali
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
df_AQI_daily = pd.DataFrame(data_df)

df_AQI_delhi = df_AQI_daily[df_AQI_daily['city'] == "Delhi"]
df_AQI_delhi
df_AQI_beng = df_AQI_daily[df_AQI_daily['city'] == "Bengaluru"]
df_AQI_beng


# In[43]:


#Getting weekly AQI data from daily AQI data
df_AQI_delhi['conv_date']=pd.to_datetime(df_AQI_delhi.date)
df_week_delhi = df_AQI_delhi.resample('W',on="conv_date").mean()
df_week_delhi

df_AQI_beng['conv_date']=pd.to_datetime(df_AQI_beng.date)
df_week_beng = df_AQI_beng.resample('W',on="conv_date").mean()
df_week_beng


# In[44]:


#cleaning na values for bengaluru
print(df_week_beng.value_AQI.isna().sum())
print(df_week_beng.loc[pd.isna(df_week_beng['value_AQI']), :])
df_week_beng = df_week_beng[df_week_beng.index != '2018-03-04']
print(df_week_beng.loc[pd.isna(df_week_beng['value_AQI']), :])


# In[45]:


#seasonal decomposition
components = seasonal_decompose(df_week_delhi['value_AQI'])
components.plot()
plt.show()


# In[46]:


#For non stationary
plt.figure(figsize=(30,10))
additive = components.trend + components.seasonal + components.resid
multiplicative = components.trend * components.seasonal * components.resid
#print(additive.sample(30))
plt.plot(components.observed, label="Original")

plt.plot(additive, label="Additive")
#plt.plot(multiplicative, label="Multiplicative")
plt.legend()
plt.show()

plot_acf(df_week_delhi['value_AQI'], lags = 70)
plt.xlabel('Lag')
plt.ylabel('Auto correlations')
plt.title('Plot of ACF')
plt.show()

plot_pacf(df_week_delhi['value_AQI'], lags = 70)
plt.xlabel('Lag')
plt.ylabel('Partial Auto correlations')
plt.title('Plot of PACF')
plt.show()


# In[47]:


#before converting to stationary
dftest = adfuller(df_week_delhi['value_AQI'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)
#not stationary

df_week_delhi['value_AQI_diff'] = df_week_delhi['value_AQI'] - df_week_delhi['value_AQI'].shift(1)
#Repeating the 0th value for the NA
df_week_delhi['value_AQI_diff'][0] = df_week_delhi['value_AQI'][0]

#Performing the Dickey Fuller test
print("\n\nperforming differencing and then dickey fuller\n\n")
dftest = adfuller(df_week_delhi['value_AQI_diff'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)
#became stationary


# In[48]:


#After converting to stationary
additive = components.trend + components.seasonal + components.resid
plt.plot(components.observed, label="Original")
plt.plot(additive, label="Additive")

plot_acf(df_week_delhi['value_AQI_diff'], lags = 70)
plt.xlabel('Lag')
plt.ylabel('Auto correlations')
plt.title('Plot of ACF')
plt.show()

plot_pacf(df_week_delhi['value_AQI_diff'], lags = 70)
plt.xlabel('Lag')
plt.ylabel('Partial Auto correlations')
plt.title('Plot of PACF')
plt.show()


# In[49]:


df_test_delhi = df_week_delhi[78:118]
df_train_delhi = df_week_delhi[0:78]


# In[50]:


#We know that the order of differencing = 1  
#From the PACF plot we know that p should be = 1 and hence keep it fixed
import warnings
import itertools
warnings.filterwarnings("ignore")
import statsmodels.api as sm

p=[0,1]
d=[0,1]
q=[0,1]

min_aic=10000
pdq = list(itertools.product(p, d, q))
print(pdq)
P=[i for i in range(0,3)]
#change this to 0,1,2
D=[0,1]
Q=[i for i in range(0,3)]
PDQ=list(itertools.product(P, D, Q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(P, D, Q))]
print(seasonal_pdq)
#For possible combinations of (p,d,q) and (P,D,Q) we loop over and try to observe how the model performs
for param in pdq:
    print(param)
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(df_train_delhi.value_AQI.interpolate(),order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            
            print('SARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
            if(results.aic < min_aic):
                min_aic=results.aic
                min_param=param
                min_seasonal=param_seasonal
            
        except: 
            #print("Exception")
            continue


# In[51]:


#Printing the chosen parameters
print(min_param)
print(min_seasonal)
print(min_aic)


# In[52]:


#Model fitting
mod = sm.tsa.statespace.SARIMAX(df_train_delhi['value_AQI'],
                                order=min_param,
                                seasonal_order=min_seasonal,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[53]:


#Predicting
yhat = results.predict(start=1, end=len(df_test_delhi['value_AQI']))
y_truth = df_test_delhi['value_AQI']
yhat.index=y_truth.index
plt.figure(figsize=(30,10))
plt.plot(yhat, label = "Predicted")
plt.plot(y_truth, label = "Actual")
plt.xlabel('Date')
plt.ylabel('Values aqi')
plt.title('Plot of meantemp for test set')
plt.legend(loc=2)
plt.show()
#Calculating error
mse = ((yhat - y_truth) ** 2).mean()
print('The Mean Squared Error is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))


# The RMSE value obtained for the test set = 234.92 <br>The model does not seem to be doing a good job. <br>This forecast is not  accurate  since the AQI depends on instantaneous variables which cannot be accounted for in a simple SARIMA model.<br> Parameters such as rainfall, temperature, instantaneous weather parameters determine the Air Quality.<br>
# A univariate time series model such as the one we have made will not provide substantial information in determining the air quality in the future.

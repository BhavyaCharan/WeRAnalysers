## Data Analytics Project for the year 2019

This project is an attempt to analyse the air pollution levels in India.<br>
We would like analyse the trends in the concentrations of various pollutants like NO<sub>2</sub>,SO<sub>2</sub>,O<sub>3</sub>,PM<sub>10</sub>,PM<sub>2.5</sub> and CO through the years 2016-2018.<br>
An effort shall also be made to try and attribute the changes in pollutant levels to phenomenon such as Rainfall, temperature, weather etc.<br>
We are working on data collected by openAQ which can be found here https://openaq-data.s3.amazonaws.com/index.html<br>
We have scraped a part of the data for India and our dataset can be found here https://www.kaggle.com/ruben99/air-pollution-dataset-india20162018

## About the Dataset
There are around 63 lakh rows with 11 features as mentioned below<br>
<b>Location</b> : Describes the location where the measurement was made. Varies from locations throughout the country.<br>
<b>City</b> : Specifies the city in which the reading was taken, provides a layer of abstraction as compared to location.<br>
<b>Country</b> : Specifies the country. In our case it's India which is abbreviated as IN.<br>
<b>utc</b> : UTC/GMT timimg for the particular location when the measurement was made.<br>
<b>local</b> : The timimg in the local timezone for the measurement.<br>
<b>parameter</b> : Mentions the pollutant which was measured.<br>
<b>Value</b> : Measured value for pollutant <br>
<b>Unit</b> : Specifies the unit in which the measurement was made <br>
<b>Latitude</b> : Latitude of the corresponding location<br>
<b>Longitude</b> : Longitude of the corresponding location<br>
<b>Attribution</b> : The organisation from which the measurement was obtained<br>

## Overview
<ul>
  <li><b>extract.R </b>: Script for extracting required data from OpenAQ database.</li>
  <li><b>Stocktaking.ipynb</b> : Basic summary statistics of the data. Visualizations capturing some key aspects of the data.</li>
  <li><b>Stocktaking.html</b> : HTML version obtained from nbviewer incase the .ipynb files are not rendered.</li>
  <li><b>WeRAnalysers_LiteratureSurveyReport.pdf</b> : Literature survey report.</li>
  <li><b>WeRAnalysers_FinalReport.pdf</b> : Final report for the project.</li>
  <li><b>AnalysisOfAirPollutantsInIndia.ipynb</b> : Complete code available here</li>
  
</ul>
<h4>How to run the code?</h4>
 The dataset used is avilable on kaggle here https://www.kaggle.com/ruben99/air-pollution-dataset-india20162018, and in order to replicate the tests, the AnalysisOfAirPollutantsInIndia.<br>ipynb notebook can be uploaded onto kaggle for this particular dataset and the tests can be replicated.
<h2>Authors</h2>
 <ul>
  <li><a href="https://github.com/bharaniuk">Bharani Ujjaini Kempaiah</a></li>
  <li><a href="https://github.com/rubenjohn1999">Ruben John Mampilli</a></li>
  <li><a href="https://github.com/BhavyaCharan">Bhavya Charan</a></li>
  </ul>
 

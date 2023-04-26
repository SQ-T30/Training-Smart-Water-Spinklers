# Forecasting Rainfall to Optimise Smart Water Sprinklers

Collaborators
- Aaron Toh Sheng Rong: Problem Motivation, Data Collection, Presentation
- Teo Shao Qi: EDA, Machine Learning

# 1. Problem Motivation

Water is precious and finite, especially in Singapore. With only 1% of its land area used for collecting rainwater, Singapore has to rely heavily on imported water from neighboring countries, such as Malaysia. However, this reliance on imported water has proven to be precarious and unreliable. If Singapore fails to improve water efficiency and diversify its water sources, experts have estimated that it will face a water scarcity challenge by 2030.  

With limited water supply and growing water demand, we must explore every avenue to optimise our water usage. Smart irrigation techniques, such as drip irrigation and weather-based controllers, can significantly reduce water wastage in landscaping and agriculture by adjusting watering schedules based on weather forecasts. One way we can do this is by predicting rainfall and guiding smart water sprinklers to reduce water usage. 

We'll discussing how weather forecasting can be used to help smart water sprinklers conserve water. By analyzing the past rainfall data, we aim to uncover patterns and trends that will help us create a more accurate model for predicting future rainfall. By using predicted rainfall data to guide smart water sprinklers, we can reduce our water usage and conserve this precious resource for future generations.

# 2. Data Collection & Cleaning

We've collected data of concern from the following sites, between the periods of Apr 2013 and Mar 2023

1) Daily Rainfall Total (mm)
2) Mean Temperature (°C)
3) Average Humidity (%)

We then remove unnecessary columns from our csv files, leaving us with the columns of:

![Columns](https://user-images.githubusercontent.com/128040899/233806889-74c37483-f56d-4d33-b9a7-65d157c348c2.jpg)

We then exported our cleaned dataset for analysis.

# 3. Exploratory Data Analysis

3a. Background Knowledge

Singapore's rainfall distribution contains as annual trend, with high amount of rain around the period of 
- Nov to Jan (NE Monsoon)
- May to Jul (SW Monsoon)

3b. Studying Annual Trend

We then analyse our time series data for this trend. First we removed outliers that may affect our analysis. We've identified outliers to be daily rainfall observations exceeding 100mm.

![Outliers Removed](https://user-images.githubusercontent.com/128040899/233784198-590723e9-c618-40e6-93de-c127c01dfb54.jpg)

We saw that our dataset contains fluctuations in daily rainfall that adds no meaning to the annual trend. We considered this fluctuations as "noises", and used a moving average of 30 day window to smooth out this noise and help us analyse our data better.

![Smoothed](https://user-images.githubusercontent.com/128040899/233784210-80f4673c-a69f-4e61-81d8-492ea1a4f784.jpg)

After smoothing our dataset, we find it difficult to analyse the annual trend when data from all observations are clustered together. We used box plots to represent average daily rainfalls across different months of each year, but notice that rainfall distribution behaves differently between each year.

We averaged the records of each months from different years to study the long term pattern of SG's rainfall. From the boxplots, we observed that rainfalls are marginally high between Apr and Jun and notably high between Nov and Dec, which closely resembles to annual trend we saw from our background information. Note that we've removed outliers from the boxplots as they do not add any value to our analysis.

![Average Monthly Rainfall](https://user-images.githubusercontent.com/128040899/233784222-278baddd-19bf-454b-9338-d1eb291b5095.jpg)

3c. Correlation With Other Variables

![CorrMatrix](https://user-images.githubusercontent.com/128040899/233784121-e7bf3238-d098-4599-9f6a-355a8cdde00b.jpg)

We then study the correlationship of other variables (Temperature and Humidity) in our dataset with rainfall, and observed this general behaviour:
- As Mean Temperature ↓, Rainfall ↑
- As Mean Humidity ↑, Rainfall ↑

We then exported our dataset with outliers removed for Machine Learning.

# 4. Machine Learning

4a. Time Series Split

Since we are predicting into the future, our model needs to learn the data's dependence on past observations to give us a better prediction. Hence we cannot split our data into train/test sets randomly like any typical linear regression problems. We've used SkLearn's Time Series Split to fulfill our purpose.

In Time Series Split, our dataset is spliced into N folds. Each fold contains a train dataset of increased time interval from the previous fold, and a test set with fixed time interval (1 Year in our context). Each fold other than the 1st one successively trains the test dataset of the previous fold to learn their dependence of past observations. 

4b. Choice of Machine Learning Model

We noticed from our EDA that although our dataset contains a seasonal trend across each year, its pattern is not entirely consistent. For example, the rainfall is substantially low between Aug to Sep 2022 and high between Oct to Dec 2022, but high between Aug to Sep 2021 and slight lower in Oct to Dec 2021.

Due to these pattern discrepancies that have no linear relationship, we felt that Linear Regression may not give us a good prediction. Therefore, we've decided to employ eXtreme Gradient Boosting (or XGBoost) to tackle this problem. XGBoost works by iteratively adding decision trees (known as boosting rounds) and using gradient boosting to correct errors and learn pattern differences in data. It also contains certain features that control overfitting of our data, such as limiting the depth of a tree and setting early stopping rounds when our model performance does not improve. It is known to be effective when a dataset becomes complex or has nonlinear pattern. 

4c. Smoothing Fluctuations in Train Data

![Noisy](https://user-images.githubusercontent.com/128040899/233784239-abb5ba37-4d13-47a6-a985-354fee9533a9.jpg)

Recall from the EDA section that our dataset is noisy due to daily rainfall fluctuations, and these noises does not add any meaning to our prediction. We've used the same moving average technique from EDA on our train data, to monitor how our model performs throughout the training. We've also tried averaging over windows of 7, 15 and 30 days to see which window gives us the best performance. (As we are studying seasonal trend across different months, averaging beyond 30 days will change the seasonal trend of our dataset)

(Caution: As we are evaluating our test data's performance, we must use actual data for the test set and not smooth them out. Otherwise we would be deceiving ourselves with the results.)

We've trained 4 different versions of dataset and obtained their RMSEs
- Unsmoothed (RMSE: 12.08)
- Smoothed over 7 Days (RMSE: 11.82)
- Smoothed over 15 Days (RMSE: 11.50)
- Smoothed over 30 Days (RMSE: 11.35)

![No Lag Graph B](https://user-images.githubusercontent.com/128040899/233784255-f4729d0d-43b7-4169-9fcb-dfe94049af12.jpg)

Seems like smoothing over 30 days gave us the best performance. We predicted SG's rainfall 1 year ahead from 1 Apr 23 (See Notebook for prediction graphs), but our time series graph does not look anything close to the one from EDA.

4d. Adding Lagged Features

As our rainfall data contains an annual trend, we'll need past values of the same period of prediction (known as lagged features) from our data for the model to capture this seasonal pattern and accurately predict future rainfall.

We began by adding days (D-3, D-7, D-14, D-28) as our lagged features, and obtained the following RMSEs:

- (D-3) Unsmoothed: 12.87, Smoothed 7 days: 12.31, Smoothed 15 days: 11.70, Smoothed 30 days: 11.27
- (D-7) Unsmoothed: 13.03, Smoothed 7 days: 13.39, Smoothed 15 days: 11.71, Smoothed 30 days: 11.45
- (D-14) Unsmoothed: 12.87, Smoothed 7 days: 12.79, Smoothed 15 days: 11.85, Smoothed 30 days: 11.43
- (D-28) Unsmoothed: 12.72, Smoothed 7 days: 12.81, Smoothed 15 days: 12.08, Smoothed 30 days: 11.36

Seemed like D-3 smoothed over 30 days gave us the best performance. We then added months to see if our model performance improves:

- (M-3) Unsmoothed: 12.98, Smoothed 7 Days: 12.24, Smoothed 15 Days: 11.62, Smoothed 30 Days: 11.21
- (M-6) Unsmoothed: 13.01, Smoothed 7 Days: 12.16, Smoothed 15 Days: 11.58, Smoothed 30 Days: 11.18
- (M-9) Unsmoothed: 13.05, Smoothed 7 Days: 12.23, Smoothed 15 Days: 11.52, Smoothed 30 Days: 11.15
- (M-12) Unsmoothed: 13.31, Smoothed 7 Days: 12.19, Smoothed 15 Days: 11.57, Smoothed 30 Days: 11.17

Seemed like M-9 smoothed over 30 days gave us the best performance. We then added years to see if our model performance improves:

- (Y-3) Unsmoothed: 13.18, Smoothed 7 Days: 12.21, Smoothed 15 Days: 11.54, Smoothed 30 Days: 11.16
- (Y-5) Unsmoothed: 13.17, Smoothed 7 Days: 12.19, Smoothed 15 Days: 11.50, Smoothed 30 Days: 11.16
- (Y-7) Unsmoothed: 13.26, Smoothed 7 Days: 12.22, Smoothed 15 Days: 11.53, Smoothed 30 Days: 11.13

Seemed like Y-7 smoothed over 30 days gave us the best performance. Let's see our 1 year prediction graph now.

![Yearly Prediction B](https://user-images.githubusercontent.com/128040899/233784265-df390f9a-9e9a-4e40-9204-1eeb53e6a0c9.jpg)

The graph of the 30 day smoothed window lookws closer to the one from our EDA when we added lagged features, which higher rainfall in the months of Nov to Dec. We then added temperature/humidity/temperature & humidity to see if our model performance improves:

Temperature:

- Unsmoothed: 13.60, Smoothed 7 Days: 11.96, Smoothed 15 Days: 11.57, Smoothed 30 Days: 11.19

Humidity:

- Unsmoothed: 12.78, Smoothed 7 Days: 12.02, Smoothed 15 Days: 11.46, Smoothed 30 Days: 11.20

Temperature & Humidity:

- Unsmoothed: 12.96, Smoothed 7 Days: 11.96, Smoothed 15 Days: 11.47, Smoothed 30 Days: 11.20

We concluded that adding temperatue and humidity did not improve our performance.

The performance of all the 30 day smoothed sets tend to outperform their other counterparts, as indicated by their RMSE and the shape of their 1 year prediction graph (See Notebook for prediction graphs). We concluded that smoothing our train dataset over 30 days gives us the best prediction, as it removes all fluctuations that may affect our prediction within the month.

4e. Testing Our Model on a Hypothetical Smart Water Sprinkler System

Back to our problem motivation, we've created a hypothetical smart water sprinkler system that forecasts water requirements based on crop's requirement and predicted rainfall data.
    
![Forecast](https://user-images.githubusercontent.com/128040899/233784272-3899dbb6-c1ca-43cd-a594-c28dd8bfcca4.jpg)


# 5. Takeaways from This Project

- During data collection, we noticed that Singapore experiences higher annual rainfall compared to countries in the tropical regions. This is because we are situated on the equator and around waters. Hence we expect our model to forecast higher water requirements for crops due to the decreased amount of rainfall in these countries.
- Notice from our EDA that although Temperature and Humidity have correlation with Rainfall, adding them to our model did not improve its performance.  We then learnt that even if the temperature is low, there may be a lack of moisture in our atmosphere, an important ingredient for precipitation. Also, even as humidity is high, it simply means a high amount of moisture in the air, and does not guarantee high rainfall. Precipitation is also affected by convection activity which brings moisture up in the air, and condensation of water vapors forming clouds. These clouds then produce rain when the water droplets become too much for them to hold. 
- Forecasting rainfall can have a wide range of applications beyond just benefitting agriculture. The ability to accurately predict rainfall patterns can aid in the strategic placement of reservoirs and hydroelectric power plants, which can have a significant impact on energy production. Additionally, accurate forecasting can help with flood control efforts by allowing the government to better prepare for potential flooding and mitigate its impacts efficiently. Lastly, it might benefit wildlife conservation efforts, as changes in rainfall patterns can affect natural habitats and ecosystems. Therefore, accurate rainfall forecasting can play an important role in decision-making for a variety of sectors beyond just agriculture.


# 6. References

Meterological Service Singapore. Climate of Singapore
www.weather.gov.sg/climate-climate-of-singapore/

Ryan H.(Year Unknown) Trend
https://www.kaggle.com/code/ryanholbrook/trend

Rob M. (2022) PT2: Time Series Forecasting with XGBoost
https://www.kaggle.com/code/robikscube/pt2-time-series-forecasting-with-xgboost

Ryan H.(Year Unknown) Time Series as Features
https://www.kaggle.com/code/ryanholbrook/time-series-as-features


# Forecasting Rainfall to Optimise Smart Water Sprinklers

Collaborators
- Aaron Toh Sheng Rong: Problem Motivation, Data Collection, Presentation
- Teo Shao Qi: EDA, Machine Learning

# 1. Problem Motivation

Water is a precious and finite resource, especially in Singapore, where the island-city state faces a constant struggle to ensure that its population has access to clean and safe water. With only 1% of its land area used for collecting rainwater, Singapore has to rely heavily on imported water from neighboring countries, such as Malaysia. However, this reliance on imported water has proven to be a precarious and unreliable solution, as it exposes Singapore to potential geopolitical risks and supply disruptions. To make matters worse, Singapore's water demand is expected to double by 2060, putting an enormous strain on its already limited water supply. Furthermore, Singapore experienced its worst drought in 2018, which led to a 30% decrease in local water production, forcing the government to ramp up its water supply from Malaysia. And If Singapore fails to improve water efficiency and diversify its water sources, experts have estimated that it will face a water scarcity challenge by 2030. Moreover, Singapore is also one of the most water-stressed countries globally, and this situation is expected to worsen due to climate change and population growth, according to a report by the World Wildlife Fund (WWF). Furthermore, a water shortage could have devastating consequences for Singapore's economy, which relies heavily on industries that require large amounts of water, such as manufacturing and shipping. Therefore, these data illustrate the pressing need for water conservation in Singapore and the potential consequences if we fail to do so. That's why the topic of conserving water in Singapore is so crucial. NEWater, ultra-clean, high-grade reclaimed water produced from wastewater, currently meets up to 40% of Singapore's water demand, and this is expected to increase to 55% by 2060. Therefore, conserving water in Singapore isn't just a matter of being mindful of our water consumption; it's a matter of survival and securing our future water supply. Every drop counts, and each of us can make a significant difference in preserving this precious resource that sustains us all.

Conserving water in Singapore is critical to ensuring a sustainable water supply for our future generations. With our limited water supply and growing water demand, we must explore every avenue to optimise our water usage. Smart irrigation techniques, such as drip irrigation and weather-based controllers, can significantly reduce water wastage in landscaping and agriculture by adjusting watering schedules based on weather forecasts. One way we can do this is by predicting rainfall and guiding smart water sprinklers to reduce water usage. By using tools such as weather forecasting, soil moisture sensors, evapotranspiration controllers, and rain sensors, we can better predict rainfall and adjust our watering schedules accordingly. This will not only reduce our water consumption but also prevent overwatering and water waste. Therefore, by conserving water and adopting sustainable water-saving habits, we can contribute towards a brighter and more secure water future for Singapore.

We'll discussing how weather forecasting can be used to help smart water sprinklers conserve water. We will be collecting rainfall data from April 2013 to March 2023, to develop a model for predicting and forecasting future rainfall. By analyzing the past rainfall data, we aim to uncover patterns and trends that will help us create a more accurate model for predicting future rainfall. With this information, we can make informed decisions about water usage and conservation, and ensure that we are prepared for any weather changes that may affect our operations. For instance, if the weather forecast predicts rainfall, the sprinklers can be programmed to skip the next watering session. This approach helps to conserve water by ensuring that the plants receive only the water they need. By using weather forecasting to guide smart water sprinklers, we can reduce our water usage and conserve this precious resource for future generations.

# 2. Data Collection & Cleaning

We've collected data of concern from the following sites, between the periods of Apr 2013 and Mar 2023 (10 Years)

- (http://www.weather.gov.sg): 1) Daily Rainfall Total (mm), 2) Mean Temperature (°C)
- (https://www.wunderground.com): 1) Average Humidity (%)

![Info](https://user-images.githubusercontent.com/128040899/233806913-c80c9d1d-2098-4da3-bf35-c050b1531020.jpg)

We then remove unnecessary columns from our csv files, leaving us with the columns of:

![Columns](https://user-images.githubusercontent.com/128040899/233806889-74c37483-f56d-4d33-b9a7-65d157c348c2.jpg)

We also saw that all columns other than Year/Month/Day are numerical columns. We then checked for any null/negative values that may affect our data analysis and machine learning, 

![Cleaning](https://user-images.githubusercontent.com/128040899/233806904-72f0abc9-ae14-47a2-a99e-35126c705ef9.jpg)

We then combined Year/Month/Day columns into 1 date column and set it as our new index, before exporting our cleaned dataset for analysis.

![Index](https://user-images.githubusercontent.com/128040899/233838985-d6e4c909-3bb8-44b2-a96f-a6ab51268e5e.jpg)

# 3. Exploratory Data Analysis

3a. Background Knowledge

Singapore's rainfall distribution contains as annual trend, with high amount of rain around the period of 
- Nov to Jan (NE Monsoon)
- May to Jul (SW Monsoon)

![Monsoon](https://user-images.githubusercontent.com/128040899/233784067-8a0a8dad-642b-4350-b30a-33c763086054.jpg)

3b. Studying Annual Trend

![Outliers](https://user-images.githubusercontent.com/128040899/233784190-92fd6915-3983-4703-a263-64f9c256a045.jpg)

We then analyse our time series data for this trend. First we removed outliers that may affect our analysis. We've identified outliers to be daily rainfall observations exceeding 100mm.

![Outliers Removed](https://user-images.githubusercontent.com/128040899/233784198-590723e9-c618-40e6-93de-c127c01dfb54.jpg)

We saw that our dataset contains fluctuations in daily rainfall that adds no meaning to the annual trend. We considered this fluctuations as "noises", and used a moving average of 30 day window to smooth out this noise and help us analyse our data better.

![Smoothed](https://user-images.githubusercontent.com/128040899/233784210-80f4673c-a69f-4e61-81d8-492ea1a4f784.jpg)

After smoothing our dataset, we find it difficult to analyse the annual trend when data from all observations are clustered together. We used box plots to represent average daily rainfalls across different months of a year. From the boxplots, we observed that rainfalls are marginally high between Apr and Jun and notably high between Nov and Dec, which closely resembles to annual trend we saw from our background information.

![Average Monthly Rainfall](https://user-images.githubusercontent.com/128040899/233784222-278baddd-19bf-454b-9338-d1eb291b5095.jpg)

3c. Correlation With Other Variables

![CorrMatrix](https://user-images.githubusercontent.com/128040899/233784121-e7bf3238-d098-4599-9f6a-355a8cdde00b.jpg)

We then study the correlationship of other variables (Temperature and Humidity) in our dataset with rainfall, and observed this general behaviour:
- As Mean Temperature ↓, Rainfall ↑
- As Mean Humidity ↑, Rainfall ↑

Note that this is not always the case, and we confirm our theory by checking their graphical representation using pairplots. We saw that there may be days when Mean Temperature is low, but Rainfall is low as well. The same can be said for Mean Humidity.

![PairPlot](https://user-images.githubusercontent.com/128040899/233784130-53f2ac1c-3131-428f-8c85-4219b779cce0.jpg)

We then exported our dataset with outliers removed for Machine Learning.

# 4. Machine Learning

4a. Time Series Split

Since we are predicting into the future, our needs to learn the data's dependence on past observations to give us a better prediction. Hence we cannot split our data into train/test sets randomly like any typical linear regression problems. We've used SkLearn's Time Series Split to fulfill our purpose.

![KFoldVSTimeSeriesSplit](https://user-images.githubusercontent.com/128040899/233784143-2e9e1770-f5bb-4309-bf5c-63a8cfe9240b.png)

In Time Series Split, our dataset is spliced into N folds. Each fold contains a train dataset of increased time interval from the previous fold, and a test set with fixed time interval (1 Year in our context). Each fold other than the 1st one successively trains the test dataset of the previous fold to learn their dependence of past observations. Note that this is different from K-Fold Cross Validation, which splits data into K random folds, and uses the (K+1) Fold as the test set on each split. The train set of each fold also has a fixed time interval, unlike the one from Time Series Fold.

After trying different number of folds for training, we've decided that 9 folds gave us the best model performance (RMSE) and most consistent performance across the folds.

4b. Choice of Machine Learning Model

We noticed from our EDA that although our dataset contains a seasonal trend across each year, its pattern is not entirely consistent. For example, the rainfall is substantially low between Aug to Sep 2022 and high between Oct to Dec 2022, but high between Aug to Sep 2021 and slight lower in Oct to Dec 2021.

![Pattern](https://user-images.githubusercontent.com/128040899/233784167-8a306437-62c9-41e6-bd4c-7e5c1191e656.jpg)

Due to these pattern discrepancies that have no linear relationship, we felt that Linear Regression may not give us a good prediction. Therefore, we've decided to employ eXtreme Gradient Boosting (or XGBoost) to tackle this problem. XGBoost works by iteratively adding decision trees (known as boosting rounds) and using gradient boosting to correct errors and learn pattern differences in data. It also contains certain features that control overfitting of our data, such as limiting the depth of a tree and setting early stopping rounds when our model performance does not improve. It is known to be effective when a dataset becomes complex or has nonlinear pattern. 

4c. Smoothing Fluctuations in Train Data

![Noisy](https://user-images.githubusercontent.com/128040899/233784239-abb5ba37-4d13-47a6-a985-354fee9533a9.jpg)

Recall from the EDA section that our dataset is noisy due to daily rainfall fluctuations, and these noises does not add any meaning to our prediction. We've used the same moving average technique from EDA on our train data, to monitor how our model performs throughout the training. We've also tried averaging over windows of 7, 15 and 30 days to see which window gives us the best performance. (As we are studying seasonal trend across different months, averaging beyond 30 days will change the seasonal trend of our dataset)

![Smoothing](https://user-images.githubusercontent.com/128040899/233784244-c20da88b-8ae0-4c3a-878d-085d7ca17ff1.jpg)

(Caution: As we are evaluating our test data's performance, we must use actual data for the test set and not smooth them out. Otherwise we would be deceiving ourselves with the results.)

We've trained 4 different versions of dataset and obtained their RMSEs
- Unsmoothed (RMSE: 12.08)
- Smoothed over 7 Days (RMSE: 11.82)
- Smoothed over 15 Days (RMSE: 11.50)
- Smoothed over 30 Days (RMSE: 11.35)

![No Lag Graph A](https://user-images.githubusercontent.com/128040899/233784252-33fedae7-35a6-47ec-b5c5-a17c57b3bd6a.jpg)
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

![Yearly Prediction A](https://user-images.githubusercontent.com/128040899/233784264-c8680d11-dda3-49e1-8688-d886fd43f6f8.jpg)
![Yearly Prediction B](https://user-images.githubusercontent.com/128040899/233784265-df390f9a-9e9a-4e40-9204-1eeb53e6a0c9.jpg)

The graph of the 30 day smoothed window lookws closer to the one from our EDA when we added lagged features, which higher rainfall in the months of Nov to Dec. We then added temperature to see if our model performance improves:

- Unsmoothed: 13.60, Smoothed 7 Days: 11.96, Smoothed 15 Days: 11.57, Smoothed 30 Days: 11.19

Adding temperature did not improve our performance. Let's see if adding humidity does:

- Unsmoothed: 12.78, Smoothed 7 Days: 12.02, Smoothed 15 Days: 11.46, Smoothed 30 Days: 11.20

Adding humidity did not improve our performance either. Let's see if adding both temperature & humidity does:

- Unsmoothed: 12.96, Smoothed 7 Days: 11.96, Smoothed 15 Days: 11.47, Smoothed 30 Days: 11.20

We concluded that adding temperatue and humidity did not improve our performance.

Note that in all our predictions, the performance of the 30 day smoothed sets tend to outperform their other counterparts, as indicated by their RMSE and the shape of their 1 year prediction graph (See Notebook for prediction graphs). We concluded that smoothing our train dataset over 30 days gives us the best prediction, as it removes all fluctuations that may affect our prediction within the month.

4e. Testing Our Model on a Hypothetical Smart Water Sprinkler System

Back to our problem motivation, we want to input these daily predicted values into our smart water sprinkler system. We've created a hypothetical smart water sprinkler system that has the following inputs and outputs.

Inputs

    Crop's daily irrigation requirements (From Human Input)
    Predicted daily rainfall for 1 year (From our Machine Learning Model)

Outputs

    Residual water requirements adjusted for predicted rainfall.
    
 ![Ask](https://user-images.githubusercontent.com/128040899/233784279-b6433ce5-f5c9-4a9e-837e-bac301c94a5f.jpg)

In our example, we input 10mm as our crop's daily irrigation requirements. The smart sprinkler system forecasted a daily water requirement of ~3.8mm for the 1st week of Apr to supplement daily rainfall (See Notebook for bar charts on other forecasting windows). Through this Hypothetical Sprinkler System, we've discovered that we can save more water if a sprinkler system has informed data on the irrigation requirements for crops, rather than operating solely based on human commands.

![Forecast](https://user-images.githubusercontent.com/128040899/233784272-3899dbb6-c1ca-43cd-a594-c28dd8bfcca4.jpg)

# 5. Area for Improvement

Our forecasted water requirements implied that it will rain everyday within the window of prediction. This may not be the case in reality, and it happened because we did a moving average of 30 days, effectively removing all zero values in our training set. To measure the probability of rain happening, we need other predictors such as clouds and wind movements. However, that will be another type of weather forecasting which is out of our project's scope.

# 6. Takeaways from This Project

- During data collection, we noticed that Singapore experiences higher annual rainfall compared to countries in the tropical regions. This is because we are situated on the equator and around waters. Hence we expect our model to forecast higher water requirements for crops due to the decreased amount of rainfall in these countries.
- Notice from our EDA that although Temperature and Humidity have correlation with Rainfall, adding them to our model did not improve its performance.  We then learnt that even if the temperature is low, there may be a lack of moisture in our atmosphere, an important ingredient for precipitation. Also, even as humidity is high, it simply means a high amount of moisture in the air, and does not guarantee high rainfall. Precipitation is also affected by convection activity which brings moisture up in the air, and condensation of water vapors forming clouds. These clouds then produce rain when the water droplets become too much for them to hold. 
- Forecasting rainfall can have a wide range of applications beyond just benefitting agriculture. The ability to accurately predict rainfall patterns can aid in the strategic placement of reservoirs and hydroelectric power plants, which can have a significant impact on energy production. Additionally, accurate forecasting can help with flood control efforts by allowing the government to better prepare for potential flooding and mitigate its impacts efficiently. Lastly, it might benefit wildlife conservation efforts, as changes in rainfall patterns can affect natural habitats and ecosystems. Therefore, accurate rainfall forecasting can play an important role in decision-making for a variety of sectors beyond just agriculture.


# 7. References

Meterological Service Singapore. Climate of Singapore
www.weather.gov.sg/climate-climate-of-singapore/

Ryan H.(Year Unknown) Trend
https://www.kaggle.com/code/ryanholbrook/trend

Rob M. (2022) PT2: Time Series Forecasting with XGBoost
https://www.kaggle.com/code/robikscube/pt2-time-series-forecasting-with-xgboost

Ryan H.(Year Unknown) Time Series as Features
https://www.kaggle.com/code/ryanholbrook/time-series-as-features


# Data-analysis-on-Rain-in-Australia
Predict tomorrow’s precipitation 


## Introduction
Weather forecasting has always been extremely challenging, given the number of variables involved and the complex interactions between those massive variables. 
However, nowadays thanks to drastic enhancement in computing capabilities and networking technologies, We have a more objective and scientific basis for weather forecasting.  In aspects of our daily lives, It is a most important and interesting issue whether It will rain or not tomorrow. We aim to identify and analyze what features affect tomorrow's precipitation and furthermore, finally we will predict the probability of precipitation of tomorrow by using our prediction model. To establish this prediction model, we plan to use the [Rain in Australia] dataset from kaggle.com. This dataset contains about 10 years of daily weather observations from many locations across Australia. RainTomorrow is the target variable to predict.

## Identify the Dataset
https://www.kaggle.com/jsphyg/weather-dataset-rattle-package?select=weatherAUS.csv

We derived a dataset for our dataset from Kaggle “Rain on Australia.csv”. This dataset contains about 10 years of daily weather observations from many locations across Australia. Observations were drawn from numerous weather stations for a long time. So the size of dataset is enough big to analyze and to acquired feature which affects the tomorrow’s precipitation. 

## Preprocessing
![image](https://user-images.githubusercontent.com/75998991/142978588-fa8b20a3-7329-488d-9146-f9f7310078da.png)

In order to preprocess datasets, divide dataset into two types. “Categorical” and “Numerical”. 
First Categorical data is consist of [Date, Location, Wind Gust Dir, Wind Dir 9am, Wind Dir 3pm, Rain Today, Rain Tomorrow]. 
To check the incorrect values in the data and view them at once, the values in each data were listed and plotted. 
Second Numerical data is consist of [Min Temp, Max Temp, Rainfall, Evaporation, Sunshine, Wind Gust Speed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm].
Similarly, the values of each data were listed and plotted to check the incorrect values in the data and to view them at once.

![image](https://user-images.githubusercontent.com/75998991/142978655-dff08d94-4fd2-40c4-86ef-01a6c474e770.png)

In the “rainfall” columns, we can find “Outlier” caused by typhoons or heavy rain. 
This dataset has few wrong value. But, We could see there were a lot of missing value in the eye.
The target column which we are trying to predict should not have a missing value, so drop the row including missing value in Target column. 
And we dropped a row containing any missing value, and then encoder the categorical data with a Label Encoder and drew a heatmap.

![image](https://user-images.githubusercontent.com/75998991/142978826-2b08afed-869a-40e2-adf9-1ffef349dd42.png)
Check the heatmap to remove unrelated columns and use the most relevant list. 

The reconstructed data group identified a correlation between “rain fall” and "Rain Today". 
If “Rain fall” =< 1, it wouldn’t have rain that day. And if “Rain fall” > 1, it would be rainy day.
On days when both were missing value, they dropped because we can’t estimate rainy day. In addition, rainfall is not common and special circumstances (50+) are dropped as unnecessary outlier.
Cloud column’s missing value fill by median of value and humidity column’s missing value fill by mean of value

## Find Best Model Automatically 
We defined auto function to find best model with computing all combination of parameters that specified scaler, encoder, and algorithms. 
It find the model that scores the best and return it.

In ‘find_best_comb’ function, define 'config' dictionary to set list of scalers, encoders and classifiers. Call 'tuner()' function to find best model. 
In this process, we set each important parameters as 

> * encoders : [OneHotEncoder(), LabelEncoder() ]
> * scalers : [StandardScaler(),MinMaxScaler(),MaxAbsScaler(),RobustScaler()]
> * models : [KNeighborsClassifer() , SVC(), GaussianNB() ]

With these parameters, ‘tuner’ function computes 2(encoders) * 4(scalers) = 8 combinations with for-loop. 
In loop, it calls ‘find_best_model’ function to find best algorithm with each scaled and encoded dataset.  

## Result

The results of all combination of scalers, encoders. And best algorithm for each scaled dataset.
 ![image](https://user-images.githubusercontent.com/75998991/142979332-3630ccaf-722d-47fc-a52c-733bfb808138.png)



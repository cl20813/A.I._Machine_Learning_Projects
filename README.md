# Settings

-[Make and share environment](https://github.com/cl20813/Softwares_Setup/blob/main/install_python.md)        

-[Create an environment on Rutgers Amarel](https://github.com/cl20813/Softwares_Setup/blob/main/amarel_environment.md)
            
-[Install mypackage on Amarel](https://github.com/cl20813/Softwares_Setup/blob/main/install_mypackage_amarel.md)      



## Spatio-Temporal Modeling of Ozone Data Using a Deep Learning Framework (CNN-LSTM) (Updated January 2025)
#### Research Proposal and Exploratory Data Analysis
-[Research Proposal](https://github.com/cl20813/GEMS_TCO/blob/main/GEMS_TCO_EDA/Spatio_temporal_modeling.pdf): The main goal of the project is to develop tools that can help in modeling the spatio-temporal Ozone process.

-[Yearly scale EDA](https://github.com/cl20813/GEMS_TCO/blob/main/GEMS_TCO_EDA/yearly_eda.ipynb): Presented time series of means, variances and semivariograms per hour from January 2023 to December 2024. The plots show not only ```short-term cycles``` but also ```long-term cycles```.

-[Monthly scale EDA ](https://github.com/cl20813/GEMS_TCO/blob/main/GEMS_TCO_EDA/monthly_eda.ipynb): We present time series of semivariograms, and variances. It shows that the process is ```anisotropic``` and this needs to be reflected in the modeling.

-[Hourly scale EDA ](https://github.com/cl20813/GEMS_TCO/blob/main/GEMS_TCO_EDA/hourly_eda.ipynb): We explored data on an hourly scale. ```The cyclic pattern``` shown in the ```semivariograms``` indicates that we should fit the data with a model that can explain this cyclic pattern. Secondly, ```asymmetric cross-variograms``` on some days imply that there is ```space and time interaction```, hence we should consider a non-separable model. Lastly, ```latitude-sliced``` data shows ```spatial non-stationarity```. I plan to ```detrend for 5x10 spatial points``` in the N5N10 E110E120 region. 
#### Models

-[CNN integrated with LSTM](https://github.com/cl20813/GEMS_TCO/blob/main/models/fit_deep_learning.ipynb): Still working on it. It seems CNN does not capture high resolution spatial information. Maybe I should replace it with Gaussian Process followed by LSTM. 

## Exercise : Travelers Insurance Conversion Modeling (Updated Dec.2024)
The goal of the project is to predict the probability that a prospect consumer will choose Travelers as their insurer.

Last AUC score for probability prediction is 0.8015 and recall for convertors (class 1) was 0.77 using the LightGBM.(12-28-2024). 

Tabnet showed good AUC score but it performed very poorly on predicting actual labels. Another observation is that, CNN is good for spatial, sequential data and not a good tool for analyzing tabular, well structured data.

1. Perform base modeling to compare basic models: CNN, LightGBM, and linear models.
2. Conduct feature engineering. -[Feature Engineering*](trav/data_engineering_lightgbm.ipynb)  
3. Optimize hyperparameters using Rutgers HPC computing resources.   
4. Refine feature engineering.  
5. Re-tune the hyperparameters.

The final modeling result is shwon below.                  
-[Final model: LightGBM jupyter notebook*](trav/travelers_lightgbm.ipynb)                             
              
Reference: 
-[LightGBM hyper parameter optimization through Rutgers HPC](trav/amarel/lightgbm_param_opt.txt)                 
-[Neural Network (CNN) and Tabnet](trav/trav_neural_network.ipynb)                                  
-[Deep Learning nn model hyper parameter optimization through Rutgers HPC](trav/amarel/nn_param_opt)                            
            

## Exercise : Spotify genre classification using random forest.
  1. Remove duplicated rows, columns with high-entropy features.
  2. Missing values: use mean for continous and mode for categorical features.
  3. Box cox (postive data only) or yeojohnson transformation to make features more normal.
  4. Remove extreme values or outliers.
  5. Split data into train and test.
  6. Random forest, lowest auc score is 92 for 5 labels in genre.

 -[Note Link](cl20813_SPOTIFY_GENRE.ipynb)


## Exercise . NLP with Python: Spam detection.

  1. Read a text-based dataset into pandas: If I have a raw data, I would have to remove redundant columns other than 'label', and 'message'.
  2. Text pre processing: 1) Remove common words, ('the','a', etc...) by using NLTK library. 2) Remove punctuation (!@#$%).
  3. Use Countervectorizer to convert text into a matrix of token counts.
  4. Exploratory Data Analysis(EDA): Spam messages tend to have lesser characters.
  5. Multinomial Naive Bayes Classifer is suitable for classification with discrete features(word counts), note that it requires **integer** feature counts.
  6. We can examine type 1 error and type 2 error and AUC score.

 -[Note Link](NLP_exercise_scam_detector/NLP_exercise_scam_detector.ipynb)


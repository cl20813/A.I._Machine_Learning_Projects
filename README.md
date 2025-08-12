# IMC Algorithmic Trading Team Competition: Prosperity 3

---

## Top 2.39% Placement as a Solo Player (April 2025)

---

- **[Project Overview](imc/readme.md)**
  - Secured a top 2.39% finish in the IMC Algorithmic Trading competition.
  - Leveraged advanced market-making techniques and financial models.

- **[Trading Strategy](imc/readme.md)**
  - My core strategy was built on an optimized **market-making** algorithm.
  - I used the **Black-Scholes model** and **implied volatility** to inform my decisions.

# Gaussian Process Spatio-Temporal Modeling of Ozone Data Ongoing proejct, updated January 2025)
### Research Proposal and Exploratory Data Analysis
-**[Research Proposal](https://github.com/cl20813/GEMS_TCO/blob/main/GEMS_TCO_EDA/Spatio_temporal_modeling.pdf)**: The main goal of the project is to develop tools that can help in modeling the spatio-temporal Ozone process.      

-**[Yearly scale EDA](https://github.com/cl20813/GEMS_TCO/blob/main/GEMS_TCO_EDA/will_use/yearly_eda.ipynb)**: Presented time series of means, variances and semivariograms per hour from January 2023 to December 2024. The plots show not only ```short-term cycles``` but also ```long-term cycles```.

-**[Monthly scale EDA ](https://github.com/cl20813/GEMS_TCO/blob/main/GEMS_TCO_EDA/will_use/monthly_eda.ipynb)**: We present time series of semivariograms, and variances. It shows that the process is ```anisotropic``` and this needs to be reflected in the modeling.

-**[Hourly scale EDA ](https://github.com/cl20813/GEMS_TCO/blob/main/GEMS_TCO_EDA/will_use/hourly_eda.ipynb)**: We explored data on an hourly scale. ```The cyclic pattern``` shown in the ```semivariograms``` indicates that we should fit the data with a model that can explain this cyclic pattern. Secondly, ```asymmetric cross-variograms``` on some days imply that there is ```space and time interaction```, hence we should consider a non-separable model. Lastly, ```latitude-sliced``` data shows ```spatial non-stationarity```. I plan to ```detrend for 5x10 spatial points``` in the N5N10 E110E120 region. 

# Travelers Insurance Conversion Modeling using LightGBM (Updated Dec.2024)
The goal of the project is to predict the probability that a prospect consumer will choose Travelers as their insurer.

Last AUC score for probability prediction is 0.8015 and recall for convertors (class 1) was 0.77 using the LightGBM.(12-28-2024). 

Tabnet showed good AUC score but it performed very poorly on predicting actual labels. Another observation is that, CNN is good for spatial, sequential data and not a good tool for analyzing tabular, well structured data.

1. Perform base modeling to compare basic models: CNN, LightGBM, and linear models.
2. Conduct feature engineering. -**[Feature Engineering](trav/data_engineering_lightgbm.ipynb)**    # Target Encoding, Missing Imputation, Truncating numerical outliers. 
3. Optimize hyperparameters using Rutgers HPC computing resources to avoid overfitting.              # max_depth, max number of leaves in a tree, max number of data in a leaf, learning rate, feature fraction, etc.
4. Refine feature engineering.  
5. Re-tune the hyperparameters.

The final modeling result is shwon below.                  
-**[Final model: LightGBM jupyter notebook*](trav/travelers_lightgbm.ipynb)**                            
              
Reference:         
-[LightGBM hyper parameter optimization through Rutgers HPC](trav/amarel/lightgbm_param_opt.txt)                    
-[Neural Network (CNN) and Tabnet](trav/trav_neural_network.ipynb)                                      
-[Deep Learning nn model hyper parameter optimization through Rutgers HPC](trav/amarel/nn_param_opt)                                

## Exercise: American Airline stock price prediction (Dec,2024):           
-**[Stock price prediction](American_airline/lstm.ipynb)**: Still working on it. I plan to apply ```lstm``` model.    

            



import pickle
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import lightgbm as lgb


from pathlib import Path
import json
from json import JSONEncoder
import csv
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import typer
import json




app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})
@app.command()


# /opt/anaconda3/envs/faiss_env/bin/python /Users/joonwonlee/Documents/A.I._Machine_Learning_Projects/trav/experiments/hyper_opt.py

def cli(
    v: float = typer.Option(0.5, help="smooth")
) -> None:
    
    class data_setup:
        def __init__(self, df:pd.DataFrame):
            self.df = df
        
        def stratified_train_val_test(self,  stratify_col='convert_ind', test_size=0.2, val_size=0.1, random_state=24):
            # Step 1: Split into train+val and test
            train_val_set, test_set = train_test_split(
                self.df,
                test_size=test_size,
                stratify=self.df[stratify_col],
                random_state=random_state
            )

            # Step 2: Split train_val into train and validation
            # Adjust validation size relative to train_val_set
            val_relative_size = val_size / (1 - test_size)
            train_set, val_set = train_test_split(
                train_val_set,
                test_size=val_relative_size,
                stratify=train_val_set[stratify_col],
                random_state=random_state
            )
            return train_set, val_set, test_set

        def stratified_train_test(self, stratify_col='convert_ind', test_size=0.2, val_size=0.1, random_state=24):  
            # from sklearn.model_selection import train_test_split
            train_set, test_set = train_test_split(self.df, test_size=0.2, stratify=self.df[stratify_col], random_state=24)

            # Separate features and target from the entire training set
            train_y = train_set[stratify_col]
            train_x = train_set.drop(columns=[stratify_col])

            test_y = test_set[stratify_col]
            test_x = test_set.drop(columns=[stratify_col])

            return train_y, train_x, test_y, test_x
        
        

    class Feature_engineering:
        def __init__(self, df:pd.DataFrame):
            self.df = df
    
        def fill_mode(self,col_name):
            tmp = self.df[col_name].mode()
            self.df[col_name] = self.df[col_name].fillna(tmp[0])
        
        def fill_mean(self, col_name):
            tmp = self.df[col_name].mean()
            self.df[col_name] = self.df[col_name].fillna(tmp)

        def fill_median(self, col_name:str):
            tmp = self.df[col_name].median()
            self.df[col_name] = self.df[col_name].fillna(tmp)
        

    class EDA_tools:
        def __init__(self, df:pd.DataFrame):
            self.df = df

        def see_kurtosis(self, col_name:str):
            print(f'Kurtosis of {col_name}: {self.df[col_name].kurtosis()}')

        def see_skewness(self, col_name:str):
            print(f'Skewness of {col_name}: {self.df[col_name].skew()}')
    
    class hyp_optimization:
        def __init__(self, best_params, model_result):
            """
            Initialize the optimization algorithm parameters.

            Parameters:
            - day (int): Day of the optimization.
            - cov_name (str): Name of the covariance model.
            - lat_lon_resolution (List[int]): Resolution for latitude and longitude.
            - lr (float): Learning rate.
            - stepsize (float): Step size for the optimization.
            - params (List[float]): List of parameters for the model.
            - time (float): Time parameter.
            - epoch (int): Number of epochs.
            """
            self.final_auc = final_model.best_score['train']['auc'].round(4)
            self.final_average_precision = model_result['train']['average_precision'].round(4)
            self.objective = best_params['objective']
            self.boosting_type = best_params['boosting_type']
            self.metric = best_params['metric']
            self.num_leaves = best_params['num_leaves']
            self.max_depth = best_params['max_depth']
            self.learning_rate = best_params['learning_rate']
            self.feature_fraction = best_params['feature_fraction']
            self.bagging_fraction = best_params['bagging_fraction']
            self.bagging_freq = best_params['bagging_freq']
            self.min_gain_to_split = best_params['min_gain_to_split']
            
        def toJSON(self) -> str:
            """
            Convert the object to a JSON string.

            Returns:
            - str: JSON representation of the object.
            """
            return json.dumps(self, cls=alg_opt_Encoder, sort_keys=False)

        def save(self, input_filepath: Path, data: Any) -> None:
            """
            Save the aggregated data back to the JSON file.

            Parameters:
            - input_filepath (Path): Path to the JSON file.
            - data (Any): Data to be saved.
            """
            with input_filepath.open('w', encoding='utf-8') as json_file:
                json_file.write(json.dumps(data, separators=(",", ":"), indent=4))


        def to_dict(self) -> Dict[str, Any]:
            return self.__dict__



        def load(self, input_filepath: Path) -> Any:
            """
            Load data from a JSON file.

            Parameters:
            - input_filepath (Path): Path to the JSON file.

            Returns:
            - Any: Loaded data.
            """
            try:
                with input_filepath.open('r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
            except FileNotFoundError:
                loaded_data = []
            return loaded_data
        
        def tocsv(self, jsondata: List[str], fieldnames: List[str], csv_filepath: Path) -> None:
            """
            Convert JSON data to CSV format.

            Parameters:
            - jsondata (List[str]): List of JSON strings.
            - fieldnames (List[str]): List of field names for the CSV.
            - csv_filepath (Path): Path to the CSV file.
            """
            data_dicts = [json.loads(data) for data in jsondata]
            with csv_filepath.open(mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for data in data_dicts:
                    writer.writerow(data)


    class alg_opt_Encoder(JSONEncoder):
        """
        Custom JSON encoder for alg_optimization objects.
        """
        def default(self, obj: Any) -> Dict[str, Any]:
            """
            Override the default method to handle alg_optimization objects.

            Parameters:
            - o (Any): Object to be encoded.

            Returns:
            - Dict[str, Any]: Dictionary representation of the object.
            """
            if isinstance(obj, hyp_optimization):
                return obj.__dict__
            return super().default(obj)  # delegates the serialization process to the standard JSONEncoder

    #df = pd.read_csv("/Users/joonwonlee/Documents/A.I._Machine_Learning_Projects/trav/trav_dataset3.csv")
    df = pd.read_csv("/Users/joonwonlee/Documents/A.I._Machine_Learning_Projects/trav/competitor.csv")

    # Convert object columns to categorical.
    # Note that there is no categorical format in previous data,
    # so some variables are changed to object type.
    categorical_columns = ['credit_score_bin', 'Prior_carrier_grp']
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    data_setup_instance = data_setup(df)
    train_y, train_x, test_y, test_x = data_setup_instance.stratified_train_test()

    best_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': ['auc', 'average_precision'],
        'num_leaves': 15,  # optimal   15>14,20,16, 18
        'learning_rate': 0.009, # 0.009>0.01, 0.008,0.007 >0.005,0.02> 0.003, 0.004, 0.006
        'feature_fraction': 0.8, # 0.8 >0.75 for prob but not for label
        # 0.6> 0.55, 0.65> 0.7 >0.75 
        'bagging_fraction': 0.8,  #0.8>0.6 ,0.7 , 0.85
        'bagging_freq': 5, # 5>4,6
        'random_state': 42,
        'min_data_in_leaf': 20, # 20> 15,18, 22
        'lambda_l1': 1,  # this 1,0.1 is nearly optimal
        'lambda_l2': 0.1,
        'n_jobs': -1,  # Use all CPU cores
        'max_depth': 12,  # Prevent overfitting 6 default  5>6,4   
        # but 4 is better for prob and 5 is better for label prediction
        'min_gain_to_split': 0,  # Minimum gain for splitting
        # 0.005 >0 for label but 0 better for probability prediction
        'max_bin': 255,  # Precision for bins
        'min_sum_hessian_in_leaf': 1e-2,  # Prevent overfitting in leaves 1e-3=1e-2
        
    }


    # Create the dataset
    full_train_dataset = lgb.Dataset(train_x, label=train_y)
    # Train the final model using the best number of boosting rounds
    final_model = lgb.train(
        best_params,
        full_train_dataset,
        num_boost_round= 500,  # 2000 better
        valid_sets=[full_train_dataset],
        valid_names=['train'],
        #callbacks=[lgb.log_evaluation(period=50)]
    )

    base_path = Path('/Users/joonwonlee/Documents/A.I._Machine_Learning_Projects/trav/experiments/')
    input_filepath = base_path / f"testing.json"

    res = hyp_optimization(best_params, final_model.best_score)
    loaded_data = res.load(input_filepath)
    # loaded_data.append( res.toJSON() )
    loaded_data.append(res.to_dict())  # Use to_dict instead of toJSON
    res.save(input_filepath, loaded_data)

    # Sort the list of dictionaries by final_auc
    sorted_list = sorted(loaded_data, key=lambda x: x['final_auc'], reverse=True)
    print(sorted_list[0])


    model_path = base_path / f"outputs/testing.pickle"
    with open (model_path, 'wb') as f:
        pickle.dump(final_model,f)


    with open (model_path, 'rb') as f:
        test = pickle.load(f)


if __name__ == '__main__':
    app()
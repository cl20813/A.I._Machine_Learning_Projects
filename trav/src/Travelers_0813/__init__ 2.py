import os
import pickle
import sys
import pandas as pd
import numpy as np
import torch

from pathlib import Path
import json
from json import JSONEncoder
import csv
from typing import List, Tuple, Dict, Any

gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
sys.path.append(gems_tco_path)

from GEMS_TCO import orderings as _orderings
from GEMS_TCO import configuration as config



class load_data:
    def __init__(self, datapath:str):
        self.datapath = datapath
    
    def read_pickle(self,folder_path:str, filename:str):
        ''' 
        Load pickle data of estimates, then save it into a csv file.
        This is for July 2024.
        Each row contains 7 parameters and loss.
        Returns:
        pd.DataFrame: DataFrame containing the loaded data.

        '''
        input_filepath = os.path.join(folder_path, filename)
        # Load pickle
        with open(input_filepath, 'rb') as pickle_file:
            df_map = pickle.load(pickle_file)
        df = pd.DataFrame()
        for key in df_map:
            tmp = pd.DataFrame(df_map[key][0].reshape(1, -1), columns=['sigmasq', 'range_lat', 'range_lon', 'advec_lat', 'advec_lon', 'beta', 'nugget'])
            tmp['loss'] = df_map[key][1]
            df = pd.concat((df, tmp), axis=0)

        date_range = pd.date_range(start='07-01-24', end='07-31-24')
    
        # Ensure the number of dates matches the number of rows
        if len(date_range) == len(df):
            df.index = date_range
        else:
            print("The number of dates does not match the number of rows in the DataFrame.")

        # Save DataFrame to CSV
        output_filename = 'full_estimates_1250_july24.csv'
        output_csv_path = os.path.join(folder_path, output_filename)
        df.to_csv(output_csv_path, index=False)
        return df

    def load_mm20k_data_bymonthyear(
        self, 
        lat_lon_resolution: List[int] = [10, 10], 
        mm_cond_number: int = 10, 
        years_: List[str] = ['2024'], 
        months_: List[int] = [7, 8]
    ) -> Tuple[Dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
        """
        Load and process data by month and year.

        Parameters:
        - lat_lon_resolution (List[int]): Resolution for latitude and longitude. Default is [10, 10].
        - mm_cond_number (int): Maximum number of nearest neighbors. Default is 10.
        - years_ (List[str]): List of years to process. Default is ['2024'].
        - months_ (List[int]): List of months to process. Default is [7, 8].

        Returns:
        - Tuple[Dict[str, pd.DataFrame], np.ndarray, np.ndarray]: 
            - coarse_dicts: Dictionary of processed dataframes.
            - ord_mm: Array of ordered indices.
            - nns_map: Array of nearest neighbors.
        """
        # Load the dictionary to set spatial coordinates
        filepath = Path(self.datapath) / "pickle_2024/coarse_cen_map24_07.pkl"
        
        with open(filepath, 'rb') as pickle_file:
            coarse_dict_24_1 = pickle.load(pickle_file)

        keys = list(coarse_dict_24_1.keys())

        sample_df = coarse_dict_24_1[keys[0]]
        sample_key = coarse_dict_24_1.get(keys[0])
        if sample_key is None:
            print(f"Key {keys[0]} not found in the dictionary.")

        rho_lat = lat_lon_resolution[0]          
        rho_lon = lat_lon_resolution[1]

        #lat_n = sample_df['Latitude'].unique()[::rho_lat]
        #lon_n = sample_df['Longitude'].unique()[::rho_lon]

        unique_lats = sample_df['Latitude'].unique()
        sorted_lats_descending = np.sort(unique_lats)[::-1] # Sorts ascending, then reverses
        lat_n = sorted_lats_descending[::rho_lat]
        unique_lons = sample_df['Longitude'].unique()
        sorted_lons_descending = np.sort(unique_lons)[::-1] # Sorts ascending, then reverses
        lon_n = sorted_lons_descending[::rho_lon]

        # Set spatial coordinates for each dataset
        coarse_dicts = {}
        years = years_
        for year in years:
            for month in range(months_[0], months_[1]):  # Iterate over all months
                filepath = Path(self.datapath) / f"pickle_{year}/coarse_cen_map{year[2:]}_{month:02d}.pkl"
                with open(filepath, 'rb') as pickle_file:
                    loaded_map = pickle.load(pickle_file)
                    for key in loaded_map:
                        tmp_df = loaded_map[key]
                        coarse_filter = (tmp_df['Latitude'].isin(lat_n)) & (tmp_df['Longitude'].isin(lon_n))
                        coarse_dicts[f"{year}_{month:02d}_{key}"] = tmp_df[coarse_filter].reset_index(drop=True)

        key_idx = sorted(coarse_dicts)
        if not key_idx:
            raise ValueError("coarse_dicts is empty")

        # Extract first hour data because all data shares the same spatial grid
        data_for_coord = coarse_dicts[key_idx[0]]
        x1 = data_for_coord['Longitude'].values
        y1 = data_for_coord['Latitude'].values 
        coords1 = np.stack((x1, y1), axis=-1)

        ord_mm = _orderings.maxmin_cpp(coords1)
        data_for_coord = data_for_coord.iloc[ord_mm].reset_index(drop=True)
        coords1_reordered = np.stack((data_for_coord['Longitude'].values, data_for_coord['Latitude'].values), axis=-1)
        nns_map = _orderings.find_nns_l2(locs=coords1_reordered, max_nn=mm_cond_number)
        return coarse_dicts, ord_mm, nns_map


    def load_working_data_byday(
        self, 
        coarse_dicts: Dict[str, pd.DataFrame],  
        ord_mm: np.ndarray, 
        nns_map: np.ndarray, 
        idx_for_datamap: List[int] = [0, 8]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Load and process working data by day.

        Parameters:
        - coarse_dicts (Dict[str, pd.DataFrame]): Dictionary of processed dataframes.
        - ord_mm (np.ndarray): Array of ordered indices.
        - nns_map (np.ndarray): Array of nearest neighbors.
        - idx_for_datamap (List[int]): Indices for the data map. Default is [0, 8].

        Returns:
        - Tuple[Dict[str, torch.Tensor], torch.Tensor]: 
            - analysis_data_map: Dictionary of tensors for analysis.
            - aggregated_data: Aggregated tensor data.
        """
        key_idx = sorted(coarse_dicts)
        if not key_idx:
            raise ValueError("coarse_dicts is empty")
        
        analysis_data_map = {}
        for i in range(idx_for_datamap[0], idx_for_datamap[1]):
            tmp = coarse_dicts[key_idx[i]].copy()
            tmp['Hours_elapsed'] = np.round(tmp['Hours_elapsed'] - 477700)
            tmp = tmp.iloc[ord_mm, :4].to_numpy()  # reorder the data
            tmp = torch.from_numpy(tmp).double()
            analysis_data_map[key_idx[i]] = tmp

        aggregated_data = pd.DataFrame()
        for i in range(idx_for_datamap[0], idx_for_datamap[1]):
            tmp = coarse_dicts[key_idx[i]].copy()
            tmp['Hours_elapsed'] = np.round(tmp['Hours_elapsed'] - 477700)
            tmp = tmp.iloc[ord_mm].reset_index(drop=True)  
            aggregated_data = pd.concat((aggregated_data, tmp), axis=0)

        aggregated_data = aggregated_data.iloc[:, :4].to_numpy()
        aggregated_data = torch.from_numpy(aggregated_data).double()
        return analysis_data_map, aggregated_data

    ## maybe I should delete reorder_data someday

    def reorder_data(self, analysis_data_map, key_order):
        # key_order = [0, 1, 2, 4, 3, 5, 7, 6]
        keys = list(analysis_data_map.keys())
        reordered_dict = {keys[key]: analysis_data_map[keys[key]] for key in key_order}
        reorder_keys = list(reordered_dict.keys())
        data_frames = []

        for key in reorder_keys:
            tensor_data = reordered_dict[key]
            if isinstance(tensor_data, torch.Tensor):
                tensor_data = tensor_data.numpy()  # Convert tensor to NumPy array
                tensor_df = pd.DataFrame(tensor_data)  # Convert NumPy array to DataFrame
            else:
                tensor_df = tensor_data  # If it's already a DataFrame
            data_frames.append(tensor_df)

        reordered_df = pd.concat(data_frames, axis=0)
        reordered_df = reordered_df.to_numpy()
        reordered_df = torch.from_numpy(reordered_df).double()

        return reordered_dict, reordered_df


    def load_working_data_by_quarterday(
        self, 
        coarse_dicts: Dict[str, pd.DataFrame], 
        ord_mm: np.ndarray, 
        nns_map: np.ndarray, 
        which_group: int, 
        qrt_idx: int, 
        avg_days: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Load and process working data by quarter day.

        Parameters:
        - coarse_dicts (Dict[str, pd.DataFrame]): Dictionary of processed dataframes.
        - ord_mm (np.ndarray): Array of ordered indices.
        - nns_map (np.ndarray): Array of nearest neighbors.
        - which_group (int): Group index to process.
        - qrt_idx (int): Quarter index (1, 2, 3, 4).
        - avg_days (int): Number of days to average.

        Returns:
        - Tuple[Dict[str, torch.Tensor], torch.Tensor]: 
            - analysis_data_map: Dictionary of tensors for analysis.
            - entire_data: Aggregated tensor data.
        """
        keys = sorted(coarse_dicts)
        if not keys:
            raise ValueError("coarse_dicts is empty")
        
        avg_idx = 8 * avg_days
        analysis_data_map = {}
        entire_data = []

        # qrt_idx takes 1, 2, 3, 4 for 4 quarters
        for i in range(which_group - 1, which_group):
            idx_quarter = [[avg_idx * i + 8 * j + 2 * (qrt_idx - 1), avg_idx * i + 8 * j + (2 * qrt_idx - 1)] for j in range(avg_days)]
            idx_quarter = [item for sublist in idx_quarter for item in sublist]

            aggregated_data = []
            for key_idx in idx_quarter:
                tmp = coarse_dicts[keys[key_idx]].copy()
                tmp['Hours_elapsed'] = np.round(tmp['Hours_elapsed'] - 477700)
                tmp['new_key'] = key_idx % 8
                aggregated_data.append(tmp)
                
                tmp = tmp.iloc[ord_mm, [0, 1, 2, 3, 5]].to_numpy()
                tmp = torch.from_numpy(tmp).double()
                analysis_data_map[f'unit_{i}_quarter_{key_idx % 8}'] = tmp

            aggregated_data = pd.concat(aggregated_data, axis=0)
            aggregated_data = aggregated_data[['Latitude', 'Longitude', 'ColumnAmountO3', 'new_key']].groupby(['Latitude', 'Longitude', 'new_key']).mean().reset_index()
            aggregated_data['quarter'] = qrt_idx
        
            aggregated_data = aggregated_data.iloc[:, :5].to_numpy()
            aggregated_data = torch.from_numpy(aggregated_data).double()
            entire_data.append(aggregated_data)

        entire_data = torch.cat(entire_data, dim=0)
        entire_data = entire_data[:, [0, 1, 3, 2]]
        return analysis_data_map, entire_data


class log_semivariograms:
    def __init__(self, deltas: List[Tuple[float, float]], semivariograms, tolerance: float):
        """
        Initialize the log semivariograms parameters.

        Parameters:
        - deltas (List[Tuple[float, float]]): List of deltas for semivariogram calculation.
        - tolerance (float): Tolerance level for semivariogram calculation.
        """
        self.deltas = deltas
        self.tolerance = tolerance

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

class alg_optimization:
    def __init__(self, day: int, cov_name: str, lat_lon_resolution: List[int], lr: float, stepsize: float, params: List[float], time: float, epoch: int):
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
        self.day = day
        self.cov_name = cov_name
        self.lat_lon_resolution = lat_lon_resolution
        self.lr = lr
        self.stepsize = stepsize
        self.sigma = params[0]
        self.range_lat = params[1]
        self.range_lon = params[2]
        self.advec_lat = params[3]
        self.advec_lon = params[4]
        self.beta = params[5]
        self.nugget = params[6]
        self.loss = params[7]
        self.time = time
        self.epoch = epoch

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
    def default(self, o: Any) -> Dict[str, Any]:
        """
        Override the default method to handle alg_optimization objects.

        Parameters:
        - o (Any): Object to be encoded.

        Returns:
        - Dict[str, Any]: Dictionary representation of the object.
        """
        if isinstance(o, alg_optimization):
            return o.__dict__
        return super().default(o)  # delegates the serialization process to the standard JSONEncoder


class likelihood_comparison:
    def __init__(self, day: int, cov_name: str, lat_lon_resolution: List[int], params: List[float], time: float):
        """
        Initialize the likelihood comparison parameters.

        Parameters:
        - day (int): Day of the comparison.
        - cov_name (str): Name of the covariance model.
        - lat_lon_resolution (List[int]): Resolution for latitude and longitude.
        - params (List[float]): List of parameters for the model.
        - time (float): Time parameter.
        """
        self.day = day
        self.cov_name = cov_name
        self.lat_lon_resolution = lat_lon_resolution
        
        self.sigma = params[0]
        self.range_lat = params[1]
        self.range_lon = params[2]
        self.advec_lat = params[3]
        self.advec_lon = params[4]
        self.beta = params[5]
        self.nugget = params[6]
        self.loss = params[7]
        self.time = time
   
    def toJSON(self) -> str:
        """
        Convert the object to a JSON string.

        Returns:
        - str: JSON representation of the object.
        """
        return json.dumps(self, cls=likelihood_comp_Encoder, sort_keys=False)

    def save(self, input_filepath: Path, data: Any) -> None:
        """
        Save the aggregated data back to the JSON file.

        Parameters:
        - input_filepath (Path): Path to the JSON file.
        - data (Any): Data to be saved.
        """
        with input_filepath.open('w', encoding='utf-8') as json_file:
            json_file.write(json.dumps(data, separators=(",", ":"), indent=4))

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

class likelihood_comp_Encoder(JSONEncoder):
    def default(self, o: Any) -> Dict[str, Any]:
        """
        Custom JSON encoder for likelihood_comparison objects.

        Parameters:
        - o (Any): Object to be encoded.

        Returns:
        - Dict[str, Any]: Dictionary representation of the object.
        """
        if isinstance(o, likelihood_comparison):
            return o.__dict__
        return super().default(o)  # delegates the serialization process to the standard JSONEncoder


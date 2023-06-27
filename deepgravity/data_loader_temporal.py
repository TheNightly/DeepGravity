# geoid_o,geoid_d,lng_o,lat_o,lng_d,lat_d,date,visitor_flows,pop_flows

import numpy as np
import pandas as pd
import torch
import glob
import os


def import_all_files(dir: str, reset_date: bool = True) -> pd.DataFrame:
    current_df = None
    for i, file in enumerate(glob.glob(os.path.join(dir, "*.csv"))):
        if current_df is None:
            current_df = pd.read_csv(file)
            current_df['date'].values[:] = 0
        else:
            new_df = pd.read_csv(file)
            new_df['date'].values[:] = i
            current_df =current_df.merge(new_df)
    return current_df

class TemporalFlowDataset(torch.utils.data.Dataset):
    ''' Class for loading sequences as temporal flow datasets- that is, a sequence of flows from one POI to
        another in a time-sequential manner. i.e., an item in this dataloader will have the format
        X = [x_i,j,t , x_i,j,t+1 .... x_i,j,t+n]
        Y = [y_i,j,t , y_i,j,t+1 .... y_i,j,t+n]
        where x_i,j,t denotes the features from point i to j at time t 
        and y_i,j,t represents the true flow from those two points.'''
    def __init__(self, flows_dir, sample_time, time_stride=1, feature_map_dir:str = None):
        self.sample_time = sample_time
        self.time_stride = time_stride
        self.raw_data = import_all_files(flows_dir)
        self.all_ids =  pd.unique(self.raw_data[['geoid_o', 'geoid_d']].values.ravel('K'))
        self.time_length = self.raw_data['date'].max() + 1
        # Time stride == 1 currently. Other sizes not supported
        self.time_samples = (self.time_length - self.sample_time - 1 ) + 1
        num_geo = len(self.all_ids)
        self.geo_samples = num_geo * (num_geo - 1)

        # TODO: Precompute data to reduce load times.

    def __len__(self):
        return self.time_samples * self.geo_samples
    
    def __getitem__(self, idx):
        # decompose into time, geo
        time_idx = idx % self.time_samples
        geo_idx = idx // self.time_samples
        geo_o, geo_d = self.__idx_to_pair(geo_idx)

        result_x = []

        for i in range(self.sample_time):
            result_row = self.raw_data[self.raw_data['geoid_o'] == geo_o & self.raw_data['geoid_d'] == geo_d & self.raw_data['date'] == time_idx + i * self.time_stride]
            if result_row.any():
                result_x.append(result_row['pop_flows'])
            else:
                # Pad with 0 if missing
                result_x.append(0)
        
        result_y = self.raw_data[self.raw_data['geoid_o'] == geo_o & self.raw_data['geoid_d'] == geo_d & self.raw_data['date'] == time_idx + self.sample_time * self.time_stride]

        return result_x, result_y

    def __idx_to_pair(self, geo_idx):
        first_idx = geo_idx // len(self.all_ids)
        second_idx = geo_idx % len(self.all_ids)
        return self.all_ids[first_idx], self.all_ids[second_idx]



class TemporalGraphDataset(torch.utils.data.Dataset):
    ''' Class for loading temporal flow datasets as a temporal graph dataset - that is, a sequence 
        of graph with edge-features corresponding to the flow in a time sequential manner. 
        i.e., an item in this dataloader will have the format
        X = [X_1, X_2, ... X_t, ... X_n]
        Y = [Y_1, Y_2, ... Y_t, ... Y_n]

        where each entry X_t, Y_t is the following format:

        X_t = [x_1,1,t  x_2,1,t ... x_i,1,t ... x_V,1,t] 
              [x_1,2,t  x_2,2,t ...   ...   ...   ...  ] 
              [  ...      ...   ...   ...   ...   ...  ] 
              [x_1,j,t    ...   ... x_i,j,t ...   ...  ] 
              [  ...      ...   ...   ...   ...   ...  ] 
              [x_1,V,t  x_2,V,t ... x_i,V,t ... x_V,V,t] 

        Y_t = [y_1,1,t  y_2,1,t ... y_i,1,t ... y_V,1,t]
              [y_1,2,t  y_2,2,t ...   ...   ...   ...  ]
              [  ...      ...   ...   ...   ...   ...  ]
              [y_1,j,t    ...   ... y_i,j,t ...   ...  ]
              [  ...      ...   ...   ...   ...   ...  ]
              [y_1,V,t  y_2,V,t ... y_i,V,t ... y_V,V,t]

        where V = |G|, the number of vertices in the graph,
        x_i,j,t denotes the features from point i to j at time t,
        and y_i,j,t represents the true flow from those two points.
        '''
    def __init__(self, flows_dir, sample_time, time_stride=1, feature_map_dir:str = None):
        self.sample_time = sample_time
        self.time_stride = time_stride
        self.raw_data = import_all_files(flows_dir)
        self.all_ids =  pd.unique(self.raw_data[['geoid_o', 'geoid_d']].values.ravel('K'))
        self.time_length = self.raw_data['date'].max() + 1
        # Time stride == 1 currently. Other sizes not supported
        self.time_samples_num = (self.time_length - self.sample_time - 1 ) + 1
        
        self.time_slices = []

        for time_step in range(self.time_samples_num):
            current_slice = np.zeros((len(self.all_ids), len(self.all_ids)))
            for i in range(self.all_ids):
                for j in range(i, self.all_ids):
                    current_slice[i,j] = self.raw_data[self.raw_data['geoid_o'] == self.all_ids[i] & self.raw_data['geoid_d'] == self.all_ids[j] 
                                                       & self.raw_data['date'] == time_step]['pop_flows']
            self.time_slices.append(current_slice)
        

    def __len__(self):
        return self.time_samples_num
    
    def __getitem__(self, idx):
        return self.time_slices[idx : idx + self.sample_time]



if __name__ == "__main__":
    path = r"C:\Users\jieru\OneDrive\Documents\2023 UW Winter\DeepGravity\COVID19USFlows-DailyFlows\daily_flows\county2county"


    # geo_samples = 10
    # time_samples = 5

    # for idx in range(geo_samples* time_samples):
    #     time_idx = idx % time_samples
    #     geo_idx = idx // time_samples
    #     print(f"i: {idx} | time {time_idx} | geo {geo_idx}")


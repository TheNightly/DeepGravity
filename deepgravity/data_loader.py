import torch
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
import numpy as np
from importlib.machinery import SourceFileLoader

path = './utils.py'
utils = SourceFileLoader('utils', path).load_module()

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    ids = [item[2] for item in batch]
    #target = torch.LongTensor(target)
    return [data, target], ids


class FlowDataset(torch.utils.data.Dataset):
    def __init__(self,
                 list_IDs: List[str],
                 tileid2oa2features2vals: Dict,
                 o2d2flow: Dict,
                 oa2features: Dict,
                 oa2pop: Dict,
                 oa2centroid: Dict,
                 dim_dests: int,
                 frac_true_dest: float, 
                 model: str
                ) -> None:
        'Initialization'
        self.list_IDs = list_IDs
        self.tileid2oa2features2vals = tileid2oa2features2vals
        self.o2d2flow = o2d2flow
        self.oa2features = oa2features
        self.oa2pop = oa2pop
        self.oa2centroid = oa2centroid
        self.dim_dests = dim_dests
        self.frac_true_dest = frac_true_dest
        self.model = model
        self.oa2tile = {oa:tile for tile,oa2v in tileid2oa2features2vals.items() for oa in oa2v.keys()}
        self.loc_blacklist = set()

        # Remove all ids not in features
        for i in self.list_IDs:
            if i not in self.oa2features:
                self.loc_blacklist.add(i)
                # print(f"Loc {i} feature missing. Added to blacklist.")
        
        for k in self.tileid2oa2features2vals.keys():
            v = tileid2oa2features2vals[k]
            to_remove = []
            for k2 in v.keys():
                if (len(v[k2]) == 0):
                    self.loc_blacklist.add(k2)
                    to_remove.append(k2)
                    # print(f"Loc {k2} idmap feature missing. Added to blacklist.")
            for i in to_remove:
                v.__delitem__(i)
                # print(f"Removing loc {i} from tileid")
                
        for i in self.loc_blacklist:
            # if i in tileid2oa2features2vals.keys():
            #     self.tileid2oa2features2vals.__delitem__(i)
            #     print(f"Removing loc {i} from tileid")
            if i in self.list_IDs:
                self.list_IDs.remove(i)
                # print(f"Removing loc {i} from idlist")
        
    def __len__(self) -> int:
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def get_features(self, oa_origin, oa_destination, df='exponential'):
        oa2features = self.oa2features
        oa2centroid = self.oa2centroid
        oa2pop = self.oa2pop
        dist_od = utils.earth_distance(oa2centroid[oa_origin], oa2centroid[oa_destination])
    
        if self.model in ['G']:
            if df == 'powerlaw':
                return [np.log(oa2features[oa_destination])] + [np.log(dist_od)]
            else:
                return [np.log(oa2features[oa_destination])] + [dist_od]
        elif self.model in ['NG']:
            # print(f"originfeatures {oa_origin} {oa_destination}")
            # print(f"originfeatures len {len(oa2features[oa_origin])} {oa2features[oa_origin]}")
            # print(f"destfeatures len {len(oa2features[oa_destination])} {oa2features[oa_destination]}")
            # print(f"dist len {[dist_od]}")
            if df == 'powerlaw':
                return np.log(oa2features[oa_origin][0]) + np.log(oa2features[oa_destination][0]) + [np.log(dist_od)]
            else:
                return np.log(oa2features[oa_origin][0]) + np.log(oa2features[oa_destination][0]) + [dist_od]
        elif self.model in ['DGsum']:
            return [oa2features[oa_origin][0]] + [np.sum(oa2features[oa_origin][1:])] + [oa2features[oa_destination][0]] + [np.sum(oa2features[oa_destination][1:])] + [dist_od]
            return None
        else:   
            if oa_origin in oa2features:
                origin_features = oa2features[oa_origin]
            else:
                print(f"origin Loc {oa_origin} not found")
                origin_features = [0] * 19 

            if oa_destination in oa2features:
                dest_features = oa2features[oa_destination]
            else:
                print(f"dest Loc {oa_destination} not found")
                dest_features =  [0] * 19 
            # print(f"originfeatures len {len(oa2features[oa_origin])} {oa2features[oa_origin]}")
            # print(f"destfeatures len {len(oa2features[oa_destination])} {oa2features[oa_destination]}")
            # print(f"dist len {[dist_od]}")
            return origin_features + dest_features + [dist_od] #+ [int(oa_origin)] + [int(oa_destination)]
            #return [np.log(oa2pop[oa_origin])] + oa2features[oa_origin] + \
            #       [np.log(oa2pop[oa_destination])] +  oa2features[oa_destination] + [dist_od]


    def get_flow(self, oa_origin, oa_destination):
        o2d2flow = self.o2d2flow
        try:
            # return od2flow[(oa_origin, oa_destination)]
            return o2d2flow[oa_origin][oa_destination]
        except KeyError:
            return 0

    def get_destinations(self, oa, size_train_dest, all_locs_in_train_region):
        o2d2flow = self.o2d2flow
        frac_true_dest = self.frac_true_dest
        try:
            true_dests_all = list(o2d2flow[oa].keys())
        except KeyError:
            true_dests_all = []
        size_true_dests = min(int(size_train_dest * frac_true_dest), len(true_dests_all))
        size_fake_dests = size_train_dest - size_true_dests
        # print(size_train_dest, size_true_dests, size_fake_dests, len(true_dests_all))

        true_dests = np.random.choice(true_dests_all, size=size_true_dests, replace=False)
        fake_dests_all = list(set(all_locs_in_train_region) - set(true_dests))
        fake_dests = np.random.choice(fake_dests_all, size=size_fake_dests, replace=False)

        dests = np.concatenate((true_dests, fake_dests))
        np.random.shuffle(dests)
        return dests

    def get_X_T(self, origin_locs, dest_locs):
        """
        origin_locs  :  list 1 X n_orig, IDs of origin locations
        dest_locs  :  list n_orig X n_dest, for each origin, IDs of destination locations
        """
        o2d2flow = self.o2d2flow
        oa2features = self.oa2features
        oa2centroid = self.oa2centroid
        # print(origin_locs)
        # print(dest_locs)
        X, T = [], []
        for en, i in enumerate(origin_locs):
            X += [[]]
            T += [[]]
            for j in dest_locs[en]:
                if j in oa2features:
                    X[-1] += [self.get_features(i, j)]
                    T[-1] += [self.get_flow(i, j)]
            # print(X)

        teX = torch.from_numpy(np.array(X)).float()
        teT = torch.from_numpy(np.array(T)).float()
        return teX, teT

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        'Generates one sample of data (one location)'

        tileid2oa2features2vals = self.tileid2oa2features2vals
        o2d2flow = self.o2d2flow
        oa2features = self.oa2features
        oa2centroid = self.oa2centroid
        dim_dests = self.dim_dests
        frac_true_dest = self.frac_true_dest
        oa2tile = self.oa2tile

        # Select sample (tile)
        sampled_origins = [self.list_IDs[index]]
        tile_ID = oa2tile[sampled_origins[0]]

        #print('tile_ID: %s'%tile_ID)

        # Load data and get flows

        # Select a subset of OD pairs
        all_locs_in_train_region = list(tileid2oa2features2vals[tile_ID].keys())
        size_train_dest = min(dim_dests, len(all_locs_in_train_region))
        sampled_dests = [self.get_destinations(oa, size_train_dest, all_locs_in_train_region)
                         for oa in sampled_origins]
        
        # print(sampled_origins)
        

        # get the features and flows
        sampled_trX, sampled_trT = self.get_X_T(sampled_origins, sampled_dests)
        # print (f"trx {sampled_trX}")
        # print(sampled_trT)
        # print(sampled_origins)
        int_dests = []
        for dest in sampled_dests:
            int_dests.append(dest.astype(np.int64))
        

        return sampled_trX, sampled_trT, sampled_origins, int_dests


    def __getitem_tile__(self, index: int) -> Tuple[Any, Any]:
        'Generates one sample of data (one tile)'

        tileid2oa2features2vals = self.tileid2oa2features2vals
        o2d2flow = self.o2d2flow
        oa2features = self.oa2features
        oa2centroid = self.oa2centroid
        dim_dests = self.dim_dests
        frac_true_dest = self.frac_true_dest

        # Select sample (tile)
        tile_ID = self.list_IDs[index]
        #print('tile_ID: %s'%tile_ID)

        # Load data and get flows

        # get all the locations in tile_ID
        sampled_origins = list(tileid2oa2features2vals[tile_ID].keys())

        # Select a subset of OD pairs
        train_locs = sampled_origins
        all_locs_in_train_region = train_locs
        size_train_dest = min(dim_dests, len(all_locs_in_train_region))
        sampled_dests = [self.get_destinations(oa, size_train_dest, all_locs_in_train_region)
                         for oa in sampled_origins]

        # get the features and flows
        sampled_trX, sampled_trT = self.get_X_T(sampled_origins, sampled_dests)

        return sampled_trX, sampled_trT


import os
print(os.getcwd())
import sys
sys.path.append(os.getcwd())
print(sys.path)

# TODO: Set to LOBCAST, potentially change paths to make LOBCAST a sbubmodule of LOBBench
import LOBCAST_Clean

from torch.utils import data
import torch.nn.functional as F
import numpy as np
import torch
import pandas as pd
import src.constants as cst
import collections
import LOBCAST_Clean.src.data_preprocessing.preprocessing_utils as ppu

from LOBCAST_Clean.src.data_preprocessing.LOB.LOBSTERDataBuilder import LOBSTERDataBuilder
from LOBCAST_Clean.src.config import Configuration as OldConfiguration
from LOBCAST_Clean.src.utils.utilis_lobster_datasource import message_columns,orderbook_columns

from datetime import datetime, timedelta
import tqdm
import re

LOSS_WEIGHTS_DICT = {m: 1e6 for m in cst.Models}


class Configuration(OldConfiguration):
    def __init__(self, run_name_prefix=None):
        super().__init__(run_name_prefix)
        self.LOBBENCH_MODEL = cst.LBModels.LARGESAMPLE
        self.TrainRealPercent = cst.DEFAULT_REAL_PERCENT



class LOBBenchDataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """

    def __init__(
            self,
            config: Configuration,
            dataset_type :str , 
            stocks_list :list[str],
            start_end_trading_day: tuple[str],
            vol_price_mu=None, #for norm calculated from training set.
            vol_price_sig=None,
            num_classes=cst.NUM_CLASSES
    ):
        self.config = config
        self.start_end_trading_day = start_end_trading_day
        self.num_classes = num_classes

        self.vol_price_mu, self.vol_price_sig = vol_price_mu, vol_price_sig
        self.sample_size = self.config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_SNAPSHOTS]  # 100, samples looking back. 

        map_stock_databuilder = dict() #one databuilder per stock. 

        _local_stocks_list = stocks_list.copy()

        # Choose the stock names to open to build the specific dataset.
        # No need to open all for test set, because mu/sig are pre-computed when prev opened train and dev
        stocks_open = None
        if dataset_type == cst.DatasetType.TRAIN:
            # we open also the TEST stock(s) to determine mu and sigma for normalization, needed for all
            stocks_open = list(set(config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].value + config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].value))  # = [LYFT, NVDA]

        elif dataset_type == cst.DatasetType.VALIDATION:
            stocks_open = config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].value  # = [LYFT]

        elif dataset_type == cst.DatasetType.TEST:
            stocks_open = config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].value   # = [NVDA]

        for stock in stocks_open:
            if dataset_type != cst.DatasetType.TRAIN:
                paths_list = [cst.DATASET_LOBBENCH +f"{config.LOBBENCH_MODEL}/{stock}/{cst.RealGenType.REAL}"]   
            elif config.TrainRealPercent == 0: #Use only generated data for training.
                paths_list = [cst.DATASET_LOBBENCH +f"{config.LOBBENCH_MODEL}/{stock}/{cst.RealGenType.GEN}"]
            elif config.TrainRealPercent == 1: #Use only real data for training.
                paths_list = [cst.DATASET_LOBBENCH +f"{config.LOBBENCH_MODEL}/{stock}/{cst.RealGenType.REAL}"]
            else:
                paths_list = [cst.DATASET_LOBBENCH +f"{config.LOBBENCH_MODEL}/{stock}/{cst.RealGenType.REAL}",
                              cst.DATASET_LOBBENCH +f"{config.LOBBENCH_MODEL}/{stock}/{cst.RealGenType.GEN}"]
                if stock in _local_stocks_list:
                    _local_stocks_list.append(stock+"2")
            for i,path in enumerate(paths_list):
                databuilder = LOBSTERDataBuilder(
                    stock,
                    path,
                    config=config,
                    n_lob_levels=cst.N_LOB_LEVELS,
                    dataset_type=dataset_type,
                    start_end_trading_day=start_end_trading_day,
                    crop_trading_day_by=0,
                    window_size_forward=config.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW],
                    window_size_backward=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW],
                    # normalization_mean=normalization_mean,
                    # normalization_std=normalization_std,
                    num_snapshots=self.sample_size,
                    label_dynamic_scaler=config.HYPER_PARAMETERS[cst.LearningHyperParameter.LABELING_SIGMA_SCALER],
                    is_data_preload=False, #CHANGED
                    data_granularity=cst.Granularity.Events1 #CHANGED
                )
                
                if i==0:
                    map_stock_databuilder[stock] = databuilder  # STOCK: databuilder
                elif i==1:
                    map_stock_databuilder[stock+"2"] = databuilder
                else:
                    raise ValueError(f"Unknown path {path} for stock {stock}.")

            # normalization_mean = self.vol_price_mu[stock] if stock in self.vol_price_mu else None
            # normalization_std = self.vol_price_sig[stock] if stock in self.vol_price_sig else None

            # print()
            # print(dataset_type, stocks_list, stock, start_end_trading_day, path, sep="\n")
            # print()




            # self.vol_price_mu[stock], self.vol_price_sig[stock] = databuilder.normalization_means, databuilder.normalization_stds
            
        # print('vol_price_mu:', self.vol_price_mu)
        # print('vol_price_sig:', self.vol_price_sig)

        def to_ohlc(df):
            mid = (df.loc[:, 'psell1'] + df.loc[:, 'pbuy1']) / 2
            mid = pd.DataFrame(mid, columns=["mid"])
            mid["dummy"] = np.arange(len(mid))
            mid["dummy"] = pd.to_datetime(mid["dummy"], unit="s")
            mid = mid.set_index("dummy")

            df_10s = mid.resample('10S')  # eventi
            OHLC = df_10s.agg({'mid': ['min', 'max', 'first', 'last']})
            OHLC.columns = ['low', 'high', 'open', 'close']
            OHLC = OHLC.reset_index(drop=True)
            # print(OHLC.shape)
            # print(OHLC.describe())
            return OHLC
        


        Xs, Ys, Ss, ignore_indices_len, index_array = list(), list(), list(), [0], list()
        for stock in _local_stocks_list:
            # print("Handling", stock, "for dataset", dataset_type)
            databuilder = map_stock_databuilder[stock]

            data_x, data_y,seq_lens = databuilder.get_X_nx40(), databuilder.get_Y_n(), databuilder.get_sequence_lengths()
            # data_x_umb = databuilder.get_Xung_nx40()
            index_array.append(self.get_valid_sample_indices(seq_lens, self.sample_size))
            Xs.append(data_x)
            Ys.append(data_y)
            Ss.extend([stock]*len(data_y))
            ignore_indices_len.append(len(data_y))

            # print(dataset_type, stock, data_x_umb.shape, data_x.shape)
            # OHLC = to_ohlc(data_x_umb)
            # OHLC = OHLC.iloc[:data_x.shape[0], :]
            # print("saving", OHLC.shape)
            # write_data(OHLC, "data/", "OHLC_{}.data".format(stock))

        index_array= self.merge_valid_indices(index_array, self.sample_size)

        self.index_array= index_array
        # removes the indices that are the first sample_size
        ignore_indices = []
        ind_sf = 0
        for iv in range(len(ignore_indices_len)-1):
            p = ind_sf + ignore_indices_len[iv]
            ignore_indices += list(range(p, p+self.sample_size))  # [range(0, 100), range(1223, 1323), ]
            ind_sf = p

        # X and Y ready to go
        self.x=pd.concat(Xs,axis=0)


        self.x, self.vol_price_mu, self.vol_price_sig = self.__stationary_normalize_data(self.x, self.vol_price_mu, self.vol_price_sig)


        

        if config.CHOSEN_MODEL in [cst.Models.S5BOOK,
                                       cst.Models.S5MSGS,
                                       cst.Models.S5MSGSBOOK]:
            self.x=np.array(self.x.values)
        else:
            self.x = torch.from_numpy(self.x.values).type(torch.FloatTensor)

        y = np.concatenate(Ys, axis=0).astype(float)

        if config.CHOSEN_MODEL == cst.Models.DEEPLOBATT:
            self.ys_occurrences = dict()
            for i, window in enumerate(cst.FI_Horizons):
                if window.value is not None:
                    y_i = y[:, i]
                    self.ys_occurrences[window] = collections.Counter(y_i)
        else:
            self.ys_occurrences = collections.Counter(y)
            occs = np.array([self.ys_occurrences[c] for c in sorted(self.ys_occurrences)])
            self.loss_weights = torch.Tensor(LOSS_WEIGHTS_DICT[config.CHOSEN_MODEL] / occs)

        if config.CHOSEN_MODEL in [cst.Models.S5BOOK,
                                       cst.Models.S5MSGS,
                                       cst.Models.S5MSGSBOOK]:
            self.y=np.array(y)
        else:
            self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.stock_sym_name = Ss

        # self.indexes_chosen = self.__under_sampling(self.y, ignore_indices)
        self.x_shape = (self.sample_size, self.x.shape[1])

    def __len__(self):
        """ Denotes the total number of samples. """
        # len(self.indexes_chosen)
        return len(self.index_array)

    def __getitem__(self, index):
        """ Generates samples of data. """
        index=self.index_array[index]
        x = self.x[index: index + self.sample_size]
        y = self.y[index + self.sample_size - 1]
        s = self.stock_sym_name[index]
        return x, y, s
        # id_sample = self.indexes_chosen[index]
        # x, y, s = self.x[id_sample-self.sample_size:id_sample], self.y[id_sample], self.stock_sym_name[id_sample]

    def __under_sampling(self, y, ignore_indices):
        """ Discard instances of the majority class. """
        print("Doing under-sampling...")

        y_without_snap = [y[i] for i in range(len(y)) if i not in ignore_indices]  # removes the indices of the first sample for each stock

        occurrences = self.compute_occurrences(y_without_snap)
        i_min_occ = min(occurrences, key=occurrences.get)  # index of the class with the least instances
        n_min_occ = occurrences[i_min_occ]                 # number of occurrences of the minority class

        # if we balance the classes, loss_weights is not useuful anymore
        # occs = np.array([occurrences[k] for k in sorted(occurrences)])
        # self.loss_weights = torch.Tensor(occs / np.sum(occs))

        indexes_ignore = set(ignore_indices)
        indexes_chosen = []
        for i in [cst.Predictions.UPWARD.value, cst.Predictions.STATIONARY.value, cst.Predictions.DOWNWARD.value]:
            indexes = np.where(y == i)[0]
            indexes = np.array(list(set(indexes) - indexes_ignore))  # the indices of the first sample for each stock

            assert len(indexes) >= self.config.INSTANCES_LOWER_BOUND, "The instance is not well formed, there are less than {} instances for the class {} ({}).".format(self.config.INSTANCES_LOWER_BOUND, i, len(indexes))
            indexes_chosen += list(self.config.RANDOM_GEN_DATASET.choice(indexes, n_min_occ, replace=False))

        indexes_chosen = np.sort(indexes_chosen)
        return indexes_chosen

    def __stationary_normalize_data(self, data, normalization_mean=None, normalization_std=None):
        """ DONE: remember to use the mean/std of the training set, to z-normalize the test set. """

        col_choice = {"volumes": ppu.get_volume_column_name(data.columns),
                      "prices":  ppu.get_price_column_name(data.columns)}

        # print("Normalization... (using means", normalization_mean, "and stds", normalization_std, ")")

        means_dict, stds_dict = dict(), dict()
        for col_name in col_choice:
            cols = col_choice[col_name]

            if normalization_mean is None and normalization_std is None:
                means_dict[col_name] = data.loc[:, cols].stack().mean()
                stds_dict[col_name] = data.loc[:, cols].stack().std()

            elif normalization_mean is not None and normalization_std is not None:
                means_dict[col_name] = normalization_mean[col_name]
                stds_dict[col_name] = normalization_std[col_name]

            data.loc[:, cols] = (data.loc[:, cols] - means_dict[col_name]) / stds_dict[col_name]
            data.loc[:, cols] = data.loc[:, cols]

            # TODO: volumes and prices can be negative, add min value
            # + abs(data.loc[:, cols].stack().min())  # scale positive
            # data.loc[:, cols].stack().plot.hist(bins=200, alpha=0.5, title=col_name)
            # plt.show()
        # data = data.fillna(method="bfill")
        # data = data.fillna(method="ffill")

        return data, means_dict, stds_dict

    def get_valid_sample_indices(self,df_lengths: list[int], window_length: int) -> list[int]:
        """
        Given a list of dataframe lengths and a window length, return indices that can be 
        accessed in a concatenated dataframe such that the window [index, index+window_length) 
        stays within a single original dataframe.
        
        Args:
            df_lengths: List of lengths of individual dataframes before concatenation
            window_length: Size of the window (index to index+window_length)
        
        Returns:
            List of valid indices in the concatenated dataframe
        """
        valid_indices = []
        current_pos = 0
        
        for df_len in df_lengths:
            # For each dataframe, valid indices are those where:
            # index + window_length <= current_pos + df_len
            # which means: index <= current_pos + df_len - window_length
            
            max_valid_index = current_pos + df_len - window_length
            
            # Add all valid indices for this dataframe
            for idx in range(current_pos, max_valid_index + 1):
                valid_indices.append(idx)
            
            current_pos += df_len
        
        return valid_indices
    

    def merge_valid_indices(self,indices_list: list[list[int]], offset: int) -> list[int]:
        """
        Merge multiple lists of valid window indices with a single offset applied cumulatively.
        
        Args:
            indices_list: List of valid indices arrays
            offset: Scalar offset (window size) applied cumulatively between each list
        
        Returns:
            Merged list of valid indices
        """
        merged = []
        current_offset = 0
        
        for indices in indices_list:
            merged.extend([idx + current_offset for idx in indices])
            if indices:  # Only add offset if list is not empty
                current_offset = indices[-1] + current_offset + offset
        
        return sorted(merged)



def read_sub_routine(file_7z: str, first_date: str = "1990-01-01",
                    last_date: str = "2100-01-01",
                    type_file: str = "orderbook",
                    level: int = 10,
                    path: str = "") -> dict:
    """
        :param file_7z: the input file where the csv with old_data are stored
        :param first_date: the first day to load from the input file
        :param last_date: the last day to load from the input file
        :param type_file: the kind of old_data to read. type_file in ("orderbook", "message")
        :param level: the LOB level of the orderbook
        :param path: data path
        :return: a dictionary with {day : dataframe}
    """
    assert type_file in ("orderbook", "message"), "The input type_file: {} is not valid".format(type_file)

    columns = message_columns() if type_file == "message" else orderbook_columns(level)
    # if both none then we automatically detect the dates from the files
    first_date = datetime.strptime(first_date, "%Y-%m-%d")
    last_date = datetime.strptime(last_date, "%Y-%m-%d")

    all_period = {}  # day :  df

    path = path + file_7z
    print("Reading all", type_file, "files...")
    for file in tqdm.tqdm(sorted(os.listdir(path))):
        # print("processed file", path, file)
        # read only the selected type of file
        if type_file not in str(file):
            continue

        # read only the old_data between first_ and last_ input dates
        m = re.search(r".*([0-9]{4}-[0-9]{2}-[0-9]{2}).*", str(file))
        id=re.search(r"real_id_([0-9]+)", str(file))
        if m:
            entry_date = datetime.strptime(m.group(1), "%Y-%m-%d")
            if entry_date < first_date or entry_date > last_date:
                continue
            if id:
                idnum=int(id.group(1))
                entry_date=entry_date + timedelta(milliseconds=idnum)
        else:
            print("error for file: {}".format(file))
            continue

        curr = path + '/' + file

        # inferring type has a high memory usage low_memory can't be true as default
        
        df = pd.read_csv(curr, sep=",", header=None, low_memory=False)
        # Adjust columns if there are fewer or more than expected
        if df.shape[1] < len(columns):
            # Add missing columns as empty
            for i in range(df.shape[1], len(columns)):
                df[i] = np.nan
        elif df.shape[1] > len(columns):
            # Truncate extra columns
            df = df.iloc[:, :len(columns)]
        df.columns = columns
        # df.columns = columns[:len(df.columns)]
        # df.drop(df.columns.difference(columns), inplace=True)
        all_period[entry_date] = df

    return all_period
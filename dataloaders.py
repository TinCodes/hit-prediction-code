"""Dataloaders for the hit song prediction."""
import json

from dbispipeline.base import Loader

import numpy as np

import pandas as pd

from common import feature_columns


class MsdBbLoader(Loader):
    """Million song dataset / billboard charts loaer."""

    def __init__(self,
                 hits_file_path,
                 non_hits_file_path,
                 features_path,
                 non_hits_per_hit=None,
                 features=None,
                 label=None,
                 nan_value=0,
                 random_state=None):
        self._config = {
            'hits_file_path': hits_file_path,
            'non_hits_file_path': non_hits_file_path,
            'features_path': features_path,
            'non_hits_per_hit': non_hits_per_hit,
            'features': features,
            'label': label,
        }

        # Get csv data about hits and non-hits
        hits = pd.read_csv(hits_file_path)
        non_hits = pd.read_csv(non_hits_file_path)

        # Ratio of hits to non hits to use for sample size and non-hits
        # Downsampling done here
        if non_hits_per_hit:
            num_of_samples = len(hits) * non_hits_per_hit
            non_hits = non_hits.sample(
                n=num_of_samples, random_state=random_state)

        # Build the data to use (not yet processed)
        # hits and non-hits are appended first and then used to define data
        # .h5 files (low and high level features) are merged to the data
        # keymapping using one hot encoding for  
        data = hits.append(non_hits, sort=False, ignore_index=True)
        ll_features = pd.read_hdf(features_path + '/msd_bb_ll_features.h5')
        data = data.merge(ll_features, on='msd_id')
        hl_features = pd.read_hdf(features_path + '/msd_bb_hl_features.h5')
        data = data.merge(hl_features, on='msd_id')
        data = key_mapping(data)

        # Generate labels for the dataloader
        # ravel == flatten on an array the column with the label used for the constructor
        self.labels = np.ravel(data[[label]])
        # isnan() returns same array, True or False values (check if its NaN)
        nan_values = np.isnan(self.labels)
        # While using a boolean array as index for another array
        # only True values will be used.
        self.labels[nan_values] = nan_value

        # List data columns, remove label column from list
        # Delete column with label. => new data variable
        non_label_columns = list(data.columns)
        non_label_columns.remove(label)
        data = data[non_label_columns]

        feature_cols = []
        self._features_list = []
        for feature in features:
            cols, part = feature_columns(data.columns, feature)
            feature_cols += cols
            self._features_list.append((cols, part))

        self.data = data[feature_cols]
        self._features_index_list = []
        for cols, part in self._features_list:
            index = [self.data.columns.get_loc(c) for c in cols]
            self._features_index_list.append((index, part))

        self._config['features_list'] = self._features_list
        print(self._features_list)

    def load(self):
        return self.data.values, self.labels

    @property
    def feature_indices(self):
        return self._features_index_list

    @property
    def configuration(self):
        return self._config


def _get_highlevel_feature(features_path, msd_id):
    file_suffix = '.mp3.highlevel.json'
    return _load_feature(features_path, msd_id, file_suffix)


def _get_lowlevel_feature(features_path, msd_id):
    file_suffix = '.mp3'
    return _load_feature(features_path, msd_id, file_suffix)


def _load_feature(features_path, msd_id, file_suffix):
    file_prefix = '/features_tracks_' + msd_id[2].lower() + '/'
    file_name = features_path + file_prefix + msd_id + file_suffix

    with open(file_name) as features:
        return json.load(features)


def key_mapping(df):
    # https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
    string_columns = [
        'tonal.chords_key', 'tonal.chords_scale', 'tonal.key_scale',
        'tonal.key_key'
    ]

    for c in string_columns:
        # list() on a Dataframe returns a list of the DataFrame's columns  
        if c in list(df):
            # Generate numerical values for qualitative values (string_columns)
            # returns a DataFrame with columns: "c" + "uniqueValue"
            # and values 0 to 1 depending on the position values appear. 
            dummies = pd.get_dummies(df[c], prefix=c, drop_first=False)

            # Concat this new columns with the values to original dataset
            df = pd.concat([df.drop(c, axis=1), dummies], axis=1)
    return df

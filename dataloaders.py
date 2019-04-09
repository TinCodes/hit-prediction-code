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

        hits = pd.read_csv(hits_file_path)
        non_hits = pd.read_csv(non_hits_file_path)

        if non_hits_per_hit:
            num_of_samples = len(hits) * non_hits_per_hit
            non_hits = non_hits.sample(
                n=num_of_samples, random_state=random_state)

        data = hits.append(non_hits, sort=False, ignore_index=True)
        ll_features = pd.read_hdf(features_path + '/msd_bb_ll_features.h5')
        data = data.merge(ll_features, on='msd_id')
        hl_features = pd.read_hdf(features_path + '/msd_bb_hl_features.h5')
        data = data.merge(hl_features, on='msd_id')
        data = data.replace(to_replace=key_mapping())

        self.labels = np.ravel(data[[label]])
        nan_values = np.isnan(self.labels)
        self.labels[nan_values] = nan_value

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


def key_mapping():
    return {
        'tonal.chords_key': {
            'C': 1,
            'Em': 2,
            'G': 3,
            'Bm': 4,
            'D': 5,
            'F#m': 6,
            'A': 7,
            'C#m': 8,
            'E': 9,
            'G#m': 10,
            'B': 11,
            'D#m': 12,
            'F#': 13,
            'A#m': 14,
            'C#': 15,
            'Fm': 16,
            'G#': 17,
            'Cm': 18,
            'D#': 19,
            'Gm': 20,
            'A#': 21,
            'Dm': 22,
            'F': 23,
            'Am': 24,
        },
        'tonal.chords_scale': {
            'minor': 0,
            'major': 1,
        },
        'tonal.key_key': {
            'A': 1,
            'A#': 2,
            'B': 3,
            'C': 4,
            'C#': 5,
            'D': 6,
            'D#': 7,
            'E': 8,
            'F': 9,
            'F#': 10,
            'G': 11,
            'G#': 12,
        },
        'tonal.key_scale': {
            'minor': 0,
            'major': 1,
        },
    }

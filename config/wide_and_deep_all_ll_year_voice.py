import dbispipeline.result_handlers as result_handlers
from dbispipeline.evaluators import GridEvaluator

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import common

from dataloaders import MsdBbLoader

import evaluations

from models.wide_and_deep import WideAndDeep

dataloader = MsdBbLoader(
    hits_file_path='/storage/nas3/datasets/music/billboard/msd_bb_matches.csv',
    non_hits_file_path=
    '/storage/nas3/datasets/music/billboard/msd_bb_non_matches.csv',
    features_path='/storage/nas3/datasets/music/billboard',
    non_hits_per_hit=1,
    features=[
        *common.all_ll_year_voice_list(),
    ],
    label='peak',
    nan_value=150,
    random_state=42,
)

pipeline = Pipeline([
    ('scale', MinMaxScaler()),
    ('wide_and_deep', WideAndDeep(features=dataloader.feature_indices)),
])

evaluator = GridEvaluator(
    parameters={
        'wide_and_deep__epochs': [10, 50, 100],
        'wide_and_deep__batch_normalization': [False, True],
        'wide_and_deep__dropout_rate': [None],
    },
    grid_parameters=evaluations.grid_parameters(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]

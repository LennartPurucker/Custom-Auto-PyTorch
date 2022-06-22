from typing import Dict
from autoPyTorch.constants import BINARY, MULTICLASS, CONTINUOUS
from enum import Enum


DEFAULT_METRIC_INDEX = 0


def get_compatible_metric_dict(output_type: int) -> Dict[str, str]:
	if output_type == BINARY:
		return dict(
        accuracy='error',
        log_loss='logloss',
        roc_auc='auc',
    )
	elif output_type == MULTICLASS:
		return dict(
        accuracy='merror',
        log_loss='mlogloss',
    )
	elif output_type == CONTINUOUS:
		return dict(
			mean_absolute_error='mae',
			root_mean_squared_error='rmse',
		)

def get_metric(output_type: int, optimize_metric: str) -> str:
	metric_dict = get_compatible_metric_dict(output_type=output_type)
	return metric_dict.get(optimize_metric, list(metric_dict.values())[DEFAULT_METRIC_INDEX])


DEFAULT_NUM_BOOST_ROUND = 10000
# Options: [10, 100, 200, 300, 400, 500, 1000, 10000]


def get_param_baseline(output_type):
    if output_type == BINARY:
        return get_param_binary_baseline()
    elif output_type == MULTICLASS:
        return get_param_multiclass_baseline()
    elif output_type == CONTINUOUS:
        return get_param_regression_baseline()
    else:
        return get_param_binary_baseline()


def get_base_params():
    base_params = {
        'n_estimators': DEFAULT_NUM_BOOST_ROUND,
        'learning_rate': 0.1,
        'n_jobs': -1,
    }
    return base_params


def get_param_binary_baseline():
    params = get_base_params()
    baseline_params = {
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'use_label_encoder': False,
    }
    params.update(baseline_params)
    return params


def get_param_multiclass_baseline():
    params = get_base_params()
    baseline_params = {
        'objective': 'multi:softmax',
        'booster': 'gbtree',
        'use_label_encoder': False,
    }
    params.update(baseline_params)
    return params


def get_param_regression_baseline():
    params = get_base_params()
    baseline_params = {
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
    }
    params.update(baseline_params)
    return params

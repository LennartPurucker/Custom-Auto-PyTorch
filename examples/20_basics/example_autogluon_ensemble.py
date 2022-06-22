"""
======================
Tabular Classification
======================

The following example shows how to fit a sample classification model
with AutoPyTorch
"""
import os
import tempfile as tmp
import warnings
from autoPyTorch.datasets.resampling_strategy import RepeatedCrossValTypes

from autoPyTorch.optimizer.utils import autoPyTorchSMBO

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import openml
import sklearn.model_selection

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.ensemble.utils import EnsembleSelectionTypes

############################################################################
# Data Loading
# ============
task = openml.tasks.get_task(task_id=3917)
dataset = task.get_dataset()
X, y, categorical_indicator, _ = dataset.get_data(
    dataset_format='dataframe',
    target=dataset.default_target_attribute,
)

train_indices, test_indices = task.get_train_test_split_indices()
# AutoPyTorch fails when it is given a y DataFrame with False and True
# values and category as dtype. in its inner workings it uses sklearn
# which cannot detect the column type.
if isinstance(y[1], bool):
    y = y.astype('bool')

# uncomment only for np.arrays

X_train = X.iloc[train_indices]
y_train = y.iloc[train_indices]
X_test = X.iloc[test_indices]
y_test = y.iloc[test_indices]

feat_type = ["numerical" if not indicator else "categorical" for indicator in categorical_indicator]

############################################################################
# Build and fit a classifier
# ==========================
api = TabularClassificationTask(
    # To maintain logs of the run, you can uncomment the
    # Following lines
    temporary_directory='./tmp/stacking_autogluon_tmp_10',
    output_directory='./tmp/stacking_autogluon_out_10',
    delete_tmp_folder_after_terminate=False,
    delete_output_folder_after_terminate=False,
    seed=1,
    ensemble_method=EnsembleSelectionTypes.stacking_autogluon,
    resampling_strategy=RepeatedCrossValTypes.repeated_k_fold_cross_validation,
    resampling_strategy_args={
        'num_splits': 2,
        'num_repeats': 1
    },
    ensemble_size=6,
    num_stacking_layers=1,
    feat_type=feat_type
)

############################################################################
# Search for an ensemble of machine learning algorithms
# =====================================================
api.run_autogluon_stacking(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test.copy(),
    y_test=y_test.copy(),
    dataset_name='Australian',
    optimize_metric='accuracy',
    total_walltime_limit=600,
    func_eval_time_limit_secs=130,
    all_supported_metrics=False,
    max_budget=10
)

############################################################################
# Print the final ensemble performance
# ====================================
y_pred = api.predict(X_test)
score = api.score(y_pred, y_test, metric='accuracy')
print(score)
# Print the final ensemble built by AutoPyTorch
print(api.show_models())

# Print statistics from search
# print(api.sprint_statistics())


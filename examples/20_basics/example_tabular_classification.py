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
from autoPyTorch.api.utils import get_autogluon_default_nn_config

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import openml

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.ensemble.utils import EnsembleSelectionTypes

############################################################################
# Data Loading
# ============
task = openml.tasks.get_task(task_id=146821)
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

search_space_updates = get_autogluon_default_nn_config(feat_type=feat_type)
############################################################################
# Build and fit a classifier
# ==========================
api = TabularClassificationTask(
    # To maintain logs of the run, you can uncomment the
    # Following lines
    temporary_directory='./tmp/autoPyTorch_example_tmp_06',
    output_directory='./tmp/autoPyTorch_example_out_06',
    delete_tmp_folder_after_terminate=False,
    delete_output_folder_after_terminate=False,
    seed=42,
    ensemble_size=5,
    resampling_strategy=RepeatedCrossValTypes.repeated_k_fold_cross_validation,
    search_space_updates=search_space_updates
)

############################################################################
# Search for an ensemble of machine learning algorithms
# =====================================================
api.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test.copy(),
    y_test=y_test.copy(),
    dataset_name='Australian',
    optimize_metric='accuracy',
    total_walltime_limit=1200,
    func_eval_time_limit_secs=200,
    enable_traditional_pipeline=True
)

############################################################################
# Print the final ensemble performance
# ====================================
y_pred = api.predict(X_test)
score = api.score(y_pred, y_test)
print(score)
# Print the final ensemble built by AutoPyTorch
print(api.show_models())

# Print statistics from search
print(api.sprint_statistics())

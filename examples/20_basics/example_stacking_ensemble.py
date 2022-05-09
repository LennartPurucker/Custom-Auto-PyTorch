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

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import sklearn.datasets
import sklearn.model_selection

from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.ensemble.utils import EnsembleSelectionTypes
from autoPyTorch.optimizer.utils import autoPyTorchSMBO

############################################################################
# Data Loading
# ============
X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X,
    y,
    random_state=1,
)

############################################################################
# Build and fit a classifier
# ==========================
api = TabularClassificationTask(
    # To maintain logs of the run, you can uncomment the
    # Following lines
    temporary_directory='./tmp/stacking_optimisation_ensemble_tmp_01',
    output_directory='./tmp/stacking_optimisation_ensemble_out_01',
    delete_tmp_folder_after_terminate=False,
    delete_output_folder_after_terminate=False,
    seed=4,
    ensemble_method=EnsembleSelectionTypes.stacking_optimisation_ensemble,
    resampling_strategy=RepeatedCrossValTypes.repeated_k_fold_cross_validation,
    ensemble_size=5
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
    total_walltime_limit=500,
    func_eval_time_limit_secs=100,
    enable_traditional_pipeline=False,
    smbo_class=autoPyTorchSMBO,
    all_supported_metrics=False,
    # use_ensemble_opt_loss=True
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
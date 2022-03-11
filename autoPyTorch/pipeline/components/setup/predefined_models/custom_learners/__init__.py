from typing import Any, Dict, Type, Union

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
)
from autoPyTorch.pipeline.components.setup.predefined_models.custom_learners.base_traditional_learner import \
    BaseCustomLearner
from autoPyTorch.pipeline.components.setup.predefined_models.custom_learners.lgbm.lgbm_learner import LGBModel
from autoPyTorch.pipeline.components.setup.predefined_models.custom_learners.catboost.catboost_learner import CatboostModel
from autoPyTorch.pipeline.components.setup.predefined_models.custom_learners.rf.rf_learner import RFModel
from autoPyTorch.pipeline.components.setup.predefined_models.custom_learners.extratrees.extratrees_learner import ExtraTreesModel
from autoPyTorch.pipeline.components.setup.predefined_models.custom_learners.knn.knn_learner import KNNModel


_traditional_learners = {
    # Sort by more robust models
    # Depending on the allocated time budget, only the
    # top models from this dict are two be fitted.
    # LGBM is the more robust model, with
    # internal measures to prevent crashes, overfit
    # Additionally, it is one of the state of the art
    # methods for tabular prediction.
    # Then follow with catboost for categorical heavy
    # datasets. The other models are complementary and
    # their ordering is not critical
    'lgb': LGBModel,
    'catboost': CatboostModel,
    'random_forest': RFModel,
    'extra_trees': ExtraTreesModel,
    'knn': KNNModel,
}
_addons = ThirdPartyComponents(BaseCustomLearner)


def add_traditional_learner(traditional_learner: BaseCustomLearner) -> None:
    _addons.add_component(traditional_learner)


def get_available_traditional_learners() -> Dict[str, Union[Type[BaseCustomLearner], Any]]:
    traditional_learners = dict()
    traditional_learners.update(_traditional_learners)
    return traditional_learners

from typing import Any, Dict, Type, Union

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
)
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.base_traditional_learner import \
    BaseTraditionalLearner
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.lgbm.lgbm import LGBModel
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.catboost.catboost import CatboostModel
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.random_forest.random_forest import RFModel
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.extratrees.extratrees import ExtraTreesModel
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.knn.knn import KNNModel

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
_addons = ThirdPartyComponents(BaseTraditionalLearner)


def add_traditional_learner(traditional_learner: BaseTraditionalLearner) -> None:
    _addons.add_component(traditional_learner)


def get_available_traditional_learners() -> Dict[str, Union[Type[BaseTraditionalLearner], Any]]:
    traditional_learners = dict()
    traditional_learners.update(_traditional_learners)
    return traditional_learners

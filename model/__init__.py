"""Register the models actually included in this public release.

The original initializer imported six absent variant files. Optional legacy pose
models remain available lazily without importing their dependencies for ADTT.
"""
from importlib import import_module

from .TripLane.TripLane import TripLane, TripLaneEncoder, TripLaneDecoder
from .TransDelta import DeltaTokenPredictor, DeltaTokenPredictorV2, DeltaTokenPredictorV21

_OPTIONAL = {
    "PoseDecoder": ".transformer.pose_decoder",
    "PoseEncoder": ".transformer.pose_encoder",
    "PlanUAutoRegTransformer": ".transformer.PlanUtransformer",
}


def __getattr__(name):
    if name not in _OPTIONAL:
        raise AttributeError(name)
    value = getattr(import_module(_OPTIONAL[name], __name__), name)
    globals()[name] = value
    return value


__all__ = ["TripLane", "TripLaneEncoder", "TripLaneDecoder", "DeltaTokenPredictor",
           "DeltaTokenPredictorV2", "DeltaTokenPredictorV21", *_OPTIONAL]

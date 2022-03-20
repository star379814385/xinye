# -*- coding: utf-8 -*-

from ..extractors_base import ExtractorBase
from ...registry import EXTRACTORS

from typing import Dict

@EXTRACTORS.register
class SeSeries(ExtractorBase):
    """
    The extractors for ResNet.

    Hyper-Parameters
        extract_features (list): indicates which feature maps to output. See available_feas for available feature maps.
            If it is ["all"], then all available features will be output.
    """
    default_hyper_params = {
        "extract_features": list(),
    }

    available_feas = ["base", "neck"]

    def __init__(self, model, hps: Dict or None = None):
        """
        Args:
            model (nn.Module): the model for extracting features.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        children = list(model.children())
        feature_modules = {
            "base":  children[0],
            "neck": children[2]
        }
        super(SeSeries, self).__init__(model, feature_modules, hps)

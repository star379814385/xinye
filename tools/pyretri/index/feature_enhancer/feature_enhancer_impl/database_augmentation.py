# -*- coding: utf-8 -*-

import torch

from ..feature_enhancer_base import EnhanceBase
from ...registry import ENHANCERS
from ...metric import KNN

from typing import Dict
from tqdm import tqdm


@ENHANCERS.register
class DBA(EnhanceBase):
    """
    Every feature in the database is replaced with a weighted sum of the point ’s own value and those of its top k nearest neighbors (k-NN).
    c.f. https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

    Hyper-Params:
        enhance_k (int): number of the nearest points to be calculated.
    """
    default_hyper_params = {
        "enhance_k": 10,
    }

    def __init__(self, hps: Dict or None = None, batch_size=2000):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(DBA, self).__init__(hps)
        knn_hps = {
            "top_k": self._hyper_params["enhance_k"] + 1,
        }
        self.batch_size = batch_size
        self.knn = KNN(knn_hps)

    # def __call__(self, feature: torch.tensor) -> torch.tensor:
    #     _, sorted_idx = self.knn(feature, feature)
    #     sorted_idx = sorted_idx[:, 1:].reshape(-1)
    #
    #     arg_fea = feature[sorted_idx].view(feature.shape[0], -1, feature.shape[1]).sum(dim=1)
    #     feature = feature + arg_fea
    #
    #     feature = feature / torch.norm(feature, dim=1, keepdim=True)
    #
    #     return feature

    def __call__(self, feature: torch.tensor) -> torch.tensor:
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float32)
        feature = feature.cuda()
        print("Running DBA now.")
        batch_num = int(feature.shape[0] / self.batch_size)
        if feature.shape[0] % self.batch_size != 0:
            batch_num += 1
        fea_list = []
        for i in tqdm(range(batch_num)):
            feature_batch = feature[i*self.batch_size:(i+1)*self.batch_size]
            _, sorted_idx = self.knn(feature_batch, feature)
            sorted_idx = sorted_idx[:, 1:].reshape(-1)
            arg_fea = feature[sorted_idx].view(feature_batch.shape[0], -1, feature_batch.shape[1]).sum(dim=1)
            feature_batch = feature_batch + arg_fea
            fea_list.append(feature_batch)
        feature = torch.cat(fea_list, dim=0)
        feature = feature / torch.norm(feature, dim=1, keepdim=True)
        feature = feature.cpu().numpy()
        return feature

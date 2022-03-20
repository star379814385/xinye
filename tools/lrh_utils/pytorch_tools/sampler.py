from torch.utils.data import Sampler
import numpy as np
from random import shuffle
import torch as t


# 过采样
class EnlargeLabelShufflingSampler(Sampler):
    """
        copy from https://github.com/pytorch/pytorch/pull/4153/files
    """
    """
        label shuffling technique aimed to deal with imbalanced class problem
        without replacement, manipulated by indices.
        All classes are enlarged to the same amount, so classes can be trained equally.
        argument:
        indices: indices of labels of the whole dataset

    """

    def __init__(self, indices, num_limit=None):
        print("Using EnlargeLabelShufflingSampler!!!")
        # mapping between label index and sorted label index
        sorted_labels = sorted(enumerate(indices), key=lambda x: x[1])
        uniq_labels = set(sorted_labels)
        count = 1
        count_of_each_label = []
        tmp = -1
        # get count of each label
        for (x, y) in sorted_labels:
            if y == tmp:
                count += 1
            else:
                if tmp != -1:
                    count_of_each_label.append(count)
                    count = 1
            tmp = y
        count_of_each_label.append(count)
        # get the largest count among all classes. used to enlarge every class to the same amount
        largest = int(np.amax(count_of_each_label))
        self.count_of_each_label = count_of_each_label
        self.enlarged_index = []

        # preidx used for find the mapping beginning of arg "sorted_labels"
        preidx = 0
        for x in range(len(self.count_of_each_label)):
            idxes = np.remainder(t.randperm(largest).numpy(), self.count_of_each_label[x]) + preidx
            for y in idxes:
                self.enlarged_index.append(sorted_labels[y][0])
            preidx += int(self.count_of_each_label[x])

        self.num_limit = num_limit

    def __iter__(self):
        shuffle(self.enlarged_index)
        if self.num_limit is not None and isinstance(self.num_limit, int):
            index_limit = self.enlarged_index[:self.num_limit]
            return iter(index_limit)
        return iter(self.enlarged_index)

    def __len__(self):
        return len(self.enlarged_index) if self.num_limit is None else self.num_limit
        # return np.amax(self.count_of_each_label) * len(self.count_of_each_label)


# 欠采样
class UnderShuffingSampler(Sampler):
    def __init__(self, indices):
        print("Using UnderShuffingSampler!!!")
        self.indices_dict = dict()
        for cnt, index in enumerate(indices):
            if index in self.indices_dict.keys():
                self.indices_dict[index].append(cnt)
            else:
                self.indices_dict[index] = [cnt]
        self.smallest = np.amin([len(index) for index in self.indices_dict.values()])

    def __iter__(self):
        cur_indices = []
        for indices in self.indices_dict.values():
            cur_indices += list(np.random.choice(indices, self.smallest, replace=False))
        shuffle(cur_indices)
        return iter(cur_indices)

    def __len__(self):
        return self.smallest * len(self.indices_dict.keys())

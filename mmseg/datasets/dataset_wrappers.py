from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.PALETTE = datasets[0].PALETTE


@DATASETS.register_module()
class RepeatDataset(object):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get item from original dataset."""
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """The length is multiplied by ``times``"""
        return self.times * self._ori_len

@DATASETS.register_module()
class BalanceDataset(object):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self._ori_len = len(self.dataset)
        self.new_inds = []
        self._balance_class()

    def _balance_class(self):
        import mmcv
        import os.path as osp
        import numpy as np
        infos_per_class = [[] for _ in range(len(self.CLASSES))]
        self.new_inds = [i for i in range(len(self.dataset.img_infos))]
        for i, img_info in enumerate(self.dataset.img_infos):
            ann = mmcv.imread(osp.join(self.dataset.ann_dir, img_info['ann']['seg_map']), 0)
            for j in range(len(self.CLASSES)):
                if np.count_nonzero(ann==j) > 0:
                    infos_per_class[j].append(i)
        max_num = max([len(infos_per_class[i]) for i in range(len(self.CLASSES))])
        for i in range(max_num):
            for infos in infos_per_class:
                self.new_inds.append(infos[i % len(infos)])


    def __getitem__(self, idx):
        """Get item from original dataset."""
        return self.dataset[self.new_inds[idx]]

    def __len__(self):
        """The length is multiplied by ``times``"""
        return len(self.new_inds)
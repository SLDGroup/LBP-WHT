# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class Flowers102(BaseDataset):

    def load_annotations(self):
        file_paths, gt_labels = [], []
        with open(self.ann_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                segs = line.split(' ')
                label = int(segs[-1])
                file_paths.append(segs[0])
                gt_labels.append(label)

        data_infos = []
        for path, gt_label in zip(file_paths, gt_labels):
            info = dict(
                img_prefix=self.data_prefix,
                img_info=dict(filename=path),
                gt_label=np.array(gt_label, dtype=np.int64)
            )
            data_infos.append(info)
        return data_infos

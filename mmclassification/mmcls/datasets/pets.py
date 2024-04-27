# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class Pets(BaseDataset):

    CLASSES = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British',
               'Egyptian', 'Maine', 'Persian', 'Ragdoll', 'Russian',
               'Siamese', 'Sphynx', 'american', 'basset', 'beagle',
               'boxer', 'chihuahua', 'english', 'german', 'great',
               'havanese', 'japanese', 'keeshond', 'leonberger',
               'miniature', 'newfoundland', 'pomeranian', 'pug',
               'saint', 'samoyed', 'scottish', 'shiba', 'staffordshire',
               'wheaten', 'yorkshire']

    def load_annotations(self):
        file_paths, gt_labels = [], []
        name_to_idx = {n: i for i, n in enumerate(self.CLASSES)}
        with open(self.ann_file, 'r') as f:
            for line in f.readlines():
                line = line.split(' ')[0]
                label = name_to_idx[line.split('_')[0]]
                file_paths.append(line + '.jpg')
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

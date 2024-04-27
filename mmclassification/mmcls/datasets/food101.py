# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class Food101(BaseDataset):
    CLASSES = ["apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
               "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
               "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
               "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
               "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
               "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
               "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
               "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries",
               "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt",
               "garlic_bread", "gnocchi", "greek_salad",
               "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger",
               "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream",
               "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese",
               "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings",
               "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
               "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich",
               "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops",
               "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara",
               "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki",
               "tiramisu", "tuna_tartare", "waffles"]

    def load_annotations(self):
        file_paths, gt_labels = [], []
        name_to_idx = {k: i for i, k in enumerate(self.CLASSES)}
        with open(self.ann_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                label = name_to_idx[line.split('/')[0]]
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

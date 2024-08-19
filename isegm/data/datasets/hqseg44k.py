from pathlib import Path
import numpy as np
import cv2
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class HQSegDataset(ISDataset):
    def __init__(self, dataset_path, **kwargs):
        super(HQSegDataset, self).__init__(**kwargs)
        dataset_path = Path(dataset_path)
        dataset_list = [path for path in dataset_path.iterdir() if path.is_dir()]

        self.dataset_samples = []
        self.masks_paths = []
        for dataset in dataset_list:
            if dataset.name == 'cascade_psp':
                for subset in dataset.iterdir():
                    if subset.is_dir():
                        sample_names = [x.stem for x in subset.iterdir() if x.suffix == '.jpg']

                        self.dataset_samples = self.dataset_samples + [path for path in subset.iterdir() if
                                                                       path.suffix == '.jpg']
                        self.masks_paths = self.masks_paths + [path for path in subset.iterdir() if
                                                               (path.stem in sample_names and path.suffix == '.png')]

            else:
                for subset in dataset.iterdir():
                    if subset.is_dir() and subset.stem != 'DIS-VD' and subset.stem !='COIFT' and subset.stem !='HRSOD':
                        im_path = subset / 'im'
                        mask_path = subset / 'gt'
                        sample_names = [x.stem for x in im_path.iterdir() if x.suffix == '.jpg']
                        self.dataset_samples = self.dataset_samples + [path for path in im_path.iterdir() if path.suffix == '.jpg']
                        self.masks_paths = self.masks_paths + [path for path in mask_path.iterdir() if path.stem in sample_names]

        self.dataset_samples.sort()
        self.masks_paths.sort()

    def get_sample(self, index) -> DSample:
        image_path = self.dataset_samples[index]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self.masks_paths[index]
        instances_mask = np.max(cv2.imread(str(mask_path)).astype(np.uint8), axis=2)
        instances_mask[instances_mask > 0] = 1
        if image.shape[:2] != instances_mask.shape:
            image = cv2.resize(image, [instances_mask.shape[1], instances_mask.shape[0]])

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)

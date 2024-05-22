# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import csv
import gc
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from monai.apps.auto3dseg.auto_runner import logger
from monai.apps.auto3dseg.transforms import EnsureSameShaped
from monai.auto3dseg.utils import datafold_read
from monai.bundle.config_parser import ConfigParser
from monai.config import KeysCollection
from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, decollate_batch, list_data_collate
from monai.inferers import SlidingWindowInfererAdapt
from monai.losses import DeepSupervisionLoss
from monai.metrics import CumulativeAverage, DiceHelper
from monai.networks.layers.factories import split_args
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from monai.transforms import (
    AsDiscreted,
    CastToTyped,
    ClassesToIndicesd,
    Compose,
    ConcatItemsd,
    CopyItemsd,
    CropForegroundd,
    DataStatsd,
    DeleteItemsd,
    EnsureTyped,
    Identityd,
    Invertd,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ResampleToMatchd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToDeviced,
)
from monai.transforms.transform import MapTransform
from monai.utils import ImageMetaKey, convert_to_dst_type, optional_import, set_determinism

from hecktor_crop_neck_region import HecktorCropNeckRegion

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"


print = logger.debug
tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

if __package__ in (None, ""):
    from utils import auto_adjust_network_settings, logger_configure
else:
    from .utils import auto_adjust_network_settings, logger_configure


class LabelEmbedClassIndex(MapTransform):
    """
    Label embedding according to class_index
    """

    def __init__(
        self, keys: KeysCollection = "label", allow_missing_keys: bool = False, class_index: Optional[List] = None
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be compared to the source_key item shape.
            allow_missing_keys: do not raise exception if key is missing.
            class_index: a list of class indices
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.class_index = class_index

    def label_mapping(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return torch.cat([sum([x == i for i in c]) for c in self.class_index], dim=0).to(dtype=dtype)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        if self.class_index is not None:
            for key in self.key_iterator(d):
                d[key] = self.label_mapping(d[key])
        return d


def schedule_validation_epochs(num_epochs, num_epochs_per_validation=None, fraction=0.16) -> list:
    """
    Schedule of epochs to validate (progressively more frequently)
        num_epochs - total number of epochs
        num_epochs_per_validation - if provided use a linear schedule with this step
        init_step
    """

    if num_epochs_per_validation is None:
        x = (np.sin(np.linspace(0, np.pi / 2, max(10, int(fraction * num_epochs)))) * num_epochs).astype(int)
        x = np.cumsum(np.sort(np.diff(np.unique(x)))[::-1])
        x[-1] = num_epochs
        x = x.tolist()
    else:
        if num_epochs_per_validation >= num_epochs:
            x = [num_epochs_per_validation]
        else:
            x = list(range(num_epochs_per_validation, num_epochs, num_epochs_per_validation))

    if len(x) == 0:
        x = [0]

    return x


class DataTransformBuilder:
    def __init__(
        self,
        roi_size: list,
        image_key: str = "image",
        label_key: str = "label",
        resample: bool = False,
        resample_resolution: Optional[list] = None,
        normalize_mode: str = "meanstd",
        normalize_params: Optional[dict] = None,
        crop_mode: str = "ratio",
        crop_params: Optional[dict] = None,
        extra_modalities: Optional[dict] = None,
        custom_transforms=None,
        debug: bool = False,
        rank: int = 0,
        lazy_verbose: bool = False,
        **kwargs,
    ) -> None:
        self.roi_size, self.image_key, self.label_key = roi_size, image_key, label_key

        self.resample, self.resample_resolution = resample, resample_resolution
        self.normalize_mode = normalize_mode
        self.normalize_params = normalize_params if normalize_params is not None else {}
        self.crop_mode = crop_mode
        self.crop_params = crop_params if crop_params is not None else {}

        self.extra_modalities = extra_modalities if extra_modalities is not None else {}
        self.custom_transforms = custom_transforms if custom_transforms is not None else {}

        self.extra_options = kwargs
        self.debug = debug
        self.rank = rank

        self.lazy_evaluation = False
        self.lazy_verbose = lazy_verbose

    def get_custom(self, name, **kwargs):
        tr = []
        for t in self.custom_transforms.get(name, []):
            if isinstance(t, dict):
                t.update(kwargs)
                t = ConfigParser(t).get_parsed_content(instantiate=True)
            tr.append(t)

        return tr

    def get_load_transforms(self):
        ts = self.get_custom("load_transforms")
        if len(ts) > 0:
            return ts

        keys = [self.image_key, self.label_key] + list(self.extra_modalities)
        ts.append(
            LoadImaged(keys=keys, ensure_channel_first=True, dtype=None, allow_missing_keys=True, image_only=True)
        )
        ts.append(EnsureTyped(keys=keys, data_type="tensor", dtype=torch.float, allow_missing_keys=True))
        ts.append(
            EnsureSameShaped(keys=self.label_key, source_key=self.image_key, allow_missing_keys=True, warn=self.debug)
        )

        ts.extend(self.get_custom("after_load_transforms"))

        return ts

    def get_resample_transforms(self, resample_label=True):
        ts = self.get_custom("resample_transforms", resample_label=resample_label)
        if len(ts) > 0:
            return ts

        keys = [self.image_key, self.label_key] if resample_label else [self.image_key]
        mode = ["bilinear", "nearest"] if resample_label else ["bilinear"]
        extra_keys = list(self.extra_modalities)

        if self.extra_options.get("crop_foreground", False) and len(extra_keys) == 0:
            ts.append(
                CropForegroundd(
                    keys=keys, source_key=self.image_key, allow_missing_keys=True, margin=10, allow_smaller=True
                )
            )
        if self.resample:
            if self.resample_resolution is None:
                raise ValueError("resample_resolution is not provided")

            pixdim = self.resample_resolution
            ts.append(
                Spacingd(
                    keys=keys,
                    pixdim=pixdim,
                    mode=mode,
                    dtype=torch.float,
                    min_pixdim=np.array(pixdim) * 0.75,
                    max_pixdim=np.array(pixdim) * 1.25,
                    allow_missing_keys=True,
                )
            )

            if resample_label:
                ts.append(
                    EnsureSameShaped(
                        keys=self.label_key, source_key=self.image_key, allow_missing_keys=True, warn=self.debug
                    )
                )

        for extra_key in extra_keys:
            ts.append(ResampleToMatchd(keys=extra_key, key_dst=self.image_key, dtype=np.float32))
        # crop_hecktor = HecktorCropNeckRegion(resample_label=resample_label,keys=[["image", "image2", "label"]],box_size=[200,200,310])
        # ts.extend(crop_hecktor)
        # ts.extend(self.get_custom("after_resample_transforms", resample_label=resample_label))

        return ts

    def get_normalize_transforms(self):
        ts = self.get_custom("normalize_transforms")
        if len(ts) > 0:
            return ts

        modalities = {self.image_key: self.normalize_mode}
        modalities.update(self.extra_modalities)

        for key, normalize_mode in modalities.items():
            if normalize_mode == "none":
                pass
            elif normalize_mode in ["range", "ct"]:
                intensity_bounds = self.normalize_params.get("intensity_bounds", None)
                if intensity_bounds is None:
                    intensity_bounds = [-250, 250]
                    warnings.warn(f"intensity_bounds is not specified, assuming {intensity_bounds}")

                ts.append(
                    ScaleIntensityRanged(
                        keys=key, a_min=intensity_bounds[0], a_max=intensity_bounds[1], b_min=-1, b_max=1, clip=False
                    )
                )
                ts.append(Lambdad(keys=key, func=lambda x: torch.sigmoid(x)))
            elif normalize_mode in ["meanstd", "mri"]:
                ts.append(NormalizeIntensityd(keys=key, nonzero=True, channel_wise=True))
            elif normalize_mode in ["pet"]:
                ts.append(Lambdad(keys=key, func=lambda x: torch.sigmoid((x - x.min()) / x.std())))
            else:
                raise ValueError("Unsupported normalize_mode" + str(self.normalize_mode))

        if len(self.extra_modalities) > 0:
            ts.append(ConcatItemsd(keys=list(modalities), name=self.image_key))  # concat
            ts.append(DeleteItemsd(keys=list(self.extra_modalities)))  # release memory

        label_dtype = self.normalize_params.get("label_dtype", None)
        if label_dtype is not None:
            ts.append(CastToTyped(keys=self.label_key, dtype=label_dtype, allow_missing_keys=True))

        ts.extend(self.get_custom("after_normalize_transforms"))
        return ts

    def get_crop_transforms(self):
        ts = self.get_custom("crop_transforms")
        if len(ts) > 0:
            return ts

        if self.roi_size is None:
            raise ValueError("roi_size is not specified")

        keys = [self.image_key, self.label_key]
        ts = []
        ts.append(SpatialPadd(keys=keys, spatial_size=self.roi_size))

        if self.crop_mode == "ratio":
            output_classes = self.crop_params.get("output_classes", None)
            if output_classes is None:
                raise ValueError("crop_params option output_classes must be specified")

            crop_ratios = self.crop_params.get("crop_ratios", None)
            cache_class_indices = self.crop_params.get("cache_class_indices", False)
            max_samples_per_class = self.crop_params.get("max_samples_per_class", None)
            if max_samples_per_class <= 0:
                max_samples_per_class = None
            indices_key = None

            if cache_class_indices:
                ts.append(
                    ClassesToIndicesd(
                        keys=self.label_key,
                        num_classes=output_classes,
                        indices_postfix="_cls_indices",
                        max_samples_per_class=max_samples_per_class,
                    )
                )

                indices_key = self.label_key + "_cls_indices"

            num_crops_per_image = self.crop_params.get("num_crops_per_image", 1)
            if num_crops_per_image > 1:
                print(f"Cropping with num_crops_per_image {num_crops_per_image}")

            ts.append(
                RandCropByLabelClassesd(
                    keys=keys,
                    label_key=self.label_key,
                    num_classes=output_classes,
                    spatial_size=self.roi_size,
                    num_samples=num_crops_per_image,
                    ratios=crop_ratios,
                    indices_key=indices_key,
                    warn=False,
                )
            )
        elif self.crop_mode == "rand":
            ts.append(RandSpatialCropd(keys=keys, roi_size=self.roi_size, random_size=False))
        else:
            raise ValueError("Unsupported crop mode" + str(self.crop_mode))

        ts.extend(self.get_custom("after_crop_transforms"))

        return ts

    def get_augment_transforms(self):
        ts = self.get_custom("augment_transforms")
        if len(ts) > 0:
            return ts

        if self.roi_size is None:
            raise ValueError("roi_size is not specified")

        ts = []
        ts.append(
            RandAffined(
                keys=[self.image_key, self.label_key],
                prob=0.2,
                rotate_range=[0.26, 0.26, 0.26],
                scale_range=[0.2, 0.2, 0.2],
                mode=["bilinear", "nearest"],
                spatial_size=self.roi_size,
                cache_grid=True,
                padding_mode="border",
            )
        )
        ts.append(RandFlipd(keys=[self.image_key, self.label_key], prob=0.5, spatial_axis=0))
        ts.append(RandFlipd(keys=[self.image_key, self.label_key], prob=0.5, spatial_axis=1))
        ts.append(RandFlipd(keys=[self.image_key, self.label_key], prob=0.5, spatial_axis=2))
        ts.append(
            RandGaussianSmoothd(
                keys=self.image_key, prob=0.2, sigma_x=[0.5, 1.0], sigma_y=[0.5, 1.0], sigma_z=[0.5, 1.0]
            )
        )
        ts.append(RandScaleIntensityd(keys=self.image_key, prob=0.5, factors=0.3))
        ts.append(RandShiftIntensityd(keys=self.image_key, prob=0.5, offsets=0.1))
        ts.append(RandGaussianNoised(keys=self.image_key, prob=0.2, mean=0.0, std=0.1))

        ts.extend(self.get_custom("after_augment_transforms"))

        return ts

    def get_final_transforms(self):
        return self.get_custom("final_transforms")
    def get_save_transform_data(
                                output_path="",
                                data_root_dir="",
                                out_postfix="",

                                
                            ):
        ts = []
        ts.append(
                SaveImaged(
                    keys=["image","image2","label"],
                    output_dir=output_path,
                    output_postfix="",
                    data_root_dir=data_root_dir,
                    # output_dtype=output_dtype,
                    separate_folder=False,
                    squeeze_end_dims=True,
                    resample=False,
                    print_log=False,
                )
            )
        return ts
    @classmethod
    def get_postprocess_transform(
        cls,
        save_mask=False,
        invert=False,
        transform=None,
        sigmoid=False,
        output_path=None,
        resample=False,
        data_root_dir="",
        output_dtype=np.uint8,
    ) -> Compose:
        ts = []
        if invert and transform is not None:
            # if resample:
            #     ts.append(ToDeviced(keys="pred", device=torch.device("cpu")))
            ts.append(Invertd(keys="pred", orig_keys="image", transform=transform, nearest_interp=False))

        if save_mask and output_path is not None:
            ts.append(CopyItemsd(keys="pred", times=1, names="seg"))
            ts.append(AsDiscreted(keys="seg", argmax=True) if not sigmoid else AsDiscreted(keys="seg", threshold=0.5))
            ts.append(
                SaveImaged(
                    keys=["seg"],
                    output_dir=output_path,
                    output_postfix="",
                    data_root_dir=data_root_dir,
                    output_dtype=output_dtype,
                    separate_folder=False,
                    squeeze_end_dims=True,
                    resample=False,
                    print_log=False,
                )
            )

        return Compose(ts)

    def __call__(self, augment=False, resample_label=False, lazy_evaluation=False) -> Compose:
        self.lazy_evaluation = lazy_evaluation

        ts = []
        ts.extend(self.get_load_transforms())
        ts.extend(self.get_resample_transforms(resample_label=resample_label))
        ts.extend(self.get_normalize_transforms())

        if augment:
            ts.extend(self.get_crop_transforms())
            # ts.extend(self.get_augment_transforms())            # mask0000000000000000000000000000000

        ts.extend(self.get_final_transforms())
        # ts.extend(self.get_save_transform_data(keys=["image","label"],
        #             output_dir='E:\expert\project2\preprocess\imagesTr',
        #             output_postfix="",
        #             data_root_dir="E:\expert\project2\preprocess",
        #             # output_dtype=output_dtype,
        #             separate_folder=False,
        #             squeeze_end_dims=True,
        #             resample=False,
        #             print_log=True,))

        if self.lazy_evaluation:
            warnings.warn("Lazy evaluation is not currently enabled.")
        compose_ts = Compose(ts)

        return compose_ts

    def __repr__(self) -> str:
        out: str = f"DataTransformBuilder: with image_key: {self.image_key}, label_key: {self.label_key} \n"
        out += f"roi_size {self.roi_size} resample {self.resample} resample_resolution {self.resample_resolution} \n"
        out += f"normalize_mode {self.normalize_mode} normalize_params {self.normalize_params} \n"
        out += f"crop_mode {self.crop_mode} crop_params {self.crop_params} \n"
        out += f"extra_modalities {self.extra_modalities} \n"
        for k, trs in self.custom_transforms.items():
            out += f"Custom {k} : {str(trs)} \n"
        return out




if __name__ == "__main__":

    pass

from copy import deepcopy
import sys
import torch
from tqdm import tqdm
import yaml
from monai.bundle.config_parser import ConfigParser
import monai.transforms as transforms
import os
import numpy as np
from PIL import Image

from monai.data import  DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from monai.auto3dseg.utils import datafold_read
from hecktor_crop_neck_region import HecktorCropNeckRegion
from monai.apps.auto3dseg.transforms import EnsureSameShaped
from monai.transforms import (
    SaveImage,
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

from datatransform import DataTransformBuilder, LabelEmbedClassIndex



# -*- coding: utf-8 -*-
#
# #mount google drive
# from google.colab import drive
# drive.mount('/content/drive')

#import relevant dependencies
import glob, os, functools
import numpy as np
import pandas as pd
import SimpleITK as sitk


#预处理配置文件
process_config_path = "./hyper_parameters.yaml"
data_list_file_path = "./4.json"
data_file_base_dir = "/mnt/data/fkk/DATASET/data_raw/"
# 打开配置文件并加载内容
with open(process_config_path, 'r') as file:
    config = yaml.safe_load(file)
    # print("config :  ",config)

def get_custom_transforms(config):
        config = config

        # check for custom transforms
        custom_transforms = {}
        for tr in config.get("custom_data_transforms", []):
            must_include_keys = ("key", "path", "transform")
            if not all(k in tr for k in must_include_keys):
                raise ValueError("custom transform must include " + str(must_include_keys))

            if os.path.abspath(tr["path"]) not in sys.path:
                sys.path.append(os.path.abspath(tr["path"]))

            custom_transforms.setdefault(tr["key"], [])
            custom_transforms[tr["key"]].append(tr["transform"])

        if len(custom_transforms) > 0 :
            print(f"Using custom transforms {custom_transforms}")

        if isinstance(config["class_index"], list) and len(config["class_index"]) > 0:
            # custom label embedding, if class_index provided
            custom_transforms.setdefault("final_transforms", [])
            custom_transforms["final_transforms"].append(
                LabelEmbedClassIndex(keys="label", class_index=config["class_index"], allow_missing_keys=True)
            )

        return custom_transforms
def get_data_transform_builder(config):
    # _data_transform_builder =None
    # if _data_transform_builder is None:
    config = config
    custom_transforms = get_custom_transforms(config=config)

    _data_transform_builder = DataTransformBuilder(
        roi_size=config["roi_size"],
        resample=config["resample"],
        resample_resolution=config["resample_resolution"],
        normalize_mode=config["normalize_mode"],
        normalize_params={
            "intensity_bounds": config["intensity_bounds"],
            "label_dtype": torch.uint8 if config["input_channels"] < 255 else torch.int16,
        },
        crop_mode=config["crop_mode"],
        crop_params={
            "output_classes": config["output_classes"],
            "crop_ratios": config["crop_ratios"],
            "cache_class_indices": config["cache_class_indices"],
            "num_crops_per_image": config["num_crops_per_image"],
            "max_samples_per_class": 2000,
        },
        extra_modalities=config["extra_modalities"],
        custom_transforms=custom_transforms,
        lazy_verbose=False,
        crop_foreground=config.get("crop_foreground", True),
        debug=config["debug"],
        )

    return _data_transform_builder



def datafold_read_training(datalist, basedir, key = "training",key_testing = "testing"):
    """读取json中的所有的文件,不区分training and  valing or testing 
    Read a list of data dictionary `datalist`

    Args:
        datalist: the name of a JSON file listing the data, or a dictionary.
        basedir: directory of image files.
        key: usually 'training',testing to load all file path.

    Returns:
        A list of array (training, validation).
    """
    
    if isinstance(datalist, str):
        json_data = ConfigParser.load_config_file(datalist)
    else:
        json_data = datalist

    dict_data = deepcopy(json_data[key])
    dict_data_testing = deepcopy(json_data[key_testing])

    for d in dict_data:
        for k, _ in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
    for d in dict_data_testing:
        for k, _ in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    file = []
    
    for d in dict_data:
        file.append(d)
    for d in dict_data_testing:
        file.append(d)

    return file##返回training 关键字中的所有的文件的绝对路径 为一个列表 列表中每个元素是字典，字典中是 key value














train_transform = get_data_transform_builder(config=config)(augment=True, resample_label=True, lazy_evaluation=False)####augmentmask00000000 augment
#读取文件列表
files = datafold_read_training(datalist=data_list_file_path, basedir=data_file_base_dir)
testing_files, _ = datafold_read(
                datalist=data_list_file_path, basedir=data_file_base_dir, fold=-1, key="testing"
            )

print(f" files {len(files)},\n"
    f"testing files {len(testing_files)}\n"
    )

idx_ =0 
#输出文件路径
output_image_dir = "./imagesTs"
output_label_dir = "./labelsTs"
# 示例转换函数
transform = transforms.Compose(train_transform)
for idx, train_file in tqdm(enumerate(files), total=len(files), desc="Processing Files"):
    _name = os.path.basename(train_file['label'])
    name = os.path.splitext(os.path.splitext(_name)[0])[0]
    transformed_data = transform(train_file)
    print(f"HEK_{idx} 总共有{len(transformed_data)}张图像....")
    # data, target = transformed_data[0]["image"], transformed_data[0]["label"]  #train mask000
    for i in range(len(transformed_data)):
        print(f"保存HEK_{name} 的第{i}张图像....")
        data, target = transformed_data[i]["image"], transformed_data[i]["label"]  #
            
        img_ct = torch.as_tensor(data[0,:,:,:])
        img_pt = torch.as_tensor(data[1,:,:,:])
        label = torch.as_tensor(target[0,:,:,:])

        img_ct = img_ct.permute(2, 1, 0)
        img_pt = img_pt.permute(2, 1, 0)
        label = label.permute(2, 1, 0)

        img_ct = sitk.GetImageFromArray(img_ct.numpy())
        img_pt = sitk.GetImageFromArray(img_pt.numpy())
        label = sitk.GetImageFromArray(label.numpy())
        
        # print("img_ct.shape",img_ct.GetSize())
        # print("img_pt.shape",img_pt.GetSize())
        # print("label.shape",label.GetSize())
        # 将图像写入文件  
        sitk.WriteImage(img_ct, os.path.join(output_image_dir,name+'_0000.nii.gz'))
        sitk.WriteImage(img_pt, os.path.join(output_image_dir,name+'_0001.nii.gz'))
        sitk.WriteImage(label, os.path.join(output_label_dir,name+'.nii.gz'))
        idx_ =idx_ + 1



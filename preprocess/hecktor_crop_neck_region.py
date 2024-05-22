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

from logging import Logger
import warnings
import torch
import numpy as np

from monai.data import MetaTensor
from monai.utils.misc import ImageMetaKey
from monai.transforms import CropForegroundd


class HecktorCropNeckRegion(CropForegroundd):
    """
    A simple pre-processing transform to approximately crop the head and neck region based on a PET image.
    This transform relies on several assumptions of patient orientation with a head location on the top,
    and is specific for Hecktor22 dataset, and should not be used for an arbitrary PET image pre-processing.
    """

    def __init__(
        self,
        keys=["image", "image2", "label"],
        source_key="image",
        box_size=[200, 200, 310],
        allow_missing_keys=True,
        **kwargs,
    ) -> None:
        super().__init__(keys=keys, source_key=source_key, allow_missing_keys=allow_missing_keys, **kwargs)
        self.box_size = box_size

    def __call__(self, data, **kwargs):
        d = dict(data)

        im_pet = d["image2"][0]
        box_size = np.array(self.box_size)  # H&N region to crop in mm , defaults to 200x200x310mm
        filename = ""

        if isinstance(im_pet, MetaTensor):
            filename = im_pet.meta[ImageMetaKey.FILENAME_OR_OBJ]
            box_size = (box_size / np.array(im_pet.pixdim)).astype(int)  # compensate for resolution  算出像素值为box_size[200,200,310]/[1,1,1]

        box_start, box_end = self.extract_roi(im_pet=im_pet, box_size=box_size)

        use_label = "label" in d and "label" in self.keys and (d["image"].shape[1:] == d["label"].shape[1:])# 字典d 中有label,   self.key有label,   image和label的shape的除了第一维外，其他都相同

        if use_label:
            # if label mask is available, let's check if the cropped region includes all foreground
            before_sum = 0
            before_sum = d["label"].sum().item()
            after_sum = (
                (d["label"][0, box_start[0] : box_end[0], box_start[1] : box_end[1], box_start[2] : box_end[2]])
                .sum()
                .item()
            )
            if before_sum != after_sum:
                 print(
                    "WARNING, H&N crop could be incorrect!!! ",
                    before_sum,
                    after_sum,
                    "image:",
                    d["image"].shape,
                    "pet:",
                    d["image2"].shape,
                    "label:",
                    d["label"].shape,
                    "updated box_size",
                    box_size,
                    "box_start",
                    box_start,
                    "box_end:",
                    box_end,
                    "filename",
                    filename,
                )
                # warnings.warn(
                #     "WARNING, H&N crop could be incorrect!!! ",
                #     before_sum,
                #     after_sum,
                #     "image:",
                #     d["image"].shape,
                #     "pet:",
                #     d["image2"].shape,
                #     "label:",
                #     d["label"].shape,
                #     "updated box_size",
                #     box_size,
                #     "box_start",
                #     box_start,
                #     "box_end:",
                #     box_end,
                #     "filename",
                #     filename,
                # )

        d[self.start_coord_key] = box_start
        d[self.end_coord_key] = box_end

        for key, m in self.key_iterator(d, self.mode):   #key:["image", "image2", "label"]
            if key == "label" and not use_label:    #不对label进行处理，只对其他image，image2处理
                continue
            d[key] = self.cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m)

        return d
    
    # def extract_roi(self, im_pet, box_size):
    #     crop_len = int(0.75 * im_pet.shape[2])#深度，轴向切片的0.75      512*0.75
    #     im = im_pet[..., crop_len:]#len到末尾所有切片 切片了0.75倍深度

    #     mask = ((im - im.mean()) / im.std()) > 1  ##阈值处理，用于找到具有高像素值的感兴趣区域。mask为二进制数据列表，im中的所有大于均值标准化的像素值都为True
    #     comp_idx = torch.argwhere(mask) #comp_idx为True的索引

    #     center = torch.mean(comp_idx.float(), dim=0).cpu().int().numpy()    #计算roi中心坐标
    #     xmin = torch.min(comp_idx, dim=0).values.cpu().int().numpy()        #roi 最小的坐标
    #     xmax = torch.max(comp_idx, dim=0).values.cpu().int().numpy()        #roi 最小的坐标

    #     xmin[:2] = center[:2] - box_size[:2] // 2  ##找到最上面的x y坐标
    #     xmax[:2] = center[:2] + box_size[:2] // 2  ##最下x y的坐标

    #     xmax[2] = xmax[2] + crop_len    #复原z轴的深度  512
    #     xmin[2] = max(0, xmax[2] - box_size[2]) #从最上深度裁剪到box_size的第三维

    #     return xmin.astype(int), xmax.astype(int)
    def extract_roi(self, im_pet, box_size):
        crop_len = int(0.75 * im_pet.shape[2])#深度，轴向切片的0.75      512*0.75
        im = im_pet[..., crop_len:]#len到末尾所有切片 切片了0.75倍深度

        mask = ((im - im.mean()) / im.std()) > 1  ##阈值处理，用于找到具有高像素值的感兴趣区域。mask为二进制数据列表，im中的所有大于均值标准化的像素值都为True
        comp_idx = torch.argwhere(mask) #comp_idx为True的索引

        center = torch.mean(comp_idx.float(), dim=0).cpu().int().numpy()    #计算roi中心坐标
        xmin = torch.min(comp_idx, dim=0).values.cpu().int().numpy()        #roi 最小的坐标
        xmax = torch.max(comp_idx, dim=0).values.cpu().int().numpy()        #roi 最小的坐标

        xmin[:2] = center[:2] - box_size[:2] // 2  ##找到最上面的x y坐标
        xmax[:2] = center[:2] + box_size[:2] // 2  ##最下x y的坐标

        xmax[2] = xmax[2] + crop_len    #复原z轴的深度  512
        xmin[2] = max(0, xmax[2] - box_size[2]) #从最上深度裁剪到box_size的第三维

        return xmin.astype(int), xmax.astype(int)
        # return [0,0,0] ,im_pet.shape
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

# import warnings
# import torch
# import numpy as np

# from monai.data import MetaTensor
# from monai.utils.misc import ImageMetaKey
# from monai.transforms import CropForegroundd
# # from monai.transforms import Compose, EnsureTyped, LoadImaged, Orientationd

# class HecktorCropNeckRegion(CropForegroundd):
#     """
#     A simple pre-processing transform to approximately crop the head and neck region based on a PET image.
#     This transform relies on several assumptions of patient orientation with a head location on the top,
#     and is specific for Hecktor22 dataset, and should not be used for an arbitrary PET image pre-processing.
#     """

#     def __init__(
#         self,
#         keys=["image", "image2", "label"],
#         source_key="image",
#         box_size=[200, 200, 310],
#         allow_missing_keys=True,
#         **kwargs,
#     ) -> None:
#         super().__init__(keys=keys, source_key=source_key, allow_missing_keys=allow_missing_keys, **kwargs)
#         self.box_size = box_size

#     def __call__(self, data, **kwargs):
#         d = dict(data)
        
#         im_pet = d["image2"][0]
#         box_size = np.array(self.box_size)  # H&N region to crop in mm , defaults to 200x200x310mm
#         filename = ""

#         if isinstance(im_pet, MetaTensor):
#             filename = im_pet.meta[ImageMetaKey.FILENAME_OR_OBJ]
#             box_size = (box_size / np.array(im_pet.pixdim)).astype(int)  # compensate for resolution

#         box_start, box_end = self.extract_roi(im_pet=im_pet, box_size=box_size)

#         use_label = "label" in d and "label" in self.keys and (d["image"].shape[1:] == d["label"].shape[1:])

#         if use_label:
#             # if label mask is available, let's check if the cropped region includes all foreground
#             before_sum = d["label"].sum().item()
#             after_sum = (
#                 (d["label"][0, box_start[0] : box_end[0], box_start[1] : box_end[1], box_start[2] : box_end[2]])
#                 .sum()
#                 .item()
#             )
#         if before_sum != after_sum:
#             warning_message = (
#                 "WARNING, H&N crop could be incorrect!!!\n"
#                 f"before_sum: {before_sum}, after_sum: {after_sum}, "
#                 f"image shape: {d['image'].shape}, pet shape: {d['image2'].shape}, label shape: {d['label'].shape}, "
#                 f"updated box_size: {box_size}, box_start: {box_start}, box_end: {box_end}, filename: {filename}"
#             )
#             warnings.warn(warning_message)


#         d[self.start_coord_key] = box_start
#         d[self.end_coord_key] = box_end

#         for key, m in self.key_iterator(d, self.mode):
#             if key == "label" and not use_label:
#                 continue
#             d[key] = self.cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m)

#         return d

#     def extract_roi(self, im_pet, box_size):
#         crop_len = int(0.75 * im_pet.shape[2])
#         im = im_pet[..., crop_len:]

#         mask = ((im - im.mean()) / im.std()) > 1
#         comp_idx = torch.argwhere(mask)

#         center = torch.mean(comp_idx.float(), dim=0).cpu().int().numpy()
#         xmin = torch.min(comp_idx, dim=0).values.cpu().int().numpy()
#         xmax = torch.max(comp_idx, dim=0).values.cpu().int().numpy()

#         xmin[:2] = center[:2] - box_size[:2] // 2
#         xmax[:2] = center[:2] + box_size[:2] // 2

#         xmax[2] = xmax[2] + crop_len
#         xmin[2] = max(0, xmax[2] - box_size[2])

#         return xmin.astype(int), xmax.astype(int)
    
    
    
    
    
    
    
    
# #加入可视化操作 
# import matplotlib.pyplot as plt
# import torch
# import numpy as np
# import warnings

# from monai.data import MetaTensor
# from monai.utils.misc import ImageMetaKey
# from monai.transforms import CropForegroundd

# class HecktorCropNeckRegion(CropForegroundd):
#     """
#     A simple pre-processing transform to approximately crop the head and neck region based on a PET image.
#     This transform relies on several assumptions of patient orientation with a head location on the top,
#     and is specific for Hecktor22 dataset, and should not be used for an arbitrary PET image pre-processing.
#     """

#     def __init__(
#         self,
#         keys=["image", "image2", "label"],
#         source_key="image",
#         box_size=[200, 200, 310],
#         allow_missing_keys=True,
#         **kwargs,
#     ) -> None:
#         super().__init__(keys=keys, source_key=source_key, allow_missing_keys=allow_missing_keys, **kwargs)
#         self.box_size = box_size

#     def __call__(self, data, **kwargs):
#         d = dict(data)

#         im_pet = d["image2"][0]
#         box_size = np.array(self.box_size)  # H&N region to crop in mm, defaults to 200x200x310mm
#         filename = ""

#         if isinstance(im_pet, MetaTensor):
#             filename = im_pet.meta[ImageMetaKey.FILENAME_OR_OBJ]
#             box_size = (box_size / np.array(im_pet.pixdim)).astype(int)  # compensate for resolution

#         box_start, box_end = self.extract_roi(im_pet=im_pet, box_size=box_size)

#         use_label = "label" in d and "label" in self.keys and (d["image"].shape[1:] == d["label"].shape[1:])

#         # Visualization before cropping
#         if use_label:
#             crop_len = int(0.75 * im_pet.shape[2])

#             plt.figure(figsize=(18, 6))
#             plt.subplot(1, 3, 1)
#             plt.title("Before Crop (Image)")
#             plt.imshow(d["image"][0, :, :, crop_len:], cmap="gray")

#             plt.subplot(1, 3, 2)
#             plt.title("Before Crop (PET)")
#             plt.imshow(d["image2"][0, :, :, crop_len:], cmap="gray")

#             plt.subplot(1, 3, 3)
#             plt.title("Before Crop (Label)")
#             plt.imshow(d["label"][0, :, :, crop_len:], cmap="gray")

#         d[self.start_coord_key] = box_start
#         d[self.end_coord_key] = box_end

#         for key, m in self.key_iterator(d, self.mode):
#             if key == "label" and not use_label:
#                 continue
#             d[key] = self.cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m)

#         # Visualization after cropping
#         if use_label:
#             plt.figure(figsize=(18, 6))
#             plt.subplot(1, 3, 1)
#             plt.title("After Crop (Image)")
#             plt.imshow(d["image"][0, box_start[0]:box_end[0], box_start[1]:box_end[1], box_start[2]:box_end[2]], cmap="gray")

#             plt.subplot(1, 3, 2)
#             plt.title("After Crop (PET)")
#             plt.imshow(d["image2"][0, box_start[0]:box_end[0], box_start[1]:box_end[1], box_start[2]:box_end[2]], cmap="gray")

#             plt.subplot(1, 3, 3)
#             plt.title("After Crop (Label)")
#             plt.imshow(d["label"][0, box_start[0]:box_end[0], box_start[1]:box_end[1], box_start[2]:box_end[2]], cmap="gray")

#             plt.show()

#             # Check if the cropped region includes all foreground
#             before_sum = d["label"].sum().item()
#             after_sum = (
#                 (d["label"][0, box_start[0] : box_end[0], box_start[1] : box_end[1], box_start[2] : box_end[2]])
#                 .sum()
#                 .item()
#             )
#             if before_sum != after_sum:
#                 warning_message = (
#                     "WARNING, H&N crop could be incorrect!!!\n"
#                     f"before_sum: {before_sum}, after_sum: {after_sum}, "
#                     f"image shape: {d['image'].shape}, pet shape: {d['image2'].shape}, label shape: {d['label'].shape}, "
#                     f"updated box_size: {box_size}, box_start: {box_start}, box_end: {box_end}, filename: {filename}"
#                 )
#                 warnings.warn(warning_message)

#         return d

#     def extract_roi(self, im_pet, box_size):
#         crop_len = int(0.75 * im_pet.shape[2])
#         im = im_pet[..., crop_len:]

#         mask = ((im - im.mean()) / im.std()) > 1
#         comp_idx = torch.argwhere(mask)

#         center = torch.mean(comp_idx.float(), dim=0).cpu().int().numpy()
#         xmin = torch.min(comp_idx, dim=0).values.cpu().int().numpy()
#         xmax = torch.max(comp_idx, dim=0).values.cpu().int().numpy()

#         xmin[:2] = center[:2] - box_size[:2] // 2
#         xmax[:2] = center[:2] + box_size[:2] // 2

#         xmax[2] = xmax[2] + crop_len
#         xmin[2] = max(0, xmax[2] - box_size[2])

#         return xmin.astype(int), xmax.astype(int)

    
    
    
    
    
    
    
    


# import warnings
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from monai.data import MetaTensor
# from monai.utils.misc import ImageMetaKey
# from monai.transforms import CropForegroundd
# from monai.transforms import Compose, EnsureTyped, LoadImaged, Orientationd

# class HecktorCropNeckRegion(CropForegroundd):
#     """
#     A simple pre-processing transform to approximately crop the head and neck region based on a PET image.
#     This transform relies on several assumptions of patient orientation with a head location on the top,
#     and is specific for Hecktor22 dataset, and should not be used for an arbitrary PET image pre-processing.
#     """

#     def __init__(
#         self,
#         keys=["image", "image2", "label"],
#         source_key="image",
#         box_size=[200, 200, 310],
#         allow_missing_keys=True,
#         **kwargs,
#     ) -> None:
#         super().__init__(keys=keys, source_key=source_key, allow_missing_keys=allow_missing_keys, **kwargs)
#         self.box_size = box_size

#     def __call__(self, data, **kwargs):
#         d = dict(data)
#         print("-------------数据进入预处理脚本的状态data:----------",data)#全为地址
#         tran_list = [
#             LoadImaged(keys="image2", ensure_channel_first=True, image_only=True),
#             # EnsureTyped(keys=keys, data_type="tensor", dtype=torch.float),
#             # Orientationd(keys=keys, axcodes="RAS"),
#             # HecktorCropNeckRegion()
#             # HecktorCropNeckRegion(keys="image2", source_key="image2", box_size=[200, 200, 310])
# ]
#         transform =Compose(tran_list)
#         d =transform(d)
#         print("---------打印transforms 对image2转换成metetensor的结果--------d[image2]",d["image2"],"---------------------")
#         print("=== Before HecktorCropNeckRegion ===")
#         print("box_size:", self.box_size)
#         print("im_pet shape:", d["image2"].shape)
#         # # print("PET数据进入预处理脚本的状态data:",d)
 
#         # transform_list = [
#         #     LoadImaged(keys='image2', ensure_channel_first=True, image_only=True),
#         #     EnsureTyped(keys='image2', data_type="tensor", dtype=torch.float),
#         #     Orientationd(keys='image2', axcodes="RAS"),
#         # ]
#         # transform = Compose(transform_list)
#         # d = transform(d)
#         # print('''d HECKTOR_transform转换数据后''',d)
#         im_pet = d["image2"][0]
#         box_size = np.array(self.box_size)  # H&N region to crop in mm , defaults to 200x200x310mm
#         filename = ""

#         if isinstance(im_pet, MetaTensor):
#             filename = im_pet.meta[ImageMetaKey.FILENAME_OR_OBJ]
#             box_size = (box_size / np.array(im_pet.pixdim)).astype(int)  # compensate for resolution

#         box_start, box_end = self.extract_roi(im_pet=im_pet, box_size=box_size)

#         use_label = "label" in d and "label" in self.keys and (d["image"].shape[1:] == d["label"].shape[1:])

#         if use_label:
#             # if label mask is available, let's check if the cropped region includes all foreground
#             before_sum = d["label"].sum().item()
#             after_sum = (
#                 (d["label"][0, box_start[0] : box_end[0], box_start[1] : box_end[1], box_start[2] : box_end[2]])
#                 .sum()
#                 .item()
#             )
#         if before_sum != after_sum:
#             warning_message = (
#                 "WARNING, H&N crop could be incorrect!!!\n"
#                 f"before_sum: {before_sum}, after_sum: {after_sum}, "
#                 f"image shape: {d['image'].shape}, pet shape: {d['image2'].shape}, label shape: {d['label'].shape}, "
#                 f"updated box_size: {box_size}, box_start: {box_start}, box_end: {box_end}, filename: {filename}"
#             )
#             warnings.warn(warning_message)


#         d[self.start_coord_key] = box_start
#         d[self.end_coord_key] = box_end

#         for key, m in self.key_iterator(d, self.mode):
#             if key == "label" and not use_label:
#                 continue
#             d[key] = self.cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m)
        
        
        
#         d["image2_cropped"] = d["image2"]
#         d["image2_cropped"] = self.cropper.crop_pad(img=d["image2_cropped"], box_start=box_start, box_end=box_end, mode="constant")
#         print("=== After HecktorCropNeckRegion ===")
#         print("box_start:", box_start)
#         print("box_end:", box_end)
#         print('''d["image2"]''',d["image2"].shape)

#         # 获取处理后的图像数据
#         image2_cropped = d["image2_cropped"].detach().cpu().numpy()[0, 0]

#         # 可视化对比处理前后的数据
#         fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#         axes[0].imshow(d["image2"].detach().cpu().numpy()[0, 0], cmap="gray")
#         axes[0].set_title("Before Preprocessing")
#         axes[1].imshow(image2_cropped, cmap="gray")
#         axes[1].set_title("After Preprocessing")
#         plt.show()
        
        
#         return d

#     def extract_roi(self, im_pet, box_size):
#         crop_len = int(0.75 * im_pet.shape[2])
#         im = im_pet[..., crop_len:]

#         mask = ((im - im.mean()) / im.std()) > 1
#         comp_idx = torch.argwhere(mask)

#         center = torch.mean(comp_idx.float(), dim=0).cpu().int().numpy()
#         xmin = torch.min(comp_idx, dim=0).values.cpu().int().numpy()
#         xmax = torch.max(comp_idx, dim=0).values.cpu().int().numpy()

#         xmin[:2] = center[:2] - box_size[:2] // 2
#         xmax[:2] = center[:2] + box_size[:2] // 2

#         xmax[2] = xmax[2] + crop_len
#         xmin[2] = max(0, xmax[2] - box_size[2])

#         return xmin.astype(int), xmax.astype(int)







# keys=["image", "image2", "label"]

# path_data = "hecktor_part_data/imagesTr/CHUM-001__PT.nii.gz"
# path_data2 = "hecktor_part_data/imagesTr/CHUM-001__CT.nii.gz"
# path_label = "hecktor_part_data/labelsTr/CHUM-001.nii.gz"
# data={
#     "image": path_data2,
#     "image2": path_data,
#     "label": path_label
# }
# print("-------------数据进入预处理脚本的状态data:----------",data)#全为地址
# tran_list = [
#     LoadImaged(keys="image2", ensure_channel_first=True, image_only=True),
#     # EnsureTyped(keys=keys, data_type="tensor", dtype=torch.float),
#     # Orientationd(keys=keys, axcodes="RAS"),
#     # HecktorCropNeckRegion()
#     # HecktorCropNeckRegion(keys="image2", source_key="image2", box_size=[200, 200, 310])
# ]
# transform = Compose(transform_list)
# d = transform(data)
# print('''------------d HECKTOR_transform转换数据后-----------''',d)#全为metetensor

# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# axes[0].imshow(d["image2"].detach().cpu().numpy()[0, 0], cmap="gray")
# axes[0].set_title("Before Preprocessing")
# axes[1].imshow(d["image2"].detach().cpu().numpy()[0, 0], cmap="gray")
# axes[1].set_title("After Preprocessing")
# plt.show()


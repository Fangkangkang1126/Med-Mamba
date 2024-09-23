# 空间-序列多模态CT/PET分割头颈部肿瘤方法 - nnUNetv2部署
## 项目简介
本项目基于nnUNetv2框架，针对多模态CT/PET影像数据进行头颈部肿瘤的精确分割。通过定制训练器和网络结构，本项目旨在提高肿瘤分割的准确性和效率。
## 环境部署
在开始之前，请确保已经安装了以下依赖环境：
- Python 3.10
- PyTorch
- CUDA (如使用GPU加速)
- nnUNetv2框架
### nnUNetv2部署教程
请按照以下链接完成nnUNetv2的部署：
[nnUNetv2官方部署教程](https://github.com/MIC-DKFZ/nnUNetv2/blob/master/SETUP.md)
## 项目文件替换
完成nnUNetv2部署后，使用本项目提供的定制代码进行替换：
1. 下载本项目代码：
```bash
git clone https://github.com/YourUsername/YourProject.git
cd YourProject
```
2. 替换nnUNetv2中的训练器和网络代码：
```bash
cp -r YourProject/nnUNetTrainerCustom/* path_to_nnUNetv2/nnUNetTrainer/
cp -r YourProject/nnUNetNetworkCustom/* path_to_nnUNetv2/nnUNetNetwork/
```
确保替换路径正确无误。
## 数据集准备
本项目使用的数据集需通过以下链接申请：
[头颈部肿瘤多模态CT/PET数据集申请](https://example.com/dataset_application)
请按照提供指南完成数据集的申请和下载。
## 训练流程
在开始训练前，请确保数据集已按照nnUNetv2要求进行预处理。
1. 设置数据集路径：
```bash
export nnUNet_raw_data_base=/path/to/your/nnUNet_raw_data_base
export nnUNet_preprocessed=/path/to/your/nnUNet_preprocessed
export RESULTS_FOLDER=/path/to/your/nnUNet_results
```
2. 运行以下命令进行训练：
```bash
python path_to_nnUNetv2/nnUNet_train.py 3d_fullres nnUNetTrainerCustom TaskXXXYourTask 0
```
请将`TaskXXXYourTask`替换为你的具体任务名称。
## 推理流程
完成训练后，使用以下命令进行推理：
```bash
python path_to_nnUNetv2/nnUNet_predict.py -i /path/to/your/input_data -o /path/to/your/output_data -t 3 -m 3d_fullres -f 0
```
- `-i` 参数指定输入数据的路径。
- `-o` 参数指定推理结果的保存路径。
- `-t` 参数指定任务编号。
- `-m` 参数指定使用的模型配置。
- `-f` 参数指定使用的fold编号。
## 注意事项
- 确保在替换代码和运行命令时路径正确无误。
- 根据实际硬件配置调整训练参数以优化性能。
## 联系方式
如有疑问或建议，请通过以下方式联系：
- Email: fangkangkang@nefu.edu.cn
感谢您的使用！

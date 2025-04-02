# VI-ReID Frame

## Requirement
```
numpy==1.24.2
torch==2.0.0
torchvision==0.15.1
PyYAML==6.0
pytorch-ignite==0.1.2 #IMPORTANT！！！！！！
```
> 需要对`torch._six`库作一定的修正,不好直接描述，但是很简单

## Dataset

修改`./configs/dataset.py`中SYSU数据集的位置

将`rand_perm_cam.mat`放入`SYSU/exp`文件夹中

## 最后准备

`parser.add_argument("--file_prefix", type=str, default="/home/vegetabot/Filesys/Logs&Checkpoints/VI-ReID-Frame")`这里改一下

## Train

见`Train.log`


## Contact
If you find any problem, please feel free to contact me (fangxingye@bit.edu.cn). A brief self-introduction is required, if you would like to get an in-depth help from me.
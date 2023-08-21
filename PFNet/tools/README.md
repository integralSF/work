# SOME TOOLS

## 0) construction

when you open the folder, you can see the construction.

## 1) introduce
单独测试两个文件的CD

you can use:
```
python CD.py
```
to calculate the chamfer distance.

---
两个点云A，B。
得到A-B的结果，也就是partial，为模型的输入。

calulate two points cloud the different points:
```
python DIFF.py
```
---
下采样768使用的方法

down-sampling:
```
python FPS.py
```
---
归一化单个文件

normalization:
```
python NORMALIZE.py
```
---
对于我再制作数据集的过程中把各个工具集成了这个文件。

for my own dataset, i will use:
```
python pipeline_process_dateset.py
```
to process the sigle dataset from start to end.

---
上采样到4000

up-sampleing:
```
python UPSAMPLE.py
```
---
使用open3d可视化

if you want to visualize by using open3d:
```
python Visualize.py
```

## 2) Additional instructions
All these are using for my own dataset.If it's just a simple copy, it's likely not working.
But my code can give you some enlighten.
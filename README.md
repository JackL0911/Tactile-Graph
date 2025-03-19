# Tactile-Graph
用GCL做Tactile和Vision融合（GCL for Tactile and Visual）
![image-20250319134432820](graph/pipline.png)
如上图所示，输入触觉和图像序列，分别经过触觉和视觉encoder，用Moco的方法，四个encoder两两进行动量更新。过encoder之后再分别过8个头生成8个嵌入向量序列并转换成8个graph，进行无监督的图对比学习。
数据集使用的Touch and Go https://touch-and-go.github.io/
baseline https://sites.google.com/view/mvitac/home

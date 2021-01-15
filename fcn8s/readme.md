## train fcn8s using paddlepaddle==1.8.4

```
train_fcn.py 用于训练fcn模型，可以在fcns.py修改相应的上采样大小
softmax_logit_loss.py 定义损失函数
data_transform.py 扩增数据量
dataloader.py 在pd中加载本地数据
vgg.py 定义VGG网络
utils.py 定义精度方式

train_again.py 在train_fcn.py保存好模型后，如需二次训练时可以用，只写到加载模型，后面的训练大同小异
conver_inference_model.py 用TracedLayer方法转成pd的inference model，从动态图转成静态图，可以用于部署在c++平台
run_inference_model.py 代码中上半部分是静态图使用， 下半部分是动态图使用。理论上来说静态图要好过动态图，
但测试结果却是相反，具体原因不知道是不是说在c++的部署会好的意思。

```


```
├── dummy_data
    │   ├── GroundTruth_trainval_png
    │   │   ├── 2008_000002.png
    │   │   ├── 2008_000003.png
    │   │   ├── 2008_000007.png
    │   │   ├── ...
    │   ├── JPEGImages
    │   │   ├── 2008_000002.jpg
    │   │   ├── 2008_000003.jpg
    │   │   ├── 2008_000007.jpg
    │   │   └── ...
    │   └── list.txt
```  

<br>

```
list.txt:
    JPEGImages/2008_000002.jpg GroundTruth_trainval_png/2008_000002.png
    JPEGImages/2008_000003.jpg GroundTruth_trainval_png/2008_000003.png
    JPEGImages/2008_000007.jpg GroundTruth_trainval_png/2008_000007.png
    ...
```
<br>

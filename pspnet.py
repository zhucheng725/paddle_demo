import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Dropout
from resnet_dilated import ResNet50

# pool with different bin_size
# interpolate back to input size
# concat
class PSPModule(Layer):
    def __init__(self, num_channels, bin_size_list):
        super(PSPModule, self).__init__()
        self.bin_size_list = bin_size_list
        num_filters = num_channels //len(bin_size_list)

        self.features = []
        for i in range(len(bin_size_list)):
            self.features.append(
                fluid.dygraph.Sequential(
                    Conv2D(num_channels=num_channels, num_filters=num_filters, filter_size=1,stride=1,padding=0),
                    BatchNorm(num_channels=num_filters,act="relu")
                         )
            )


        
    def forward(self, inputs):
        out = [inputs]
        for idx, feat in enumerate(self.features):
            x = fluid.layers.adaptive_pool2d(input = inputs, pool_size =self.bin_size_list [idx], pool_type="max")
            x = feat(x)
            x = fluid.layers.interpolate(x, inputs.shape[2::], resample='BILINEAR') 
            out.append(x)

        out = fluid.layers.concat(out, axis=1)

        return out


class PSPNet(Layer):
    def __init__(self, num_classes=59, backbone='resnet50'):
        super(PSPNet, self).__init__()

        res = ResNet50(pretrained=False)

        # stem: res.conv, res.pool2d_max
        self.layer0 = fluid.dygraph.Sequential(
                res.conv,
                res.pool2d_max
        )
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4
        
        bin_size_list = [1,2,3,6]
        num_channels = 2048

        self.pspmodule = PSPModule(num_channels=num_channels, bin_size_list = bin_size_list)

        num_channels = num_channels * 2

        self.classifier = fluid.dygraph.Sequential(
                Conv2D(num_channels=num_channels, num_filters=512, filter_size=3,stride=1,padding=1),
                BatchNorm(num_channels=512,act="relu"),
                Dropout(0.5),
                Conv2D(num_channels=512, num_filters=num_classes, filter_size=1)
        )
     

        # psp: 2048 -> 2048*2

        # cls: 2048*2 -> 512 -> num_classes

        # aux: 1024 -> 256 -> num_classes
        
    def forward(self, inputs):

        # aux: tmp_x = layer3
        x = self.layer0(inputs)
        print("layer0.shape", x.shape)
        x = self.layer1(x)
        print("layer1.shape", x.shape)
        x = self.layer2(x)
        print("layer2.shape", x.shape)
        x = self.layer3(x)
        print("layer3.shape", x.shape)
        x = self.layer4(x)
        print("layer4.shape", x.shape)
        x = self.pspmodule(x)
        print("pspmodule.shape", x.shape)
        x = self.classifier(x)    
        print("classifier.shape", x.shape)
        x = fluid.layers.interpolate(x, inputs.shape[2::]) #, align_corners=True
        print("interpolate.shape", x.shape)

        return x



def main():
    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        x_data=np.random.rand(2,3, 473, 473).astype(np.float32)
        x = to_variable(x_data)
        model = PSPNet(num_classes=59)
        model.train()
        #pred = model(x)
        #print("pred.shape :", pred.shape)
        pred, aux = model(x)
        print(pred.shape, aux.shape)


if __name__ =="__main__":
    main()


import paddle
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Conv2DTranspose
from paddle.fluid.dygraph import Dropout
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Linear




class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 use_bn=True,
                 act='relu',
                 name=None):
        super(ConvBNLayer, self).__init__(name)

        self.use_bn = use_bn
        if use_bn:
            self.conv = Conv2D(num_channels=num_channels,
                                num_filters=num_filters,
                                filter_size=filter_size,
                                stride=stride,
                                padding=(filter_size-1)//2,
                                groups=groups,
                                act=None,
                                bias_attr=None)
            self.bn = BatchNorm(num_filters, act=act)
        else:
            self.conv = Conv2D(num_channels=num_channels,
                                num_filters=num_filters,
                                filter_size=filter_size,
                                stride=stride,
                                padding=(filter_size-1)//2,
                                groups=groups,
                                act=act,
                                bias_attr=None)

    def forward(self, inputs):
        y = self.conv(inputs)
        if self.use_bn:
            y = self.bn(y)
        return y



class VGG(fluid.dygraph.Layer):
    def __init__(self, layers=16, use_bn=False, num_classes=1000):
        super(VGG, self).__init__()
        self.layers = layers
        self.use_bn = use_bn
        supported_layers = [16, 19]
        assert layers in supported_layers

        if layers == 16:
            depth = [2, 2, 3, 3, 3]
        elif layers == 19:
            depth = [2, 2, 4, 4, 4]

        num_channels = [3, 64, 128, 256, 512]
        num_filters = [64, 128, 256, 512, 512]

        self.layer1 = fluid.dygraph.Sequential(*self.make_layer(num_channels[0], num_filters[0], depth[0], use_bn, name='layer1'))
        self.layer2 = fluid.dygraph.Sequential(*self.make_layer(num_channels[1], num_filters[1], depth[1], use_bn, name='layer2'))
        self.layer3 = fluid.dygraph.Sequential(*self.make_layer(num_channels[2], num_filters[2], depth[2], use_bn, name='layer3'))
        self.layer4 = fluid.dygraph.Sequential(*self.make_layer(num_channels[3], num_filters[3], depth[3], use_bn, name='layer4'))
        self.layer5 = fluid.dygraph.Sequential(*self.make_layer(num_channels[4], num_filters[4], depth[4], use_bn, name='layer5'))

        self.classifier = fluid.dygraph.Sequential(
                Linear(input_dim=512 * 7 * 7, output_dim=4096, act='relu'),
                Dropout(),
                Linear(input_dim=4096, output_dim=4096, act='relu'),
                Dropout(),
                Linear(input_dim=4096, output_dim=num_classes))
                
        self.out_dim = 512 * 7 * 7


    def forward(self, inputs):
        x = self.layer1(inputs)
        x = fluid.layers.pool2d(x, pool_size=2, pool_stride=2)
        x = self.layer2(x)
        x = fluid.layers.pool2d(x, pool_size=2, pool_stride=2)
        x = self.layer3(x)
        x = fluid.layers.pool2d(x, pool_size=2, pool_stride=2)
        x = self.layer4(x)
        x = fluid.layers.pool2d(x, pool_size=2, pool_stride=2)
        x = self.layer5(x)
        x = fluid.layers.pool2d(x, pool_size=2, pool_stride=2)
        x = fluid.layers.adaptive_pool2d(x, pool_size=(7,7), pool_type='avg')
        x = fluid.layers.reshape(x, shape=[-1, self.out_dim])
        x = self.classifier(x)

        return x

    def make_layer(self, num_channels, num_filters, depth, use_bn, name=None):
        layers = []
        layers.append(ConvBNLayer(num_channels, num_filters, use_bn=use_bn, name=f'{name}.0'))
        for i in range(1, depth):
            layers.append(ConvBNLayer(num_filters, num_filters, use_bn=use_bn, name=f'{name}.{i}'))
        return layers




class FCN8s(fluid.dygraph.Layer):
    '''
    组建FCN8s网络
    '''
    def __init__(self,num_classes=59):
        super(FCN8s,self).__init__()
        backbone=VGG(layers=16, use_bn=True)

        self.layer1 = backbone.layer1
        self.layer1[0].conv._padding = [100,100]
        self.pool1 = Pool2D(pool_size=2,pool_stride=2,ceil_mode=True)
        self.layer2 = backbone.layer2
        self.pool2 = Pool2D(pool_size=2,pool_stride=2,ceil_mode=True)
        self.layer3 = backbone.layer3
        self.pool3 = Pool2D(pool_size=2,pool_stride=2,ceil_mode=True)
        self.layer4 = backbone.layer4
        self.pool4 = Pool2D(pool_size=2,pool_stride=2,ceil_mode=True)
        self.layer5 = backbone.layer5
        self.pool5 = Pool2D(pool_size=2,pool_stride=2,ceil_mode=True)
        
        self.fc6 = Conv2D(512,4096,7,act='relu')
        self.fc7 = Conv2D(4096,4096,1,act='relu')
        self.drop6 = Dropout()
        self.drop7 = Dropout()

        self.score = Conv2D(4096,num_classes,1)
        self.score_pool3 = Conv2D(256,num_classes,1)
        self.score_pool4 = Conv2D(512,num_classes,1)

        self.up_output = Conv2DTranspose(num_channels=num_classes,
                                        num_filters=num_classes,
                                        filter_size=4,
                                        stride =2,
                                        bias_attr=False)
        self.up_pool4 = Conv2DTranspose(num_channels=num_classes,
                                        num_filters=num_classes,
                                        filter_size=4,
                                        stride =2,
                                        bias_attr=False)
        self.up_final = Conv2DTranspose(num_channels=num_classes,
                                        num_filters=num_classes,
                                        filter_size=16,
                                        stride =16,
                                        bias_attr=False)
    def forward(self,inputs):
        x=self.layer1(inputs)
        x=self.pool1(x)
        x=self.layer2(x)
        x=self.pool2(x)
        x=self.layer3(x)
        x=self.pool3(x)
        pool3= x
        x=self.layer4(x)
        x=self.pool4(x)
        pool4= x
        x=self.layer5(x)
        x=self.pool5(x)

        x=self.fc6(x)
        x=self.drop6(x)
        x=self.fc7(x)
        x=self.drop7(x)

        x=self.score(x)
        x=self.up_output(x)
        up_output = x
        x=self.score_pool4(pool4)

        x = x [:,:,5:5+up_output.shape[2],5:5+up_output.shape[3]]
        up_pool4 =x
        x=up_pool4 + up_output
        x=self.score_pool3(pool3)
        x=x[:,:,9:9+up_pool4.shape[2],9:9+up_pool4.shape[3]]
        up_pool3=x
        x= up_pool3 +up_pool4
        x= self.up_final(x)
        
        x=x[:,:,31:31+inputs.shape[2],31:31+inputs.shape[3]]

        return x
    

def main():
    place =  paddle.fluid.CUDAPlace(0)
    #place = paddle.fluid.CPUPlace()
    with fluid.dygraph.guard(place):

        x_data = np.random.rand(2, 3, 512, 512).astype(np.float32)
        x = to_variable(x_data)
        model = FCN8s(num_classes=59)
        model.eval()
        pred = model(x)
        print(pred.shape)

if __name__ == "__main__":
    main()




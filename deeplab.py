import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from resnet_multi_grid import ResNet50



class ASPPPooling(Layer):
    # TODO:
    def __init__(self, num_channels, num_filters):
        super(ASPPPooling, self).__init__()
        self.conv1 = Conv2D(num_channels=num_channels,
                       num_filters= num_filters, 
                       filter_size = 1,stride=1)
        self.bn1 = BatchNorm(num_channels=num_filters,act = 'relu')
        
    def forward(self, inputs):
        x = fluid.layers.adaptive_pool2d(input = inputs, pool_size =1, pool_type="max")
        x = self.conv1(x)
        x = self.bn1(x)
        x = fluid.layers.interpolate(x, inputs.shape[2::], resample='BILINEAR') 
        return x


class ASPPConv(Layer):
    # TODO:
    def __init__(self, num_channels,num_filters, rates):
        super(ASPPConv, self).__init__()
        self.conv1 = Conv2D(num_channels=num_channels,
                       num_filters= num_filters, dilation=rates[0],
                       filter_size = 3,stride=1, padding=rates[0])#padding = dilation
        self.bn1 = BatchNorm(num_channels=num_filters, act = 'relu')

        self.conv2 = Conv2D(num_channels=num_channels,
                       num_filters= num_filters, dilation=rates[1],
                       filter_size = 3,stride=1, padding=rates[1])#padding = dilation
        self.bn2 = BatchNorm(num_channels=num_filters, act = 'relu')

        self.conv3 = Conv2D(num_channels=num_channels,
                       num_filters= num_filters, dilation=rates[2],
                       filter_size = 3,stride=1, padding=rates[2])#padding = dilation
        self.bn3 = BatchNorm(num_channels=num_filters, act = 'relu')
        

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x1 = self.bn1(x1)
        x2 = self.conv2(inputs)
        x2 = self.bn2(x2)
        x3 = self.conv3(inputs)
        x3 = self.bn3(x3)
        x = fluid.layers.concat(input = [x1, x2, x3], axis = 1) # N C H W
        return x




class ASPPModule(Layer):
    # TODO: 
    def __init__(self, num_channels,num_filters, rates):
        super(ASPPModule, self).__init__()

        self.conv0 = Conv2D(num_channels=num_channels,
                       num_filters= num_filters, 
                       filter_size = 1,stride=1)
        self.bn0 = BatchNorm(num_channels=num_filters, act = 'relu')

        self.asppconv = ASPPConv(num_channels = num_channels,num_filters =num_filters, rates = rates)
        self.aspppool = ASPPPooling(num_channels= num_channels,num_filters = num_filters)

        self.conv1 = Conv2D(num_channels=num_filters*(2+len(rates)),
                       num_filters= num_filters, 
                       filter_size = 1,stride=1)
        self.bn1 = BatchNorm(num_channels=num_filters, act = 'relu')

 

    def forward(self, inputs):
        x0 = self.conv0(inputs)
        x0 = self.bn0(x0)
        x1 = self.asppconv(inputs)
        x2 = self.aspppool(inputs)


        x = fluid.layers.concat(input = [x0, x1, x2], axis = 1) # N C H W
        x = self.conv1(x)
        x = self.bn1(x)
        return x


class DeepLabHead(fluid.dygraph.Sequential):
    def __init__(self, num_channels, num_classes):
        super(DeepLabHead, self).__init__(
                ASPPModule(num_channels, 256, [12, 24, 36]),
                Conv2D(256, 256, 3, padding=1),
                BatchNorm(256, act='relu'),
                Conv2D(256, num_classes, 1)
                )


class DeepLabConvCopy(Layer):
    def __init__(self,num_channels,num_filters, multi_grid, layer_rate, is_short):
        super(DeepLabConvCopy, self).__init__()
        self.short = is_short
        self.convcopy1 = Conv2D(num_channels=num_channels,
                       num_filters= num_filters, dilation=multi_grid[0]*layer_rate,
                       filter_size = 3,stride=1, padding=multi_grid[0]*layer_rate if multi_grid[0]*layer_rate >1 else 0)#padding = dilation
        self.bn1 = BatchNorm(num_channels=num_filters, act = 'relu')

        self.convcopy2 = Conv2D(num_channels=num_filters,
                       num_filters= num_filters, dilation=multi_grid[1]*layer_rate,
                       filter_size = 3,stride=1, padding=multi_grid[1]*layer_rate if multi_grid[1]*layer_rate >1 else 0)#padding = dilation
        self.bn2 = BatchNorm(num_channels=num_filters, act = 'relu')

        self.convcopy3 = Conv2D(num_channels=num_filters,
                       num_filters= num_filters, dilation=multi_grid[2]*layer_rate,
                       filter_size = 3,stride=1, padding=multi_grid[2]*layer_rate if multi_grid[2]*layer_rate >1 else 0)#padding = dilation
        self.bn3 = BatchNorm(num_channels=num_filters, act = 'relu')

        self.convcopy4 = Conv2D(num_channels=num_filters,
                       num_filters= num_filters, dilation=multi_grid[0]*layer_rate,
                       filter_size = 3,stride=1, padding=multi_grid[0]*layer_rate if multi_grid[0]*layer_rate >1 else 0)#padding = dilation
        self.bn4 = BatchNorm(num_channels=num_filters, act = 'relu')

        self.convcopy5 = Conv2D(num_channels=num_filters,
                       num_filters= num_filters, dilation=multi_grid[1]*layer_rate,
                       filter_size = 3,stride=1, padding=multi_grid[1]*layer_rate if multi_grid[1]*layer_rate >1 else 0)#padding = dilation
        self.bn5 = BatchNorm(num_channels=num_filters, act = 'relu')

        self.convcopy6 = Conv2D(num_channels=num_filters,
                       num_filters= num_filters, dilation=multi_grid[2]*layer_rate,
                       filter_size = 3,stride=1, padding=multi_grid[2]*layer_rate if multi_grid[2]*layer_rate >1 else 0)#padding = dilation
        self.bn6 = BatchNorm(num_channels=num_filters, act = 'relu')

        self.convcopy7 = Conv2D(num_channels=num_filters,
                       num_filters= num_filters, dilation=multi_grid[0]*layer_rate,
                       filter_size = 3,stride=1, padding=multi_grid[0]*layer_rate if multi_grid[0]*layer_rate >1 else 0)#padding = dilation
        self.bn7 = BatchNorm(num_channels=num_filters, act = 'relu')

        self.convcopy8 = Conv2D(num_channels=num_filters,
                       num_filters= num_filters, dilation=multi_grid[1]*layer_rate,
                       filter_size = 3,stride=1, padding=multi_grid[1]*layer_rate if multi_grid[1]*layer_rate >1 else 0)#padding = dilation
        self.bn8 = BatchNorm(num_channels=num_filters, act = 'relu')

        self.convcopy9 = Conv2D(num_channels=num_filters,
                       num_filters= num_filters, dilation=multi_grid[2]*layer_rate,
                       filter_size = 3,stride=1, padding=multi_grid[2]*layer_rate if multi_grid[2]*layer_rate >1 else 0)#padding = dilation
        self.bn9 = BatchNorm(num_channels=num_filters, act = 'relu')


    def forward(self, inputs):
        
        x1 = inputs
        x = self.convcopy1(inputs)
        x = self.bn1(x)
        x = self.convcopy2(x)
        x = self.bn2(x)
        x = self.convcopy3(x)
        x = self.bn3(x)
        print(f'x1.shape{x1.shape}, x.shape{x.shape}')
        if self.short == True:
            x = fluid.layers.elementwise_add(x=x1 , y=x, act='relu')
        x2 = x
        x = self.convcopy4(x)
        x = self.bn4(x)
        x = self.convcopy5(x)
        x = self.bn5(x)
        x = self.convcopy6(x)
        x = self.bn6(x)
        if self.short == True:
            x = fluid.layers.elementwise_add(x=x2 , y=x, act='relu')
        x3 = x
        x = self.convcopy7(x)
        x = self.bn7(x)
        x = self.convcopy8(x)
        x = self.bn8(x)
        x = self.convcopy9(x)
        x = self.bn9(x)
        if self.short == True:
            x = fluid.layers.elementwise_add(x=x3 , y=x, act='relu')
        return x






class DeepLab(Layer):
    # TODO:
    def __init__(self, num_classes):
        super(DeepLab, self).__init__()
        ResNet = ResNet50(pretrained=False, duplicate_blocks=True)
        self.ResLayer0 = fluid.dygraph.Sequential(
            ResNet.conv,
            ResNet.pool2d_max
        )
        self.ResLayer1 = ResNet.layer1
        self.ResLayer2 = ResNet.layer2
        self.ResLayer3 = ResNet.layer3
        #self.ResLayer4 = ResNet.layer4
        #self.ResLayer5 = ResNet.layer5
        #self.ResLayer6 = ResNet.layer6
        #self.ResLayer7 = ResNet.layer7

        self.layer_rate = [2, 4, 8, 16]
        self.multi_grid = [1, 2, 4]

        self.ResLayer4_copy0 = DeepLabConvCopy(num_channels = 1024, num_filters = 2048, multi_grid = self.multi_grid, layer_rate = self.layer_rate[0], is_short= False)
        self.ResLayer4_copy1 = DeepLabConvCopy(num_channels = 2048, num_filters = 2048, multi_grid = self.multi_grid, layer_rate = self.layer_rate[1], is_short= True)
        self.ResLayer4_copy2 = DeepLabConvCopy(num_channels = 2048, num_filters = 2048, multi_grid = self.multi_grid, layer_rate = self.layer_rate[2], is_short= True)
        self.ResLayer4_copy3 = DeepLabConvCopy(num_channels = 2048, num_filters = 2048, multi_grid = self.multi_grid, layer_rate = self.layer_rate[3], is_short= True)

        self.DeepLabHead = DeepLabHead(num_channels = 2048, num_classes = 59)



 
        
        
    def forward(self, inputs):
        x = self.ResLayer0(inputs)#[2, 64, 128, 128]
        x = self.ResLayer1(x)#[2, 256, 128, 128]
        x = self.ResLayer2(x)#[2, 512, 64, 64]
        x = self.ResLayer3(x)#[2, 1024, 64, 64]
        #x = self.ResLayer4(x)#[2, 2048, 64, 64]

        #x = self.ResLayer5(x)#[2, 2048, 64, 64]
        #x = self.ResLayer6(x)#[2, 2048, 64, 64]
        #x = self.ResLayer7(x)#[2, 2048, 64, 64]
   
        x = self.ResLayer4_copy0(x)
        x = self.ResLayer4_copy1(x)
        x = self.ResLayer4_copy2(x)
        x = self.ResLayer4_copy3(x)

        print('DeepLabConvCopy.shape',x.shape) #DeepLabConvCopy.shape [2, 2048, 64, 64]
        x = self.DeepLabHead(x)
        print('DeepLabHead.shape', x.shape)

        x = fluid.layers.interpolate(x, inputs.shape[2::], resample='BILINEAR', align_corners=True) 
        print('interpolate.shape', x.shape)

        

        return x




def main():
    #with fluid.dygraph.guard(fluid.CUDAPlace(0)):
    with fluid.dygraph.guard(fluid.CPUPlace()):
        x_data = np.random.rand(2, 3, 512, 512).astype(np.float32)
        x = to_variable(x_data)
        model = DeepLab(num_classes=59)
        model.eval()
        pred = model(x)
        print(pred.shape)



if __name__ == '__main__':
    main()

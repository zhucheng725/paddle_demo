import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Conv2DTranspose


class Encoder(Layer):
    def __init__(self, num_channels, num_filters):
        super(Encoder, self).__init__()
        #TODO: encoder contains:
        #       1 3x3conv + 1bn + relu + 
        #       1 3x3conc + 1bn + relu +
        #       1 2x2 pool
        # return features before and after pool
        self.conv_1 = Conv2D(num_channels=num_channels,
                       num_filters= num_filters,
                       filter_size = 3,stride=1, padding=0)

        self.bn_1 = BatchNorm(num_channels=num_filters, act = 'relu')

        self.conv_2 = Conv2D(num_channels=num_filters,
                       num_filters= num_filters,
                       filter_size = 3,stride=1, padding=0)

        self.bn_2 = BatchNorm(num_channels=num_filters, act = 'relu')

        self.pool_1 = Pool2D(pool_size=2,pool_stride=2,pool_type="max",ceil_mode=True)


    def forward(self, inputs):
        # TODO: finish inference part
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_pooled = self.pool_1(x)
        return x, x_pooled


class Decoder(Layer):
    def __init__(self, num_channels, num_filters):
        super(Decoder, self).__init__()
        # TODO: decoder contains:
        #       1 2x2 transpose conv (makes feature map 2x larger)
        #       1 3x3 conv + 1bn + 1relu + 
        #       1 3x3 conv + 1bn + 1relu
        self.transpose_conv_1 = Conv2DTranspose(num_channels=num_channels,
                               num_filters= num_filters, filter_size = 2, stride = 2)

        self.conv_1 = Conv2D(num_channels=num_channels,
                       num_filters= num_filters,
                       filter_size = 3,stride=1, padding=0)

        self.bn_1 = BatchNorm(num_channels=num_filters, act = 'relu')

        self.conv_2 = Conv2D(num_channels=num_filters,
                       num_filters= num_filters,
                       filter_size = 3,stride=1, padding=0)

        self.bn_2 = BatchNorm(num_channels=num_filters, act = 'relu')





    def forward(self, inputs_prev, inputs):
        #[1, 512, 64, 64] [1, 1024, 28, 28]
        # TODO: forward contains an Pad2d and Concat
        #Pad
        x = self.transpose_conv_1(inputs)
        new_inputs_prev = paddle.fluid.layers.crop(inputs_prev, shape = [x.shape[0], x.shape[1],x.shape[2], x.shape[3]])
        x = fluid.layers.concat(input = [new_inputs_prev, x], axis = 1) # N C H W
        
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        

        return x


class UNet(Layer):
    def __init__(self, num_classes=59):
        super(UNet, self).__init__()
        # encoder: 3->64->128->256->512
        # mid: 512->1024->1024

        #TODO: 4 encoders, 4 decoders, and mid layers contains 2 1x1conv+bn+relu
        self.down1 = Encoder(num_channels=3, num_filters=64)
        self.down2 = Encoder(num_channels=64, num_filters=128)
        self.down3 = Encoder(num_channels=128, num_filters=256)
        self.down4 = Encoder(num_channels=256, num_filters=512)

        self.mid_conv1 = Conv2D(num_channels=512,
                       num_filters= 1024,
                       filter_size = 3,stride=1, padding=0)

        self.mid_bn1 = BatchNorm(num_channels=1024, act = 'relu')

        self.mid_conv2 = Conv2D(num_channels=1024,
                       num_filters= 1024,
                       filter_size = 3,stride=1, padding=0)

        self.mid_bn2 = BatchNorm(num_channels=1024, act = 'relu')

        self.up4 = Decoder(num_channels=1024, num_filters=512)
        self.up3 = Decoder(num_channels=512, num_filters=256)
        self.up2 = Decoder(num_channels=256, num_filters=128)
        self.up1 = Decoder(num_channels=128, num_filters=64)

        self.last_conv = Conv2D(num_channels=64,
                       num_filters= num_classes,
                       filter_size = 1,stride=1, padding=0)


    def forward(self, inputs):
        x1, x = self.down1(inputs) #x1 has not been pooled, x has been pooled
        print(x1.shape, x.shape)
        x2, x = self.down2(x)
        print(x2.shape, x.shape)
        x3, x = self.down3(x)
        print(x3.shape, x.shape)
        x4, x = self.down4(x)
        print("x4.shape:",x4.shape, x.shape) #[1, 512, 64, 64] [1, 512, 32, 32]

        # middle layers
        x = self.mid_conv1(x)
        x = self.mid_bn1(x)
        x = self.mid_conv2(x)
        x = self.mid_bn2(x)

        print(x4.shape, x.shape)#[1, 512, 64, 64] [1, 1024, 28, 28]
        x = self.up4(x4, x)
        print(x3.shape, x.shape)
        x = self.up3(x3, x)
        print(x2.shape, x.shape)
        x = self.up2(x2, x)
        print(x1.shape, x.shape)
        x = self.up1(x1, x)
        print(x.shape)

        x = self.last_conv(x)

        return x


def main():
    with fluid.dygraph.guard(fluid.CUDAPlace(0)):

        model = UNet(num_classes=59)
        x_data = np.random.rand(1, 3, 572, 572).astype(np.float32)
        inputs = to_variable(x_data)
        pred = model(inputs)

        print(pred.shape)


if __name__ == "__main__":
    main()


















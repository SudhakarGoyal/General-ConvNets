from keras.layers import Conv2D, Dense, MaxPool2D, BatchNormalization, Activation, Input, Flatten, add, GlobalAveragePooling2D, ZeroPadding2D, DepthwiseConv2D
from keras.models import Model

class MobileNet:
    def __init__(self, input_, num_classes, depth=1):
        self.depth = depth
        self.input_ = input_
        self.classes = num_classes

    def convolution(self, input_, filters_, strides, depth_multiplier):

        if strides == (1, 1):
            x = input_
        else:
            x = ZeroPadding2D((1, 1))(input_)
        x = DepthwiseConv2D((3, 3), strides=strides, padding='same' if strides == (1, 1) else 'valid',
                            depth_multiplier=depth_multiplier)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters_, (1, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def fully_connected(self, x):

        x = GlobalAveragePooling2D()(x)
        x = Dense(1024)(x)
        output = Dense(self.classes, activation='softmax')(x)
        return output

    def forward(self):
        x = Conv2D(32, (3, 3), strides=2)(self.input_)
        x = self.convolution(x, 64, (1, 1), 1)
        x = self.convolution(x, 128, (2, 2), 1)
        x = self.convolution(x, 128, (1, 1), 1)
        x = self.convolution(x, 256, (2, 2), 1)
        x = self.convolution(x, 256, (1, 1), 1)
        x = self.convolution(x, 512, (2, 2), 1)
        for i in range(5):
            x = self.convolution(x, 512, (1, 1), 1)
        x = self.convolution(x, 1024, (2, 2), 1)
        x = self.convolution(x, 1024, (2, 2), 1)
        output = self.fully_connected(x)

        model = Model(self.input_, output)
        print(model.summary())
        return model


# mobilenet_obj = MobileNet(Input(shape=(64, 64, 3)), 10)
# mobilenet_obj.forward()


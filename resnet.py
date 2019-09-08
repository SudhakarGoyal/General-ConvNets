from keras.layers import Conv2D, Dense, MaxPool2D, BatchNormalization, Activation, Input, Flatten, add, GlobalAveragePooling2D, ZeroPadding2D, DepthwiseConv2D
from keras.models import Model


class ResNet:
    def __init__(self, num_classes, layers, input_):
        self.classes = num_classes
        self.layers = layers
        self.input_ = input_

    def identity(self, input_feature, filters, strides_=1):

        filter_1, filter_2, filter_3 = filters
        x = Conv2D(filter_1, (1, 1))(input_feature)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filter_2, (3, 3), padding='same', strides=strides_)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filter_3, (1, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def conv_layers(self, input_feature, filters, strides_=1):

        filter_1, filter_2, filter_3 = filters
        x = Conv2D(filter_1, (1, 1))(input_feature)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filter_2, (3, 3), padding='same', strides=strides_)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filter_3, (1, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        output = add([x, input_feature])
        return x

    def convolution(self, x):
        total_layers = 0

        x = self.identity(x, [64, 64, 128], strides_=2)
        total_layers += 3

        for i in range(2):
            x = self.conv_layers(x, [64, 64, 128])
            total_layers += 3

        x = self.identity(x, [128, 128, 512], strides_=2)
        total_layers += 3

        if self.layers == 152:
            for i in range(7):
                x = self.conv_layers(x, [128, 128, 512])
                total_layers += 3
        else:
            for i in range(3):
                x = self.conv_layers(x, [128, 128, 512])
                total_layers += 3

        total_layers += 9 # for the last 3 conv layers that use 2048 as last filter size
        #print((self.layers - total_layers)//3 - 1)

        x = self.identity(x, [256, 256, 1024], strides_=2)
        for i in range((self.layers - total_layers)//3 - 1):
            x = self.conv_layers(x, [256, 256, 1024])

        x = self.identity(x, [512, 512, 2048], strides_=2)
        for i in range(2):
            x = self.conv_layers(x, [512, 512, 2048])

        return x

    def fully_connected(self, x):

        x = GlobalAveragePooling2D()(x)
        output = Dense(self.classes, activation='softmax')(x)

        return output

    def forward(self):
        x = ZeroPadding2D((3, 3))(self.input_)
        x = Conv2D(64, (7, 7), strides=2)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D((3, 3))(x)
        x = MaxPool2D((3, 3), strides=2)(x)

        x = self.convolution(x)
        output = self.fully_connected(x)

        model = Model(self.input_, output)

        print(model.summary())

        return model

# resnet_obj = ResNet(10, 50, Input(shape=(224, 224, 3)))
# resnet_obj.forward()

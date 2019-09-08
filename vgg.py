from keras.layers import Conv2D, Dense, MaxPool2D, BatchNormalization, Activation, Input, Flatten, add, GlobalAveragePooling2D, ZeroPadding2D, DepthwiseConv2D
from keras.models import Model


class VggNet:
    def __init__(self, num_classes, input_, layers=16):
        self.classes = num_classes
        self.layers = layers
        self.input_ = input_ #Input(shape=(224, 224, 3))

    def convolution(self):
        x = Conv2D(64, (3, 3), padding='same')(self.input_)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPool2D((2, 2))(x)

        x = Conv2D(128, (3, 3), padding='same')(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPool2D((2, 2))(x)

        x = Conv2D(256, (3, 3), padding='same')(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        if self.layers == 19:
            x = Conv2D(256, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPool2D((2, 2))(x)

        x = Conv2D(512, (3, 3), padding='same')(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        if self.layers == 19:
            x = Conv2D(512, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPool2D((2, 2))(x)

        x = Conv2D(512, (3, 3), padding='same')(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        if self.layers == 19:
            x = Conv2D(512, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPool2D((2, 2))(x)

        x = Flatten()(x)
        return x

    def fully_connected(self, x):
        x = Dense(4096)(x)
        x = Dense(4096)(x)
        output = Dense(self.classes, activation='softmax')(x)

        return output

    def forward(self):

        x = self.convolution()
        output = self.fully_connected(x)
        model = Model(self.input_, output)
        print(model.summary())

        return model


#vgg_obj = VggNet(10, Input(shape=(224, 224, 1)), 16)
#vgg_obj.forward()
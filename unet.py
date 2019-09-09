from keras.layers import Conv2D, Dense, MaxPool2D, BatchNormalization, Activation, Input, \
    Flatten, add, GlobalAveragePooling2D, ZeroPadding2D, DepthwiseConv2D, UpSampling2D, MaxPooling2D, concatenate
from keras.models import Model


class Unet:
    def __init__(self, num_classes, size):
        self.classes = num_classes
        self.input = size

    def convolution(self, input_):

        conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_)
        conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_1)
        pool1_1 = MaxPooling2D((2, 2))(conv1_2)

        conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1_1)
        conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2_1)
        pool2_1 = MaxPooling2D((2, 2))(conv2_2)

        conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2_1)
        conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3_1)
        pool3_1 = MaxPooling2D((2, 2))(conv3_2)

        conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3_1)
        conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4_1)
        pool4_1 = MaxPooling2D((2, 2))(conv4_2)

        conv5_1 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4_1)
        conv5_2 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5_1)

        deconv1_1 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5_2))
        deconv1_3 = concatenate([deconv1_1, conv4_2], axis=3)
        deconv1_4 = Conv2D(512, (3, 3), activation='relu', padding='same')(deconv1_3)
        deconv1_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(deconv1_4)

        deconv2_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(deconv1_5))
        deconv2_2 = concatenate([deconv2_1, conv3_2], axis=3)
        deconv2_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(deconv2_2)
        deconv2_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(deconv2_3)

        deconv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(deconv2_4))
        deconv3_2 = concatenate([deconv3_1, conv2_2], axis=3)
        deconv3_3 = Conv2D(128, (3, 3), activation='relu', padding='same')(deconv3_2)
        deconv3_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(deconv3_3)

        deconv4_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(deconv3_4))
        deconv4_2 = concatenate([deconv4_1, conv1_2], axis=3)
        deconv4_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(deconv4_2)
        deconv4_4 = Conv2D(64, (3, 3), activation='relu', padding='same')(deconv4_3)

        return deconv4_4

    def forward(self):
        x = self.convolution(self.input)

        if self.classes > 1:
            output_ = Conv2D(self.classes, (1, 1), activation='softmax')(x)   # the number of filters need to be 2
        else:
            output_ = Conv2D(self.classes, (1, 1), activation='sigmoid')(x)

        model = Model(input=self.input, output=output_)
        model.summary()
    
        return model


unet_obj = Unet(2, Input(shape=(512, 512, 3)))

unet_obj.forward()
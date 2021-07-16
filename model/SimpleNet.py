'''
Refers to https://github.com/kobiso/CBAM-keras and https://github.com/xiaochus/MobileNetV3
'''



from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape,MaxPooling2D,Add,AveragePooling2D,concatenate,Dense
from model.model_base import ModelBase


class SimpleNet(ModelBase):
    def __init__(self, shape, n_class, alpha=1.0):
        super(SimpleNet, self).__init__(shape, n_class, alpha)

    def build(self):
        inputs = Input(shape=self.shape)

        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')

        x = self._bottleneck(x, 16, (3, 3), e=16, s=1, cbam=False, nl='RE6')
        # feature fusion
        y1 = AveragePooling2D((2,2),strides=(2,2))(x)
        x = self._bottleneck(x, 24, (3, 3), e=64, s=2, cbam=False, nl='RE6')
        x = concatenate([x,y1],axis=3)

        x = self._bottleneck(x, 40, (3, 3), e=72, s=1, cbam=False, nl='RE6')
        # feature fusion
        y2 = AveragePooling2D((2,2),strides=(2,2))(x)
        x = self._bottleneck(x, 40, (5, 5), e=72, s=2, cbam=True, nl='RE6')
        x = concatenate([x, y2], axis=3)

        x = self._bottleneck(x, 80, (5, 5), e=120, s=1, cbam=True, nl='RE6')
        x = self._bottleneck(x, 80, (5, 5), e=120, s=1, cbam=True, nl='RE6')
        # feature fusion
        y3 = AveragePooling2D((2,2),strides=(2,2))(x)
        x = self._bottleneck(x, 80, (3, 3), e=240, s=2, cbam=False, nl='HS')
        x = concatenate([x, y3], axis=3)

        x = self._conv_block(x, 320, (3, 3), strides=(1, 1), nl='HS')
        x = AveragePooling2D((2,2),strides=(2,2))(x)
        x = self._conv_block(x, 960, (1, 1), strides=(1, 1), nl='HS')
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 960))(x)
        # Classifier
        x = Conv2D(1280, (1, 1), padding='same')(x)
        x = self._return_activation(x, 'HS')
        x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)
        x = Reshape((self.n_class,))(x)

        model = Model(inputs, x)
        return model


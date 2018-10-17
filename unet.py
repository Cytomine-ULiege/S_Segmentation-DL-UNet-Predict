import os
import cv2
import numpy as np
from cytomine.models import AttachedFileCollection, Property
from keras import Input, Model
from keras.layers import Convolution2D, LeakyReLU, SpatialDropout2D, AveragePooling2D, UpSampling2D, merge


def load_model(job, download_path, model_filename="weights.hf5"):
    attached_files = AttachedFileCollection(job).fetch()
    if not (0 < len(attached_files) < 2):
        raise ValueError("More or less than 1 file attached to the Job (found {} file(s)).".format(len(attached_files)))
    attached_file = attached_files[0]
    if attached_file.filename != model_filename:
        raise ValueError(
            "Expected model file name is '{}' (found: '{}').".format(model_filename, attached_file.filename))
    model_path = os.path.join(download_path, model_filename)
    attached_file.download(model_path)
    return model_path


def load_property(job, property_name):
    property = Property(job, key=property_name).fetch()
    return property.value


def load_data(cj, dims, path, **monitor_params):
    images = sorted(os.listdir(path))  # to make sure that several calls return the same list
    imgs = np.ndarray([len(images), dims[0], dims[1], dims[2]], dtype=np.float32)
    for i, image_name in cj.monitor(enumerate(images), **monitor_params):
        img = cv2.imread(os.path.join(path, image_name))
        imgs[i, :, :, :] = cv2.resize(img, (dims[1], dims[0]))
    return images


def create_unet(dims):
    input = Input(shape=(dims[0], dims[1], dims[2]))

    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(input)
    conv1 = LeakyReLU()(conv1)
    conv1 = SpatialDropout2D(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = SpatialDropout2D(0.2)(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = SpatialDropout2D(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = SpatialDropout2D(0.2)(conv2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = SpatialDropout2D(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = SpatialDropout2D(0.2)(conv3)

    comb1 = merge([conv2, UpSampling2D(size=(2, 2))(conv3)], mode='concat', concat_axis=1)
    conv4 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(comb1)
    conv4 = LeakyReLU()(conv4)
    conv4 = SpatialDropout2D(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = SpatialDropout2D(0.2)(conv4)

    comb2 = merge([conv1, UpSampling2D(size=(2, 2))(conv4)], mode='concat', concat_axis=1)
    conv5 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(comb2)
    conv5 = LeakyReLU()(conv5)
    conv5 = SpatialDropout2D(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = SpatialDropout2D(0.2)(conv5)

    output = Convolution2D(1, 1, 1, activation='sigmoid')(conv5)

    model = Model(input=input, output=output)
    return model



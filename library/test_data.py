#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import pandas as pd
import os
import numpy as np
import tensorflow as tf

def read_test(data_dir, batch_size=10):
    def _read_img(name):
        content = tf.read_file(name)
        image = tf.image.decode_image(content, channels=3)
        image.set_shape((None, None, 3))
        image = tf.cast(image, dtype=tf.float32)
        return image

    def normalize(img):
        img_resized = tf.image.resize_images(img, (256, 256))
        img_normed = tf.image.per_image_standardization(img_resized)

        return img_normed

    names = os.listdir(data_dir)
    full_names = [os.path.join(data_dir, name) for name in names]

    name_dataset = tf.data.Dataset.from_tensor_slices(names)
    img_name_dataset = tf.data.Dataset.from_tensor_slices(full_names)

    image_dataset = img_name_dataset.map(_read_img)
    image_dataset = image_dataset.map(normalize)

    dataset = tf.data.Dataset.zip((name_dataset, image_dataset))

    dataset = dataset.repeat(1)

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    name, image = iterator.get_next()

    return name, image


def predict_result(test_img, test_names, model_path, model_fn):
    img_id = list()
    prob_result = list()

    test_out = model_fn(test_img, 10, is_training=False)

    test_prob = tf.nn.softmax(test_out)
    test_pred = tf.argmax(test_prob, axis=-1)
    test_onehot = tf.one_hot(test_pred, 10)

    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, model_path)

    try:
        ind = 0
        while True:
            name, pred = sess.run([test_names, test_onehot])
            img_id.append(name)
            prob_result.append(pred)
            ind += pred.shape[0]
            if ind % 1000 == 0:
                print('%d done!' % ind)
    except tf.errors.OutOfRangeError:
        pass

    prob_result = np.concatenate(prob_result, axis=0)
    img_id = np.concatenate(img_id, axis=0)[:, None]
    all_data = np.concatenate((img_id, prob_result), axis=1)
    submission = pd.DataFrame(all_data)

    sess.close()
    return submission
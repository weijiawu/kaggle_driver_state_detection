#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import os
import csv
import tensorflow as tf

def load(image_dir,list_path):
    image_names = []
    image_labels = []

    with open(list_path) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            classname = row['classname']
            image_labels.append(int(classname[-1]))
            image_names.append(os.path.join(image_dir,classname,row['img']))
            print(os.path.join(image_dir,classname,row['img']))


def read(names, labels, batch_size=None, num_epoch=None, shuffle=False, phase='train'):
    def _read_img(name):
        # TODO
        # 给定图像名称tensor, 输出3维浮点值图像
        content = tf.read_file(name)
        image = tf.image.decode_image(content, channels=3)
        image.set_shape((None, None, 3))
        image = tf.cast(image, dtype=tf.float32)

        return image

    def _train_preprocess(img):
        # TODO
        # 对训练集图像预处理
        # 例如resize到固定大小,翻转,调整对比度等等
        img_resized = tf.image.resize_images(img, (256, 256))
        img_normed = tf.image.per_image_standardization(img_resized)

        return img_normed

    def _eval_preprocess(img):
        # TODO
        # 对验证集, 测试集图像预处理
        # 例如resize到固定大小等等
        img_resized = tf.image.resize_images(img, (256, 256))
        img_normed = tf.image.per_image_standardization(img_resized)

        return img_normed

    # TODO
    # 构造图像名称 dataset
    name_dataset = tf.data.Dataset.from_tensor_slices(names)

    # TODO
    # 通过 map 函数调用 _read_img 构造图像 dataset
    image_dataset = name_dataset.map(_read_img)

    if phase == 'train':
        # TODO
        # 通过 map 函数对训练集图像进行处理
        image_dataset = image_dataset.map(_train_preprocess)
    else:
        # TODO
        # 通过 map 函数对验证集,测试集图像进行处理
        image_dataset = image_dataset.map(_eval_preprocess)

    # TODO
    # 构造图像标签 dataset
    label_dataset = tf.data.Dataset.from_tensor_slices(labels)

    # TODO
    # 将图像以及图像标签 dataset 通过 zip 合并成一个 dataset
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

    # TODO
    # 设置 dataset 的 epoch
    dataset = dataset.repeat(num_epoch)

    # TODO
    # 在需要 shuffle 时, 对 dataset 进行 shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    # TODO
    # 在需要进行 batch 时, 对 dataset 进行 batch
    if batch_size is not None:
        dataset = dataset.batch(batch_size)  # 顺序很重要

    # TODO
    # 构造 dataset 的迭代器
    iterator = dataset.make_one_shot_iterator()

    # TODO
    # 获取 dataset 的元素
    image, label = iterator.get_next()

    return image, label
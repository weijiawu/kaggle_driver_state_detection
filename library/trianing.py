#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import tensorflow as tf

# 训练的 epochs 数目
max_epoch = 50
batch_size = 10
NUM_EXAMPLES_OF_TRAIN = 20000
NUM_EXAMPLES_OF_VALID = 2424
import matplotlib.pyplot as plt
import os

def train(train_op, train_loss, train_acc, max_epoch,
          is_training=None, valid_loss=None, valid_acc=None, save_path=None, pretrained_model=None):
    '''
    训练函数.

    Arguments:
      train_op: 训练 op

      train_loss: 作用在训练集上的 loss

      train_acc: 作用在训练集上的 accuracy

      max_epoch: 训练最大步长

      is_training: 使用 BN 层时的 placeholder

      valid_loss: 作用在验证集上的 loss

      valid_acc: 作用在验证集上的 accuracy

      save_path: 希望保存的模型路径

      pretrained_model: 希望使用的与训练模型路径
    '''
    # 开始训练
    freq_print = NUM_EXAMPLES_OF_TRAIN // 10  # 2000

    log_dir = 'log'

    if pretrained_model is not None:
        # TODO
        # 找到所有需要 finetune 的变量
        vars_to_finetune = {}

        for var in tf.model_variables():
            if 'logit' not in var.op.name:
                var_name_in_ckpt = var.op.name.replace('model', 'resnet_v2_50')
                vars_to_finetune[var_name_in_ckpt] = var

        vars_to_init = filter(lambda var: var not in vars_to_finetune.values(), tf.global_variables())

        # TODO
        # 生成一个对上面变量的加载器
        restorer = tf.train.Saver(vars_to_finetune)

    if save_path is not None:
        saver = tf.train.Saver()

    print('sessing....')
    sess = tf.Session()

    graph_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)

    if pretrained_model is not None:
        # TODO
        # 使用加载器恢复变量的数值,并初始化其他变量
        sess.run(tf.variables_initializer(vars_to_init))
        restorer.restore(sess, pretrained_model)
    else:
        # TODO
        # 初始化所有变量
        sess.run(tf.global_variables_initializer())

    curr_epoch = 0
    curr_step = 0
    curr_valid_step = 0

    running_loss = 0
    running_acc = 0

    metric_log = dict()
    metric_log['train_loss'] = list()
    metric_log['train_acc'] = list()
    metric_log['valid_loss'] = list()
    metric_log['valid_acc'] = list()
    print('train....')
    while curr_epoch < max_epoch:
        if is_training is not None:
            # TODO
            # 运行训练 op 前, 对 bool 型占位符 is_training 进行赋值
            train_feed_dict = {is_training: True}

            # TODO
            # 运行训练 op, 同时输出当前训练 batch 上的 loss 和 accuracy
            #             print(curr_step)
            _, batch_loss, batch_acc = sess.run([train_op, train_loss, train_acc], feed_dict=train_feed_dict)

        else:
            # TODO
            # 运行训练 op, 同时输出当前训练 batch 上的 loss 和 accuracy

            _, batch_loss, batch_acc = sess.run([train_op, train_loss, train_acc])

        running_loss += batch_loss
        running_acc += batch_acc

        curr_step += batch_size

        if curr_step // freq_print > (curr_step - batch_size) // freq_print:
            print('[{}]/[{}], train loss: {:.3f}, train acc: {:.3f}'.format(
                curr_step, NUM_EXAMPLES_OF_TRAIN, running_loss / curr_step * batch_size,
                                                  running_acc / curr_step * batch_size))

        if curr_step > NUM_EXAMPLES_OF_TRAIN:
            # 当前 epoch 结束
            curr_epoch += 1

            metric_log['train_loss'].append(running_loss / curr_step * batch_size)
            metric_log['train_acc'].append(running_acc / curr_step * batch_size)

            if (valid_loss is not None and valid_acc is not None):
                running_loss = 0
                running_acc = 0

                if is_training is not None:
                    # 使用 BN
                    # TODO
                    # 计算验证集上所有样本的 loss 和 accuracy, 对 bool 型占位符 is_training 进行赋值
                    eval_feed_dict = {is_training: False}

                    while curr_valid_step < NUM_EXAMPLES_OF_VALID:
                        # TODO
                        # 输出当前验证 batch 上的 loss 和 accuracy

                        batch_loss, batch_acc = sess.run([valid_loss, valid_acc], feed_dict={is_training: False})

                        running_loss += batch_loss
                        running_acc += batch_acc
                        curr_valid_step += batch_size
                else:
                    # 不使用 BN
                    # TODO
                    # 计算验证集上所有样本的 loss 和 accuracy, 对 bool 型占位符 is_training 进行赋值

                    while curr_valid_step < NUM_EXAMPLES_OF_VALID:
                        # TODO
                        # 输出当前验证 batch 上的 loss 和 accuracy
                        batch_loss, batch_acc = sess.run([valid_loss, valid_acc])

                        running_loss += batch_loss
                        running_acc += batch_acc
                        curr_valid_step += batch_size

                metric_log['valid_loss'].append(running_loss / curr_valid_step * batch_size)
                metric_log['valid_acc'].append(running_acc / curr_valid_step * batch_size)

                curr_valid_step = curr_valid_step % NUM_EXAMPLES_OF_VALID

                print_str = 'epoch: {}, train loss: {:.3f}, train acc: {:.3f}, valid loss: {:.3f}, valid acc: {:.3f}'.format(
                    curr_epoch, metric_log['train_loss'][-1], metric_log['train_acc'][-1],
                    metric_log['valid_loss'][-1], metric_log['valid_acc'][-1])

            else:
                print_str = 'epoch: {}, train loss: {:.3f}, train acc: {:.3f}'.format(curr_epoch,
                                                                                      metric_log['train_loss'][-1],
                                                                                      metric_log['train_acc'][-1])

            print(print_str)
            print()

            curr_step = curr_step % NUM_EXAMPLES_OF_TRAIN
            running_loss = 0
            running_acc = 0

    # =======不要修改这里的内容========
    # 保存模型
    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        saved_path = saver.save(sess, '%s/model.ckpt' % save_path)
        print('model saved to %s' % saved_path)

    sess.close()

    # 可视化
    if valid_loss is not None and valid_acc is not None:
        nrows = 2
        ncols = 2
        figsize = (10, 10)
        _, figs = plt.subplots(nrows, ncols, figsize=figsize)
        figs[0, 0].plot(metric_log['train_loss'])
        figs[0, 0].axes.set_xlabel('train loss')
        figs[0, 1].plot(metric_log['train_acc'])
        figs[0, 1].axes.set_xlabel('train acc')
        figs[1, 0].plot(metric_log['valid_loss'])
        figs[1, 0].axes.set_xlabel('valid loss')
        figs[1, 1].plot(metric_log['valid_acc'])
        figs[1, 1].axes.set_xlabel('valid acc')
    else:
        nrows = 1
        ncols = 2
        figsize = (10, 5)
        _, figs = plt.subplots(nrows, ncols, figsize=figsize)
        figs[0].plot(metric_log['train_loss'])
        figs[0].axes.set_xlabel('train loss')
        figs[1].plot(metric_log['train_acc'])
        figs[1].axes.set_xlabel('train acc')
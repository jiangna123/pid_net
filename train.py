# -*- coding: utf-8 -*-
# Author  : jiangna
# 多GPU版本,常规不带edge训练
import argparse
import logging

from keras.utils import multi_gpu_model

from Models import build_model, color_mode
from utils.data import *
from losses import LOSS_FACTORY
from datetime import datetime
from keras.optimizers import *
from metrics import metrics
from utils.utils import get_flops, mk_if_not_exits, ParallelModelCheckpoint
from keras.callbacks import (CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint)
from keras.callbacks import History
from keras.backend.tensorflow_backend import set_session

# logging
logging.basicConfig(level=logging.INFO)

# 设置gpu运行参数
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'  # 使用BFC算法
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
set_session(session)

parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str, default='bc')
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--optimizer_name", type=str, default=1e-5)
parser.add_argument("--model_name", type=str, default="attunet")
parser.add_argument("--dataset_name", type=str, default="mszs_train_256_1")
parser.add_argument("--channels", type=int, default=1)  # 默认单通道
parser.add_argument("--input_width", type=int, default=256)
parser.add_argument("--input_height", type=int, default=256)

parser.add_argument("--exp_name", type=str, default='exp1')
parser.add_argument("--resize_op", type=int, default=1)
parser.add_argument("--train_batch_size", type=int, default=2)
parser.add_argument("--train_save_path", type=str, default="weights/")
parser.add_argument("--resume", type=str, default="")  # 继续训练时加载模型路径，默认值为``，即从头训练。
parser.add_argument("--image_init", type=str, default="divide")
parser.add_argument("--multi_gpus", type=bool, default=True)
parser.add_argument("--n_classes", type=int, default=2)
parser.add_argument("--gpu_count", type=int, default=2)

args = parser.parse_args()
logging.basicConfig(level=logging.INFO)

# 权重保存
train_save_path = os.path.join(args.train_save_path, args.exp_name, args.model_name)
mk_if_not_exits(train_save_path)

patience = 50  # patience：没有提升的轮次，即训练过程中最多容忍多少次没有提升

# 日志保存的路径
log_file_path = 'weights/' + args.exp_name + '/%s/log.csv' % args.model_name

# 模型参数
model_name = args.model_name
optimizer_name = args.optimizer_name
image_init = args.image_init
multi_gpus = args.multi_gpus
gpu_count = args.gpu_count
epochs = args.epochs
load_weights = args.resume
train_batch_size = args.train_batch_size
channels = args.channels

# 数据参数
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
resize_op = args.resize_op
target_size = (input_height, input_width)
loss_func = LOSS_FACTORY[args.loss]

# 数据存储位置
data_root = os.path.join("data", args.dataset_name)
train_images = os.path.join(data_root, "image")
train_segs = os.path.join(data_root, "label")
model_names = os.path.join(train_save_path, 'unet_membrane.hdf5')

logging.info("开始时间:" + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

history = History()
# 设置unet 模型结构
model = build_model(model_name,
                    n_classes,
                    channels,
                    input_height=input_height,
                    input_width=input_width)
# 多GPU
try:
    model = multi_gpu_model(model, gpus=gpu_count)
except Exception as e:
    model = model

model.compile(optimizer=Adam(lr=optimizer_name),
              loss=loss_func,
              metrics=['accuracy', 'precision', 'recall', 'mis_alarm', 'false_alarm', 'iou_score', 'dice_score'])

print(get_flops(model))

model.summary()  # 输出模型各层的参数状况

# 模型回调函数
# tensorboard --logdir=./weights/exp1/unet_l6/log
tb_cb = TensorBoard(log_dir='weights/' + args.exp_name + '/%s/log' % args.model_name,
                    write_images=1,
                    write_graph=1,
                    histogram_freq=0)

"""
EarlyStopping：当监测值不再改善时，该回调函数将中止训练

参数
monitor：需要监视的量，本项目默认：val_loss。通常为：val_acc 或 val_loss 或 acc 或 loss
patience：当early stop被激活（如发现loss相比上patience个epoch训练没有下降），则经过patience个epoch后停止训练。
verbose：信息展示模式
mode：'auto'，'min'，'max'之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
"""
early_stop = EarlyStopping('loss',
                           min_delta=0.1,
                           patience=patience,
                           verbose=1)

"""
ReduceLROnPlateau:定义学习率之后，经过一定epoch迭代之后，模型效果不再提升，
                  该学习率可能已经不再适应该模型。
                  需要在训练过程中缩小学习率，进而提升模型。与EarlyStopping配合使用，会非常方便
参数
monitor：监测的值，可以是accuracy(精确度)，val_loss,val_accuracy 
factor：缩放学习率的值，学习率将以lr = lr*factor的形式被减少
patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
mode：‘auto’，‘min’，‘max’之一 默认‘auto’就行
epsilon：阈值，用来确定是否进入检测值的“平原区”
cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
min_lr：学习率最小值，能缩小到的下限
"""
reduce_lr = ReduceLROnPlateau('loss',
                              factor=0.01,
                              patience=int(patience / 2),
                              verbose=1)
# csv日志
csv_logger = CSVLogger(log_file_path,
                       append=False)

"""
ModelCheckpoint：将在每个epoch后保存模型到filepath

参数
monitor：需要监视的值，通常为：val_acc 或 val_loss 或 acc 或 loss
verbose：信息展示模式，0或1。为1表示输出epoch模型保存信息，默认为0表示不输出该信息，
         信息形如：Epoch 00001: val_acc improved from -inf to 0.49240, saving model to /xxx/checkpoint/model_001-0.3902.h5
save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
mode：'auto'，'min'，'max'之一，在save_best_only=True时决定性能最佳模型的评判准则，
      例如，当监测值为val_acc时，模式应为max，当检测值为val_loss时，模式应为min。
      在auto模式下，评价准则由被监测值的名字自动推断。
save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
period：CheckPoint之间的间隔的epoch数 默认为１
"""

# 统计一下训练集样本数，确定每一个epoch需要训练的iter
images = glob.glob(os.path.join(train_images, "*.jpg")) + \
         glob.glob(os.path.join(train_images, "*.png")) + \
         glob.glob(os.path.join(train_images, "*.jpeg"))

num_train = len(images)
print("train_count:", num_train)

model_checkpoint = ParallelModelCheckpoint(model, model_names,
                                           monitor='loss', verbose=1,
                                           save_best_only=True)

call_backs = [model_checkpoint, csv_logger, early_stop, reduce_lr, tb_cb]

# data generator 数据生成器
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
myGene = trainGenerator(2, data_root,
                        'image', 'label',
                        data_gen_args,
                        image_color_mode=color_mode.get(channels),
                        save_to_dir=None,
                        target_size=target_size)

# 训练
model.fit_generator(myGene, epochs=epochs,
                    steps_per_epoch=int(num_train * 4),
                    callbacks=call_backs)

logging.info("结束时间:" + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

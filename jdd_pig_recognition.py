#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
description: 
author: clzhang
date: 06/12/2017
"""

from numpy import *
from util.utils import *
from vgg.vgg16 import Vgg16

test_type = 'test_B/'
predict_result_path = "JDD_contest/predict_result_5_64_testB_clzhang.csv"
base_path = "JDD_contest/sample5/"
model_path = base_path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)

test_base_path = 'JDD_contest/test/'
test_path = test_base_path + test_type
test_model_base_path = test_base_path + 'models/'
test_model_path = test_model_base_path + test_type

batch_size = 256
epochs = 10

# """1. 以bcolz转换数据并存储"""
def read_image_to_bcolz_array():
    trn_data = get_data(base_path+'train')
    val_data = get_data(base_path+'valid')
    test_data = get_data(test_path)
    # print(test_data.shape) # (3000, 3, 224, 224)
    save_array(model_path+'train_data.bc', trn_data)
    save_array(model_path+'valid_data.bc', val_data)
    save_array(test_model_path+'test_data.bc', test_data)


# """2. vgg训练得到模型特征"""
def train_vgg16_feature():
    # bcolz更快的方式读取数据
    trn_data = load_array(model_path+'train_data.bc')
    val_data = load_array(model_path+'valid_data.bc')
    test_data = load_array(test_model_path+'test_data.bc')

    vgg = Vgg16()
    model = vgg.model
    trn_features = model.predict(trn_data, batch_size=batch_size)
    val_features = model.predict(val_data, batch_size=batch_size)
    test_features = model.predict(test_data, batch_size=batch_size)
    # print(test_features.shape)
    save_array(model_path+'train_lastlayer_features.bc', trn_features)
    save_array(model_path+'valid_lastlayer_features.bc', val_features)
    save_array(test_model_path+'test_lastlayer_features.bc', test_features)


"""3. 利用vgg得到的特征重新训练模型并评估"""
def train_dnn():
    trn_features = load_array(model_path + 'train_lastlayer_features.bc')
    val_features = load_array(model_path + 'valid_lastlayer_features.bc')

    trn_batches = get_batches(base_path + 'train', shuffle=False, batch_size=1)
    val_batches = get_batches(base_path + 'valid', shuffle=False, batch_size=1)
    trn_classes = trn_batches.classes
    val_classes = val_batches.classes
    trn_label = onehot(trn_classes)
    val_label = onehot(val_classes)
    dnn = Sequential(get_bn_layers(0.8))
    dnn.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    dnn.fit(trn_features, trn_label, epochs=epochs, batch_size=batch_size, validation_data=(val_features, val_label))

    # dnn.summary()
    return dnn

def get_bn_layers(p):
    return [
        Dense(512, activation='relu', input_shape=(1000,)),
        BatchNormalization(),
        Dropout(p),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(p / 2),
        Dense(30, activation='softmax')
    ]

"""4. 输出测试集预测结果"""
def predict_and_save(dnn):
    test_features = load_array(test_model_path + 'test_lastlayer_features.bc')
    test_batches = get_batches(test_path, shuffle=False, batch_size=1)

    trn_batches = get_batches(base_path + 'train', shuffle=False, batch_size=1)
    trn_labels = [str(fname.split('/')[0]) for fname in trn_batches.filenames]
    tmp = -1
    label_reindex = []
    for idx in range(0, len(trn_labels)):
        fname = int(trn_labels[idx][3:]) - 1
        if tmp != fname:
            label_reindex.append(fname)
            tmp = fname

    test_probs = dnn.predict(test_features, batch_size=batch_size)
    test_probs = np.array(pd.DataFrame(test_probs).reindex(columns=label_reindex))

    test_fnames = [fname.split('/')[1].split('.')[0] for fname in test_batches.filenames]
    predict_result = []
    for i in range(0, len(test_fnames)):
        for j in range(0, 30):
            predict_result.append([test_fnames[i], j+1, test_probs[i][j]])

    predict_result_df = pd.DataFrame(predict_result)
    predict_result_df.to_csv(predict_result_path, header=None, index=False)

def main():
    # read_image_to_bcolz_array()
    # train_vgg16_feature()
    dnn = train_dnn()
    predict_and_save(dnn)

if __name__ == '__main__':
    main()

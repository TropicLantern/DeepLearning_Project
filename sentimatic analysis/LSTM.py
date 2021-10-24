
import collections
import numpy as np
import pandas as pd
import pynlpir
from keras.utils.np_utils import to_categorical
from keras_preprocessing import sequence
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import layers
import tensorflow as tf
from nlpir import *
from sklearn.model_selection import train_test_split


def load_training_data(path):
    # Ensure that the text before and after processing is in the same path
    if 'train_data' in path:  # 用于确定你的数据集是否在这个位置, 'train_data'是未处理的数据集的名字
        with open(path, 'r', encoding='UTF-8') as f:
            data_text = []
            lines = f.readlines()  # 对于每行进行读取, 尽量保证每一行的数据最多占据txt的一行(在没有自动换行的情况下)
            for line in lines:
                # 下面是原始txt数据集未经过任何处理的格式例子
                # 1 +++$+++ are wtf ... awww thanks !
                data = [line.strip()[9:], int(line[0])]  # line[0]是数据的标签位置, 即对应上面的1, line[9]是数据的文本，即对应上面的are wtf ... awww thanks !
                data_text.append(data)
        return data_text
    else:
        with open(path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x


train_data = load_training_data('E:/pycharm_project/data/train-3.txt')
# print(train_data[0])
# print(type(train_data))


# 自己写的分词
def get_tokenized(data):
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]  #review是文本内容, 例如are wtf ... awww thanks !, 按照空格分词


text = get_tokenized(train_data)
# print("text", text[0])
print(len(text))


# 建立情感词典, 这个词典感觉可以下载一个词典, 然后把它加入进去, 准确率应该会有所提升
def Vocab(counter, min_freq):  # min_freq, 代表最小的词频率
    vocab = {w: freq for w, freq in counter.most_common() if freq >= min_freq}
    return vocab


def get_vocab(data):
    tokenized_data = get_tokenized(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab(counter, min_freq=5)  # 过滤掉词频率小于5的词语


vocab = get_vocab(train_data)
print(vocab)
print(sum(vocab.values()))


# 数据填充, 确保句子数据的长度一致
def preprocess_data(data, vocab):
    max_1 = 100  # 句子最长所能容纳的词语数量, 大于它截断后面, 小于它向后填充
    total = []

    def pad(x):  # 填充具体实施函数
        return x[:max_1] if len(x) > max_1 else x + [0] * (max_1 - len(x))
    tokenized_data = get_tokenized(data)
    for train_data_1 in data:
        for train_data_2 in train_data_1:
            b = []
            if type(train_data_2) != int:
                for tok in train_data_2.split(' '):
                    if tok in vocab.keys():  # 如果分词后的这个词语的是在情感词典里, 用键值替换词语, 否则一律按照1替换
                        c = vocab[tok]
                        b.append(c)
                        # print('bbb', b)
                        d = pad(b)
                        # print('ddd', d)
                    else:
                        b.append(1)
                a = d
                # print('aaa', a)
        total.append(a)
    print('total_type', type(total))
    print('total', total)
    features = np.array(total)
    print('features.shape', features.shape)
    scores = [score for _, score in data]
    labels = np.array(scores)
    print('score', scores)
    print(labels.shape)
    return features, labels


data, label = preprocess_data(train_data, vocab)
print(data)
print(label[2:5])
print('type(label)', type(label))
#  分割数据集
X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.25, random_state=0)
print('Y_train', Y_train)
# 介绍了两种方法
# 方法1
# y_train = pd.get_dummies(Y_train).values
# y_test = pd.get_dummies(Y_test).values
# 方法2
y_train = tf.keras.utils.to_categorical(Y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(Y_test, num_classes=3)

# print('X_train',type(X_train), X_train.shape, X_train)
# print('y_train',type(y_train), y_train.shape, y_train)
print('X_test',type(X_test), X_test.shape, X_test)
print('y_test',type(y_test), y_test.shape, y_test)

# LSTM模型
vocab_size = sum(vocab.values())
embedding_dim = 30
max_length = 100
model = tf.keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    layers.Dropout(0.5),
    layers.LSTM(units=50),
    layers.Dense(units=256, activation='softmax'),
    layers.Dropout(0.5),
    layers.Dense(units=3, activation='softmax')  # 多分类问题使用softmax函数
])
model.summary()


# Configure training process
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00015),
    loss=tf.keras.losses.categorical_crossentropy,  # categorical_crossentrop与categorical_crossentrop()的区别
    metrics=['accuracy']
)
print(X_train[0])
print(X_train[1])
# print(type(y_train))
# print(y_train.shape)
# 开始训练
model.fit(
    X_train,
    y_train,
    epochs=8,
    batch_size=10
)

label = y_train.argmax(axis=1)
print('label',label)
ytrain = y_train.argmax(axis=1)
print('ytrain',ytrain)
## 评估模型
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)
ytest = y_test.argmax(axis=1)
print('ytrain',ytest)
print(type(ytest))
print(y_pred)
print('accuracy %s' % accuracy_score(y_pred, ytest))
target_names = ['消极', '积极', '中性']
print(classification_report(ytest, y_pred, target_names=target_names)) # 出现问题一般是数据集太小了, 数据集划分的时候没有包含所有的种类



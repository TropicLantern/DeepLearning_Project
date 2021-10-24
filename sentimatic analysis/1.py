
import collections
import copy
import numpy as np
import pandas as pd
import pynlpir
from keras.utils.np_utils import to_categorical
from keras_preprocessing import sequence
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import layers
import jieba
import tensorflow as tf
import nltk
from nlpir import *
from sklearn.model_selection import train_test_split


# punctuation
def without_punctuation(src, des):
    txt1 = open(src, 'r', encoding="UTF-8").read()
    punctuation = "`~!@#$%^&*()_-=/<>,.?:;[]|\\{}"
    for ch in punctuation:
        txt1 = txt1.replace(ch, '')
    with open(des, 'w', encoding="UTF-8") as f:
        f.write(txt1)


without_punctuation("E:/pycharm_project/data/train-1.txt", "E:/pycharm_project/data/train-2.txt")


def Clear_StopWords(src, des):
    # stop_words
    txt2 = open(src, 'r', encoding="UTF-8").read()
    stopwords = [line.strip() for line in open('E:/pycharm_project/data/StopWords.txt', encoding="utf-8").readlines()]
    # print(stopwords)
    pynlpir.open()
    sentences = pynlpir.segment(txt2, pos_tagging=False)
    #sentences = jieba.lcut(txt2)
    final_sentences = ''
    for sentence in sentences:
        if sentence not in stopwords:
            final_sentences += sentence
        else:
            final_sentences += ''
    with open(des, 'w', encoding="UTF-8") as f:
        f.write(txt2)
    # print(final_sentences)


Clear_StopWords("E:/pycharm_project/data/train-2.txt", "E:/pycharm_project/data/train-3.txt")


def lowercase_txt(file_name):
    """
    file_name is the full path to the file to be opened
    """
    with open(file_name, 'r+', encoding = "utf8") as f:
        contents = f.read()  # read contents of file
        contents = contents.lower()  # convert to lower case
        f.seek(0, 0)  # position back to start of file
        f.write(contents)
        f.truncate()


lowercase_txt('E:/pycharm_project/data/train-3.txt')


def load_training_data(path):
    # Ensure that the text before and after processing is in the same path
    if 'train_data' or 'test_data' in path:
        with open(path, 'r', encoding='UTF-8') as f:
            data_text = []
            lines = f.readlines()
            for line in lines:
                data = [line.strip()[9:], int(line[0])]
                data_text.append(data)
            lines = [line.strip('\n').split(' ') for line in lines]
        return data_text
    else:
        with open(path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x


train_data = load_training_data('E:/pycharm_project/data/train-3.txt')
# print(train_data[0])
# print(type(train_data))


def get_tokenized(data):
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]


text = get_tokenized(train_data)
# print("text", text[0])
print(len(text))


def Vocab(counter, min_freq):
    vocab = {w: freq for w, freq in counter.most_common() if freq >= min_freq}
    return vocab


def get_vocab(data):
    tokenized_data = get_tokenized(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab(counter, min_freq=5)


vocab = get_vocab(train_data)
print(vocab)
print(sum(vocab.values()))



def preprocess_data(data, vocab):
    max_1 = 100
    total = []

    def pad(x):
        return x[:max_1] if len(x) > max_1 else x + [0] * (max_1 - len(x))
    tokenized_data = get_tokenized(data)
    for train_data_1 in data:
        for train_data_2 in train_data_1:
            b = []
            if type(train_data_2) != int:
                for tok in train_data_2.split(' '):
                    if tok in vocab.keys():
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
X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.25, random_state=0)
print('Y_train', Y_train)
# y_train = pd.get_dummies(Y_train).values
# y_test = pd.get_dummies(Y_test).values
y_train = tf.keras.utils.to_categorical(Y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(Y_test, num_classes=3)
# print('X_train',type(X_train), X_train.shape, X_train)
# print('y_train',type(y_train), y_train.shape, y_train)
print('X_test',type(X_test), X_test.shape, X_test)
print('y_test',type(y_test), y_test.shape, y_test)

vocab_size = sum(vocab.values())
embedding_dim = 30
max_length = 100
model = tf.keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    layers.Dropout(0.5),
    layers.LSTM(units=50),
    layers.Dense(units=256, activation='softmax'),
    layers.Dropout(0.5),
    layers.Dense(units=3, activation='softmax')
])
model.summary()


# Configure training process
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00015),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy']
)
print(X_train[0])
print(X_train[1])
print(type(np.array(data)))
# print(type(y_train))
# print(y_train.shape)
# print('train_set',train_set)
# print(test_set)
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
ytest = np.append(ytest, 2)
ytest = np.append(ytest, 0)
y_pred = np.append(y_pred, 2)
y_pred = np.append(y_pred, 0)
print(y_pred)
print('accuracy %s' % accuracy_score(y_pred, ytest))
target_names = ['negative', 'positive', 'nu']
print(classification_report(ytest, y_pred, target_names=target_names))


print("保存模型")
model.save('model/LSTM_model.h5')



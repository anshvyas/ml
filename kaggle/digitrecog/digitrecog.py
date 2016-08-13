
# single hidden layer feed forward nn
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import csv


def read_file(to_read):
    return np.array(pd.read_csv(to_read))


def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


def init_bias(shape):
    bias = tf.constant(shape, dtype=tf.float32)
    return tf.Variable(bias)


def convert_one_hot_vectors(input_arr):
    num_labels = 10
    # one_hot = np.zeros((input_arr.size, 6))
    one_hot = np.eye(num_labels)[input_arr]
    return one_hot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("test_file")
    args = parser.parse_args()

    # read the input
    readData = read_file(args.train_file)
    print (np.shape(readData))

    # convert into features and labels
    readFeatures = readData[:, 1:]
    readLabels = convert_one_hot_vectors(readData[:, 0])
    no_of_feat = np.shape(readFeatures)[1]
    no_of_labels = 10

    trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(
        readFeatures, readLabels, test_size=0.01, random_state=42)

    print (trainFeatures, trainLabels, testFeatures, testLabels)

    # init the nueral net

    inputLayer = tf.placeholder(tf.float32, [None, no_of_feat])
    # weights matrix from inputlayer to hidden layer
    W_1 = tf.placeholder(tf.float32, [no_of_feat, 100])
    b_1 = init_bias([100])
    # weights and bias from hidden layer  to out layer
    W_2 = tf.placeholder(tf.float32, [100, 10])
    b_2 = init_bias([10])
    # output layer
    outLayer = tf.placeholder(tf.float32, [None, 10])

    # init the weights
    W_1 = init_weights((no_of_feat, 100))
    W_2 = init_weights((100, 10))
    # activations
    a_1 = tf.nn.sigmoid(tf.matmul(inputLayer, W_1) + b_1)
    a_2 = tf.matmul(a_1, W_2) + b_2

    # errors to minimize
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(a_2, outLayer))

    trainer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    predict_op = tf.argmax(a_2, 1)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(300):
        sess.run(trainer, feed_dict={
            inputLayer: trainFeatures, outLayer: trainLabels})
        accuracy = np.mean(np.argmax(testLabels, 1) == sess.run(
            predict_op, feed_dict={inputLayer: testFeatures,
                                   outLayer: testLabels}))
        print ("Epoch: %d Accuracy= %.2f% %" % (i, accuracy * 100))

    testFeatures = read_file(args.test_file)
    prediction = sess.run(predict_op,
                          feed_dict={inputLayer: testFeatures})
    predict_list = []
    for i, item in enumerate(prediction):
        predict_list.append({'ImageId': i + 1, 'Label': item})

    with open('predict.csv', 'wb') as csvfile:
        fields = ['ImageId', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predict_list)

    sess.close()

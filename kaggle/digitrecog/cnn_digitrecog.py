import pandas as pd
import numpy as np
import tensorflow as tf
import csv
import argparse
import random


def read_file(to_read):
    return np.array(pd.read_csv(to_read))


def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.05, dtype=tf.float32)
    return tf.Variable(weights)


def init_bias(shape):
    bias = tf.constant(0.05, shape=shape, dtype=tf.float32)
    return tf.Variable(bias)


def convert_one_hot_vectors(input_arr):
    num_labels = 10
    # one_hot = np.zeros((input_arr.size, 6))
    one_hot = np.eye(num_labels)[input_arr]
    return one_hot


def init_conv_layer(input,  # the previous layer
                    num_channels,  # num of channels in previous layer
                    filter_size,  # size of filter wisth and height
                    num_filters,  # num of filters
                    use_pooling=True  # use max pooling of 2x2 to condense info
                    ):
        # shape of filters
    shape_filter = [filter_size, filter_size, num_channels, num_filters]

    # create weights for filter
    weights = init_weights(shape_filter)

    # create biases for each filter
    biases = init_bias([num_filters])

    # create a convolutional layer on input set of images
    layer = tf.nn.conv2d(input=input, filter=weights,
                         strides=[1, 2, 2, 1], padding='SAME')

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding='SAME')

    # add non-linearity
    layer = tf.nn.tanh(layer)

    return layer, weights, biases

# used to convert 4-d output to 2-d output to serve as input for hidden
# fully connected layer


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    # features are img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()
    # convert in to form of [num_images, num_features]
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def init_fclayers(input, num_input, num_output):
    weights = init_weights([num_input, num_output])
    biases = init_bias([num_output])
    layer = tf.nn.relu(tf.matmul(input, weights) + biases)
    return layer, weights, biases


def shuffle_batch(input, batch_size):
    offset = random.randint(0, (input.shape[0] - batch_size - 1))
    return input[offset:offset + batch_size, :]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("test_file")
    args = parser.parse_args()

    # read the input
    readData = read_file(args.train_file)
    print ("File read with shape:")
    print(np.shape(readData))
    # split the data into test and train
    np.take(readData, np.random.permutation(
        readData.shape[0]), axis=0, out=readData)
    print(readData)
    trainData = readData[0:41800, :]
    testFeatures = readData[41801:, 1:]
    testLabels = convert_one_hot_vectors(readData[41801:, 0])

    # convolutional layer 1
    fitler_size1 = 5
    num_filters1 = 16

    # convolutional layer 2
    fitler_size2 = 5
    num_filters2 = 36

    # fully-connected layer
    fc_layer_size = 400

    # vars for images
    img_size = 28
    # size to store in 1-d array
    img_flat = img_size * img_size
    # tupple containing height and width of each image for reshaping
    img_shape = (img_size, img_size)
    # number of channels for images : 1 for gray-scale
    num_channels = 1
    # number of output classes
    num_classes = 10

    inputLayer = tf.placeholder(tf.float32, [None, img_flat])
    inputconvLayer = tf.reshape(
        inputLayer, [-1, img_size, img_size, num_channels])
    outLayer = tf.placeholder(tf.float32, [None, num_classes])

    # spatailly connected hidden convolutional layers
    convlayer_1, convlayerWeights_1, convlayerbias_1 = init_conv_layer(input=inputconvLayer,
                                                                       num_channels=num_channels,
                                                                       filter_size=fitler_size1,
                                                                       num_filters=num_filters1
                                                                       )
    convLayer_2, convlayerWeights_2, convlayerbias_2 = init_conv_layer(input=convlayer_1,
                                                                       filter_size=fitler_size2,
                                                                       num_filters=num_filters2,
                                                                       num_channels=num_filters1)

    flat_layer, num_features = flatten_layer(convLayer_2)

    fc_layer1, fc_weights1, fc_bias1 = init_fclayers(
        input=flat_layer, num_input=num_features, num_output=fc_layer_size)

    fc_layer2, fc_weights2, fc_bias2 = init_fclayers(
        input=tf.nn.dropout(fc_layer1, 0.5), num_input=fc_layer_size, num_output=10)

    cross_entropy = (tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(fc_layer2, outLayer)) # +
        # 0.0001 * tf.nn.l2_loss(convlayerWeights_1) +
        # 0.0001 * tf.nn.l2_loss(convlayerbias_1) +
        # 0.0001 * tf.nn.l2_loss(convlayerWeights_2) +
        # 0.0001 * tf.nn.l2_loss(convlayerbias_2) +
        # 0.0001 * tf.nn.l2_loss(fc_weights1) +
        # 0.0001 * tf.nn.l2_loss(fc_bias1) +
        # 0.0001 * tf.nn.l2_loss(fc_weights2) +
        # 0.0001 * tf.nn.l2_loss(fc_bias2)
    )

    trainer = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    predict_op = tf.argmax(fc_layer2, dimension=1)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(20000):
        train_batch = shuffle_batch(trainData, 100)
        trainFeatures = train_batch[:, 1:]
        trainLabels = convert_one_hot_vectors(train_batch[:, 0])
        sess.run(trainer, feed_dict={
            inputLayer: trainFeatures, outLayer: trainLabels})
        accuracy = np.mean(np.argmax(testLabels, 1) == sess.run(
            predict_op, feed_dict={inputLayer: testFeatures,
                                   outLayer: testLabels}))
        print ("Epoch: %d Accuracy= %.4f% %" % (i, accuracy * 100))

    testFeatures = read_file(args.test_file)
    prediction = sess.run(predict_op,
                          feed_dict={inputLayer: testFeatures})
    predict_list = []
    for i, item in enumerate(prediction):
        predict_list.append({'ImageId': i + 1, 'Label': item})

    with open('predict_cnn.csv', 'wb') as csvfile:
        fields = ['ImageId', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predict_list)

    sess.close()

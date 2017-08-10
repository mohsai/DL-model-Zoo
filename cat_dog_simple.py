import numpy as np
import cv2
import os
import tensorflow as tf
from random import shuffle
import time

# Path: train and test directories
TRAIN_DIR = 'data/train/'
TEST_DIR = 'data/test/'

# image size for training and testing
IMG_SIZE = 64
CHANNELS = 1
DEPTH1 = 32
DEPTH2 = 64
DEPTH3 = 64
DEPTH4 = 128
convolved_img = 4096
FC1_UNITS = 256  # conolved_image * conolved_image * depth4
fc2_units = 512
BETA = 0.001
NUM_STEPS = 8000 
validate = NUM_STEPS // 100

IMG_SIZE = 64
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1

NUM_LABELS = 2
PATCH_SIZE = 3

TEST_SAMP = 10000

KEEP_RATE = 0.8


# Generate test and train data
def  genrate_data():
    print('data is generated...')

    def label_img(img):
        label_name = img.split('.')[-3]

        if label_name == 'cat':
            label = [1, 0]
        elif label_name == 'dog':
            label = [0, 1]
        else:
            print('no label for the image:', img)

        return label
    def process_train_data():
        training_data = []

        for img in os.listdir(TRAIN_DIR):
            label = label_img(img)
            img = cv2.resize(cv2.imread(os.path.join(TRAIN_DIR, img), cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
        shuffle(training_data)
        np.save('train_data.npy', training_data)
        return training_data

    def process_test_data():
        testing_data = []
        for img in os.listdir(TEST_DIR):
            path = os.path.join(TEST_DIR, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            testing_data.append([np.array(img), img_num])

        shuffle(testing_data)
        np.save('test_data.npy', testing_data)
        return testing_data

    train_data = process_train_data()
    test_data = process_test_data()

    return train_data, test_data

def prep_data(train_data, test_data):
    train_dataset = train_data[:-5000]
    val_dataset = train_data[-5000:]

    X_train = np.array([smp[0] for smp in train_dataset]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_train = X_train / 255
    Y_train = np.array([smp[1] for smp in train_dataset])

    X_val = np.array([smp[0] for smp in val_dataset]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_val = X_val / 255
    Y_val = np.array([smp[1] for smp in val_dataset])

    X_test = np.array([smp[0] for smp in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_test = X_test / 255

    # Only 5000 images are being used for training

    X_train = X_train[:3000]
    Y_train = Y_train[:3000]

    return X_train, Y_train, X_val, Y_val


KEEP_PROB = tf.placeholder(tf.float32)
tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE, IMG_SIZE, CHANNELS))
tf_train_labels = tf.placeholder(tf.int64, shape=(BATCH_SIZE, NUM_LABELS))

tf_val_dataset = tf.placeholder(tf.float32, shape=(2 * BATCH_SIZE, IMG_SIZE, IMG_SIZE, CHANNELS))
tf_val_labels = tf.placeholder(tf.int64, shape=(2 * BATCH_SIZE, NUM_LABELS))

tf_test_dataset = tf.placeholder(tf.float32, shape=(TEST_BATCH_SIZE, IMG_SIZE, IMG_SIZE, CHANNELS))
# variables===========

l1_conv_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, CHANNELS, DEPTH1], stddev=0.1))
l1_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH1]))

l2_conv_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, DEPTH1, DEPTH2], stddev=0.1))
l2_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH2]))

l3_conv_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, DEPTH2, DEPTH3], stddev=0.1))
l3_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH3]))

l_fc_weights = tf.Variable(tf.truncated_normal([convolved_img, FC1_UNITS], stddev=0.1))
l_fc_biases = tf.Variable(tf.constant(1.0, shape=[FC1_UNITS]))

ol_weights = tf.Variable(tf.truncated_normal([FC1_UNITS, NUM_LABELS], stddev=0.1))

ol_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))


def model(data):
    conv_1 = tf.nn.conv2d(data, l1_conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    conv_1 = tf.nn.relu(conv_1 + l1_biases)
    pool_1 = tf.nn.avg_pool(conv_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    #         norm_1 = tf.nn.local_response_normalization(pool_1)

    # Second Convolutional Layer with Pooling
    conv_2 = tf.nn.conv2d(pool_1, l2_conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    conv_2 = tf.nn.relu(conv_2 + l2_biases)
    pool_2 = tf.nn.avg_pool(conv_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    #         norm_2 = tf.nn.local_response_normalization(pool_2)
    conv_3 = tf.nn.conv2d(pool_2, l3_conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    conv_3 = tf.nn.relu(conv_3 + l3_biases)
    pool_3 = tf.nn.avg_pool(conv_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    shape = pool_3.get_shape().as_list()

    #         print('shape:',shape[3])

    shape = tf.reshape(pool_3, [shape[0], shape[1] * shape[2] * shape[3]])

    fc1 = tf.add(tf.matmul(shape, l_fc_weights), l_fc_biases)
    fc1_do = tf.nn.dropout(fc1, KEEP_RATE)
    fc_1_do = tf.nn.relu(fc1_do)

    logit = (tf.matmul(fc_1_do, ol_weights) + ol_biases)

    return logit

logit = model(tf_train_dataset)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logit))
optimizer = tf.train.AdagradOptimizer(0.05).minimize(loss)

train_prediction = tf.nn.softmax(model(tf_train_dataset))
valid_prediction = tf.nn.softmax(model(tf_val_dataset))
test_prediction = tf.nn.softmax(model(tf_test_dataset))
# saver = tf.train.Saver()

def train_net():
    train_data,test_data = genrate_data()

    X_train, Y_train, X_val, Y_val = prep_data(train_data, test_data)

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        print('Initialized')
        start = time.time()

        for step in range(NUM_STEPS):

            offset = (step * BATCH_SIZE) % (Y_train.shape[0] - BATCH_SIZE)
            batch_data = X_train[offset:(offset + BATCH_SIZE), :, :, :]
            batch_labels = Y_train[offset:(offset + BATCH_SIZE), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)

            if (step % validate == 0):

                print('# of step:', step)
                print('Minibatch loss at step %d: %.2f' % (step, l))
                print('training accuracy %.1f' % accuracy(predictions, batch_labels))


                offset = np.random.randint(0, (130 - 2 * BATCH_SIZE))
                batch_data = X_val[offset:(offset + 2 * BATCH_SIZE), :, :, :]
                batch_labels = Y_val[offset:(offset + 2 * BATCH_SIZE), :]
                feed_dict = {tf_val_dataset: batch_data, tf_val_labels: batch_labels}

                print(
                    'Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(feed_dict={tf_val_dataset: batch_data}),
                                                             batch_labels))
                print('Time for completing %d steps: %.4s minutes' % (validate, (time.time() - start) / 60))
                start = time.time()
                print('=' * 20)


        print('Model Saved')


def accuracy(predictions, labels):
    correct_prediction = (np.argmax(predictions, 1) == np.argmax(labels, 1))

    acc = 100 * (np.sum(correct_prediction)) / labels.shape[0]

    return acc


if __name__ == "__main__":

    train_net()

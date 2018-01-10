import random
import math
import tensorflow as tf
import numpy as np

def random_sum_pairs(n_examples, n_numbers, largest):
    X, y = list(), list()
    for i in range(n_examples):
        in_pattern = [random.randint(1,largest) for _ in range(n_numbers)]
        out_pattern = sum(in_pattern)
        X.append(in_pattern)
        y.append(out_pattern)
    return X, y

def to_string(X, y, n_numbers, largest):
    # input
    max_length = n_numbers * math.ceil(math.log10(largest+1)) + n_numbers - 1
    Xstr = list()
    for pattern in X:
        strp = '+'.join([str(n) for n in pattern])
        # add padding to the left
        strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
        Xstr.append(strp)
    
    # output
    max_length = math.ceil(math.log10(n_numbers * (largest + 1)))
    ystr = list()
    for pattern in y:
        strp = str(pattern)
        strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
        ystr.append(strp)
        
    return Xstr, ystr

def integer_encode(X, y, alphabet):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
    yenc = list()
    
    for pattern in y:
        integer_encoded = [char_to_int[char] for char in pattern]
        yenc.append(integer_encoded)
    
    return Xenc, yenc

# The example below defines the one_hot_encode() function for binary encoding 
# and demonstrates how to use it.

# one hot encode
def one_hot_encode(X, y, max_int):
    Xenc, yenc = list(), list()
    
    if (X is not None):
        for seq in X:
            pattern = list()
            for index in seq:
                vector = [0 for _ in range(max_int)]
                vector[index] = 1
                pattern.append(vector)
            Xenc.append(pattern)
    
    if (y is not None):
        for seq in y:
            pattern = list()
            for index in seq:
                vector = [0 for _ in range(max_int)]
                vector[index] = 1
                pattern.append(vector)
            yenc.append(pattern)
    
    return Xenc, yenc

def generate_data(n_samples, n_numbers, largest, alphabet):
    # generate pairs
    X, y = random_sum_pairs(n_samples, n_numbers, largest)
    # convert to strings
    X, y = to_string(X, y, n_numbers, largest)
    # integer encode
    X, y = integer_encode(X, y, alphabet)
    # one hot encode
    X, y = one_hot_encode(X, y, len(alphabet))
    # return as numpy as arrays
    X, y = np.array(X), np.array(y)
    return X, y

def invert(seq, alphabet):
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    strings = list()
    for pattern in seq:
        string = int_to_char[np.argmax(pattern)]
        strings.append(string)
    return ''.join(strings)

# Model Architecture

n_batch = 10
n_epoch = 30

n_samples = 20
n_numbers = 2
largest = 30

alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', ' ']
n_chars = len(alphabet)
n_in_seq_length = n_numbers * math.ceil(math.log10(largest+1)) + n_numbers - 1
n_out_seq_length = math.ceil(math.log10(n_numbers * (largest+1)))

n_neurons_1 = 100
n_neurons_2 = 50

# tf Graph input
data = tf.placeholder("float", [None, n_in_seq_length, n_chars])
target = tf.placeholder(tf.int64, [None, n_out_seq_length, n_chars])

with tf.variable_scope('lstm1'):
    encoder_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons_1)
    outputs1, states1 = tf.nn.dynamic_rnn(encoder_cell, data, dtype=tf.float32)
    top_hidden_state = states1[1]

# Create multiple copies of encoded output
encoded_output = tf.tile(tf.expand_dims(top_hidden_state, axis=-2), 
                         [1, n_out_seq_length, 1])

# Wrap each encoded output with Dense layer
with tf.variable_scope('lstm2'):
    decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons_2),
        output_size=n_chars)
    decoded_output, decoded_states = tf.nn.dynamic_rnn(
        decoder_cell, encoded_output, dtype=tf.float32)

learning_rate = 0.01

loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
    labels = target,
    logits = decoded_output
))

optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epoch = 10
batch = 100

boolarr = []
# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    boolarr = []
    for epoch in range(n_epoch):
        batch_x, batch_y = generate_data(n_samples, n_numbers, largest, alphabet)
        for iteration in range(batch):
            sess.run(train_op, feed_dict={data: batch_x, target: batch_y})
        
        output = sess.run(decoded_output, feed_dict={data: batch_x, target: batch_y})        
        logits, _ = one_hot_encode(np.argmax(output, axis=2), None, len(alphabet))
        
        # make sure "model" folder is present
        dirpath = './model'
        ckpt = tf.train.get_checkpoint_state(dirpath)
        if (ckpt is None):
            saver.save(sess, dirpath + '/model.ckpt', global_step=epoch)
        
        #  Evaluate
        for i in range(n_samples):
            expected = invert(batch_y[i], alphabet)
            predicted = invert(logits[i], alphabet)
            correct = expected == predicted
            boolarr.append(correct)

    print('Accuracy: ', np.mean(boolarr))
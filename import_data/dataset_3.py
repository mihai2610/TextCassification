import json
import os
import pandas as pd
import tensorflow as tf


def create_dataframe(path):
    filename_queue = tf.train.string_input_producer([path])

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [[1], [1]]
    col1, col2 = tf.decode_csv(
        value, record_defaults=record_defaults)
    print("{}{}{}{}{}}",col1, col2)
    features = tf.stack([col1])
    labels = tf.stack([col2])

    # with tf.Session() as sess:
    #     # Start populating the filename queue.
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #
    #     for i in range(1200):
    #         # Retrieve a single instance:
    #         example, label = sess.run([features, col2])
    #
    #     coord.request_stop()
    #     coord.join(threads)
    return features, labels


def get_data():
    return create_dataframe("../datasets/dataset_2/train.csv")

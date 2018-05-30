import tensorflow as tf
import sys

import os
import shutil
from os import listdir
from os import mkdir
from shutil import copyfile
from os.path import isfile, join

label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]

with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    image_data =  tf.gfile.FastGFile("test.jpg", 'rb').read()

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        #print (node_id)
        print('%s (score = %.5f)' % (human_string, score))

# -*- coding: utf-8 -*-
"""
Created on Mon May 12 14:39:54 2025

@author: 701A
"""

import tensorflow as tf

w = tf.Variable([[1.0,2.0]])
b = tf.Variable([[2.],[3.]])

y = tf.multiply(w,b)

init_op = tf.compat.v1.global_variables_initializer()

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init_op)
    print(sess.run(y))
    
    
import tensorflow as tf 

tf.test.is_built_with_cuda()

tf.test.is_gpu_available()
#coding=utf-8
import  tensorflow as tf
import  numpy as np
def lrelu(x,leak=0.2,name='lrelu'):
	return tf.maximum(x,leak*x)
def conv(input,w_shape,name,phase,use_batchnorm=True,active='relu'):
	with tf.variable_scope(name):
		w=tf.get_variable(name='convw',shape=w_shape,initializer=tf.truncated_normal_initializer(0,0.02))
		b=tf.get_variable(name='convb',shape=[w_shape[-1]],initializer=tf.constant_initializer(0))
		cov=tf.nn.bias_add(tf.nn.conv2d(input,w,(1,2,2,1),'SAME'),b)
		if use_batchnorm:
			with tf.variable_scope('batch_norm'):
				cov=tf.contrib.layers.batch_norm(cov, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope='bn')
		if active=='relu':
			cov=tf.nn.relu(cov)
		elif active=='lrelu':
			cov=lrelu(cov)

	return cov


def deconv(input,w_shape,out_shape,name,phase,use_batchnorm=True,use_relu=True):
	with tf.variable_scope(name):
		w=tf.get_variable(name='deconvw',shape=w_shape,initializer=tf.truncated_normal_initializer(0,0.02))
		b=tf.get_variable(name='deconvb',shape=[w_shape[-2]],initializer=tf.constant_initializer(0))
		cov=tf.nn.bias_add(tf.nn.conv2d_transpose(input,w,out_shape,(1,2,2,1)),b)

		if use_batchnorm:
			with tf.variable_scope('batch_norm'):
				cov=tf.contrib.layers.batch_norm(cov,decay=0.9,updates_collections=None,epsilon=1e-5,scale=True,scope='bn')
		if use_relu:
			cov=tf.nn.relu(cov)
	return  cov
def liner(input,out_size,name):
	with tf.variable_scope(name):
		w=tf.get_variable(name='linerw',shape=(input.get_shape()[-1],out_size),initializer=tf.truncated_normal_initializer(0,0.02))
		b=tf.get_variable(name='linerb',shape=(out_size),initializer=tf.constant_initializer(0))

		output=tf.matmul(input,w)+b
	return  output


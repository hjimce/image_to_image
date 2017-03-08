from __future__ import division
import os
import time

import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
import  cv2

def random_get_batch_data(dataroot,batch_size):
    image_names=os.listdir(dataroot)
    choice=np.random.choice(image_names,batch_size)
    batch_image=[]
    for c in choice:
        image=cv2.imread(os.path.join(dataroot,c))
        image=cv2.resize(image,(256,256))/127.5-1.
        batch_image.append(image)
    return  batch_image

class pix2pix(object):
	def __init__(self, batch_size=1,image_size=256, output_size=256,
				 gf_dim=64, df_dim=64, L1_lambda=100,
				 input_c_dim=3, output_c_dim=3):
		"""

		Args:
			sess: TensorFlow session
			batch_size: The size of batch. Should be specified before training.
			output_size: (optional) The resolution in pixels of the images. [256]
			gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
			df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
			input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
			output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
		"""
		self.is_grayscale = (input_c_dim == 1)
		self.batch_size = batch_size
		self.image_size = image_size
		self.output_size = output_size

		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.input_c_dim = input_c_dim
		self.output_c_dim = output_c_dim

		self.L1_lambda = L1_lambda

		# batch normalization : deals with poor initialization helps gradient flow
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')
		self.d_bn3 = batch_norm(name='d_bn3')

		self.g_bn_e2 = batch_norm(name='g_bn_e2')
		self.g_bn_e3 = batch_norm(name='g_bn_e3')
		self.g_bn_e4 = batch_norm(name='g_bn_e4')
		self.g_bn_e5 = batch_norm(name='g_bn_e5')
		self.g_bn_e6 = batch_norm(name='g_bn_e6')
		self.g_bn_e7 = batch_norm(name='g_bn_e7')
		self.g_bn_e8 = batch_norm(name='g_bn_e8')

		self.g_bn_d1 = batch_norm(name='g_bn_d1')
		self.g_bn_d2 = batch_norm(name='g_bn_d2')
		self.g_bn_d3 = batch_norm(name='g_bn_d3')
		self.g_bn_d4 = batch_norm(name='g_bn_d4')
		self.g_bn_d5 = batch_norm(name='g_bn_d5')
		self.g_bn_d6 = batch_norm(name='g_bn_d6')
		self.g_bn_d7 = batch_norm(name='g_bn_d7')



	def train(self,x,y,batch_size,input_size,use_wgan=False):

		self.real_B = y
		self.real_A = x
		with tf.variable_scope('gnet'):
			self.fake_B = self.generator(self.real_A)

		self.real_AB = tf.concat(3, [self.real_A, self.real_B])
		self.fake_AB = tf.concat(3, [self.real_A, self.fake_B])
		with tf.variable_scope('dnet'):
			_, self.D_logits = self.discriminator(self.real_AB, reuse=False)
		with tf.variable_scope('dnet',reuse=True):
			_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)




		self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D_logits)))
		self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_logits)))
		self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_logits)))# \
						#+ self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))



		self.d_loss = self.d_loss_real + self.d_loss_fake



		t_vars = tf.trainable_variables()

		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		self.saver = tf.train.Saver()
		d_optim = tf.train.AdamOptimizer(0.0002, 0.5) \
						  .minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(0.0002, 0.5) \
						  .minimize(self.g_loss, var_list=self.g_vars)
		clip_updates=None
		return g_optim,d_optim,clip_updates,self.g_loss,self.d_loss
	def test(self,x,batch_size,input_size):
		with tf.variable_scope('gnet',reuse=True):
			fakex = self.generator(x)
		return  fakex



	'''def train(self):


		tf.initialize_all_variables().run()

		for epoch in xrange(10000):
			np_batch_images=random_get_batch_data('./front',5)
			_,d_loss_np = self.sess.run([d_optim,self.d_loss],feed_dict={ self.real_data: np_batch_images })
			# Update G network
			_= self.sess.run([g_optim],feed_dict={ self.real_data: np_batch_images })
			_,g_loss_np = self.sess.run([g_optim, self.g_loss],feed_dict={ self.real_data: np_batch_images })
			if epoch%100==0:
				[sampleimage]=self.sess.run([tf.squeeze(self.fake_B)],feed_dict={self.real_data:random_get_batch_data('./T',5)})
				print sampleimage.shape

				cv2.imwrite('you/'+str(epoch)+'.jpg',((sampleimage[0,:,:,:]+1)*127.5).astype(np.uint8))


			print g_loss_np,d_loss_np'''


	def discriminator(self, image, y=None, reuse=False):
		# image is 256 x 256 x (input_c_dim + output_c_dim)

		h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
		# h0 is (128 x 128 x self.df_dim)
		h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
		# h1 is (64 x 64 x self.df_dim*2)
		h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
		# h2 is (32x 32 x self.df_dim*4)
		h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
		# h3 is (16 x 16 x self.df_dim*8)
		h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

		return tf.nn.sigmoid(h4), h4

	def generator(self, image, y=None):
		s = self.output_size
		s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

		# image is (256 x 256 x input_c_dim)
		e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
		# e1 is (128 x 128 x self.gf_dim)
		e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
		# e2 is (64 x 64 x self.gf_dim*2)
		e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
		# e3 is (32 x 32 x self.gf_dim*4)
		e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
		# e4 is (16 x 16 x self.gf_dim*8)
		e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
		# e5 is (8 x 8 x self.gf_dim*8)
		e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
		# e6 is (4 x 4 x self.gf_dim*8)
		e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
		# e7 is (2 x 2 x self.gf_dim*8)
		e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
		# e8 is (1 x 1 x self.gf_dim*8)

		self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
			[self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
		d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
		d1 = tf.concat(3, [d1, e7])
		# d1 is (2 x 2 x self.gf_dim*8*2)

		self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
			[self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
		d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
		d2 = tf.concat(3, [d2, e6])
		# d2 is (4 x 4 x self.gf_dim*8*2)

		self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
			[self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
		d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
		d3 = tf.concat(3, [d3, e5])
		# d3 is (8 x 8 x self.gf_dim*8*2)

		self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
			[self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
		d4 = self.g_bn_d4(self.d4)
		d4 = tf.concat(3, [d4, e4])
		# d4 is (16 x 16 x self.gf_dim*8*2)

		self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
			[self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
		d5 = self.g_bn_d5(self.d5)
		d5 = tf.concat(3, [d5, e3])
		# d5 is (32 x 32 x self.gf_dim*4*2)

		self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
			[self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
		d6 = self.g_bn_d6(self.d6)
		d6 = tf.concat(3, [d6, e2])
		# d6 is (64 x 64 x self.gf_dim*2*2)

		self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
			[self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
		d7 = self.g_bn_d7(self.d7)
		d7 = tf.concat(3, [d7, e1])
		# d7 is (128 x 128 x self.gf_dim*1*2)

		self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
			[self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
		# d8 is (256 x 256 x output_c_dim)

		return tf.nn.tanh(self.d8)




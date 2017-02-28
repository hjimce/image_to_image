#coding=utf-8
import  tensorflow as tf
import  numpy as np
import  cv2
from util import  conv,deconv,liner
class dcgan(object):
    def __init__(self,input_shape):
        self.input_images=tf.placeholder(tf.float32,input_shape,'inputimage')
    def g_net(self,x,batch_size,phase):
        gf_dim=64
        kenel_size=5
        [s1,s2,s3,s4,s5,s6,s7,s8]=[256,256/2,256/4,256/8,256/16,256/32,256/64,256/128]
        conv1=conv(x,(kenel_size,kenel_size,3,gf_dim),'conv1',phase,use_batchnorm=False,active='lrelu')
        conv2=conv(conv1,(kenel_size,kenel_size,gf_dim,gf_dim*2),'conv2',phase,use_batchnorm=True,active='lrelu')
        conv3=conv(conv2,(kenel_size,kenel_size,gf_dim*2,gf_dim*4),'conv3',phase,use_batchnorm=True,active='lrelu')
        conv4=conv(conv3,(kenel_size,kenel_size,gf_dim*4,gf_dim*8),'conv4',phase,use_batchnorm=True,active='lrelu')
        conv5=conv(conv4,(kenel_size,kenel_size,gf_dim*8,gf_dim*8),'conv5',phase,use_batchnorm=True,active='lrelu')
        conv6=conv(conv5,(kenel_size,kenel_size,gf_dim*8,gf_dim*8),'conv6',phase,use_batchnorm=True,active='lrelu')
        conv7=conv(conv6,(kenel_size,kenel_size,gf_dim*8,gf_dim*8),'conv7',phase,use_batchnorm=True,active='lrelu')
        conv8=conv(conv7,(kenel_size,kenel_size,gf_dim*8,gf_dim*8),'conv8',phase,use_batchnorm=True,active='lrelu')


        decon8=deconv(conv8,(kenel_size,kenel_size,gf_dim*8,gf_dim*8),(batch_size,s8,s8,gf_dim*8),'deconv8',phase,use_batchnorm=False,use_relu=True)
        decon8=tf.concat(3,[tf.nn.dropout(decon8,0.5),conv7])

        decon7=deconv(decon8,(kenel_size,kenel_size,gf_dim*8,decon8.get_shape()[-1]),(batch_size,s7,s7,gf_dim*8),'deconv7',phase,use_batchnorm=False,use_relu=True)
        decon7=tf.concat(3,[tf.nn.dropout(decon7,0.5),conv6])


        decon6=deconv(decon7,(kenel_size,kenel_size,gf_dim*8,decon7.get_shape()[-1]),(batch_size,s6,s6,gf_dim*8),'deconv6',phase,use_batchnorm=False,use_relu=True)
        decon6=tf.concat(3,[tf.nn.dropout(decon6,0.5),conv5])


        decon5=deconv(decon6,(kenel_size,kenel_size,gf_dim*8,decon6.get_shape()[-1]),(batch_size,s5,s5,gf_dim*8),'deconv5',phase,use_batchnorm=True,use_relu=True)
        decon5=tf.concat(3,[decon5,conv4])


        decon4=deconv(decon5,(kenel_size,kenel_size,gf_dim*4,decon5.get_shape()[-1]),(batch_size,s4,s4,gf_dim*4),'deconv4',phase,use_batchnorm=True,use_relu=True)
        decon4=tf.concat(3,[decon4,conv3])

        decon3=deconv(decon4,(kenel_size,kenel_size,gf_dim*2,decon4.get_shape()[-1]),(batch_size,s3,s3,gf_dim*2),'deconv3',phase,use_batchnorm=True,use_relu=True)
        decon3=tf.concat(3,[decon3,conv2])

        decon2=deconv(decon3,(kenel_size,kenel_size,gf_dim,decon3.get_shape()[-1]),(batch_size,s2,s2,gf_dim),'deconv2',phase,use_batchnorm=True,use_relu=True)
        decon2=tf.concat(3,[decon2,conv1])

        decon1=deconv(decon2,(kenel_size,kenel_size,3,decon2.get_shape()[-1]),(batch_size,s1,s1,3),'deconv1',phase,use_batchnorm=False,use_relu=False)


        return  tf.nn.tanh(decon1)
    #条件对抗网络的判别网络输入是：原始输入图片以及生成网络的输出
    def d_net(self,x,batch_size,phase):
        gf_dim=64
        kenel_size=5
        conv1=conv(x,(kenel_size,kenel_size,6,gf_dim),'conv1',phase,use_batchnorm=False,active='lrelu')
        conv2=conv(conv1,(kenel_size,kenel_size,gf_dim,gf_dim*2),'conv2',phase,use_batchnorm=True,active='lrelu')
        conv3=conv(conv2,(kenel_size,kenel_size,gf_dim*2,gf_dim*4),'conv3',phase,use_batchnorm=True,active='lrelu')
        conv4=conv(conv3,(kenel_size,kenel_size,gf_dim*4,gf_dim*8),'conv4',phase,use_batchnorm=True,active='lrelu')

        flatten=tf.reshape(conv4,(batch_size,-1))
        liner1=liner(flatten,1,'liner1')


        return  liner1
    def train(self,x,y,batch_size):
        with tf.variable_scope('g_net'):
            fakex=self.g_net(x,batch_size,True)
        with tf.variable_scope('d_net'):
            negative=self.d_net(tf.concat(3,[fakex,x]),batch_size,True)
        with tf.variable_scope('d_net',reuse=True):
            positive=self.d_net(tf.concat(3,[y,x]),batch_size,True)
        vars=tf.trainable_variables()
        gnetpara=[v for v  in vars if 'g_net' in v.name]
        dnetpara=[v for v in vars if 'd_net' in v.name]

        print [g.name for g in gnetpara ]
        print [d.name for d in dnetpara ]

        fakeloss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(negative,tf.zeros_like(negative)))
        realloss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(positive,tf.ones_like(positive)))
        d_loss=fakeloss+realloss




        g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(negative,tf.ones_like(negative)))#+100*tf.reduce_mean(tf.abs(fakex-y))

        dnet_update=tf.train.AdamOptimizer(0.0002,0.5).minimize(d_loss,var_list=dnetpara)
        gnet_updata=tf.train.AdamOptimizer(0.0002,0.5).minimize(g_loss,var_list=gnetpara)
        self.fakex=fakex

        return gnet_updata,dnet_update,g_loss,d_loss











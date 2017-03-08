#coding=utf-8
import  os
import  numpy as np
import  cv2
import  tensorflow as tf
import  random
from  dcgan import dcgan
from  model import pix2pix
from data_encoder_decoeder import  encode_to_tfrecords,decode_from_tfrecords,get_batch
import time

def random_get_batch_data(dataroot,batch_size):
	image_names=os.listdir(dataroot)
	choice=np.random.choice(image_names,batch_size)
	batch_image=[]
	for c in choice:
		image=cv2.imread(os.path.join(dataroot,c))
		image=cv2.resize(image,(256,256))/127.5-1.
		batch_image.append(image)
	return  batch_image
def train(train_dataroot,test_dataroot,batch_size):
	with tf.device('/cpu:0'):
		train_image=decode_from_tfrecords(train_dataroot)
		train_batch_images=get_batch(train_image,batch_size,256)
		test_image=decode_from_tfrecords(test_dataroot)
		test_batch_images=get_batch(test_image,batch_size,256)







	use_wgan=True
	net=dcgan()
	#net=pix2pix(batch_size,256,256)

	gnet_updata,dnet_update,clip_updates,g_loss,d_loss=net.train(train_batch_images[:,:,:,:3],train_batch_images[:,:,:,3:],batch_size,256,use_wgan)
	fakex=net.test(test_batch_images[:,:,:,:3],batch_size,256)
	init=tf.initialize_all_variables()
	modelpath='model/model.cpkt'
	with tf.Session() as sess:

		sess.run(init)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess,coord=coord)
		'''if os.path.exists(modelpath):
			tf.train.Saver().restore(sess,modelpath)'''

		for i in range(50000):
			if use_wgan==False:
				_,dloss_np=sess.run([dnet_update,d_loss])
				_,gloss_np=sess.run([gnet_updata,g_loss])
				_,gloss_np=sess.run([gnet_updata,g_loss])
			else:
				for j in range(2):
					_,dloss_np=sess.run([dnet_update,d_loss])
					sess.run([clip_updates])
				_,gloss_np=sess.run([gnet_updata,g_loss])



			if i%100==0:
				start = time.clock()
				[sampleimage]=sess.run([fakex])
				end = time.clock()
				print "read: %f s" % (end - start)
				tf.train.Saver(write_version=tf.train.SaverDef.V1).save(sess,modelpath)

				cv2.imwrite('my/'+str(i)+'.jpg',((sampleimage[0,:,:,:]+1)*127.5).astype(np.uint8))
			if i%10==0:
				print i,gloss_np,dloss_np
		coord.request_stop()#queue需要关闭，否则报错
		coord.join(threads)








train_dataroot="data/train.tfrecords"
test_dataroot="data/test.tfrecords"
#encode_to_tfrecords("./train",train_dataroot,(280,280))
#encode_to_tfrecords("./test",test_dataroot,(280,280))

train(train_dataroot,test_dataroot,4)



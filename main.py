import  os
import  numpy as np
import  cv2
import  tensorflow as tf
import  random
from  dcgan import dcgan
from  model import pix2pix
def random_get_batch_data(dataroot,batch_size):
	image_names=os.listdir(dataroot)
	choice=np.random.choice(image_names,batch_size)
	batch_image=[]
	for c in choice:
		image=cv2.imread(os.path.join(dataroot,c))
		image=cv2.resize(image,(256,256))/127.5-1.
		batch_image.append(image)
	return  batch_image
def train(dataroot,batch_size):
	net=dcgan((10,10,10))
	batch_images=tf.placeholder(tf.float32,(batch_size,256,256,3),'batchimage')
	gnet_updata,dnet_update,g_loss,d_loss=net.train(batch_images,batch_images,batch_size)

	init=tf.initialize_all_variables()
	gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
	with tf.Session() as sess:
		'''mod=pix2pix(sess,256,5)
		mod.train()'''
		sess.run(init)

		for i in range(10000):
			np_batch_images=random_get_batch_data(dataroot,batch_size)
			_,dloss_np=sess.run([dnet_update,d_loss],feed_dict={batch_images:np_batch_images})

			_,gloss_np=sess.run([gnet_updata,g_loss],feed_dict={batch_images:np_batch_images})
			_,gloss_np=sess.run([gnet_updata,g_loss],feed_dict={batch_images:np_batch_images})
			if i%100==0:

				[sampleimage]=sess.run([tf.squeeze(net.fakex)],feed_dict={batch_images:random_get_batch_data('./T',5)})
				print sampleimage.shape

				cv2.imwrite('my/'+str(i)+'.jpg',((sampleimage[0,:,:,:]+1)*127.5).astype(np.uint8))
			print gloss_np,dloss_np










train('./front',5)



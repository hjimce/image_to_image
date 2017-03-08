import  tensorflow as tf

import  numpy as np
x = tf.ones((10,10,3))
y = tf.constant([[[0.,1.,0.]]])
z=x*y

sess = tf.Session()
znp,xnp,ynp=sess.run([z,x,y])
znp=np.transpose(znp,(2,0,1))

print xnp.shape,ynp.shape,znp.shape
print znp

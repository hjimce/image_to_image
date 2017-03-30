#coding=utf-8
#write by hjimce-2017-3-30
import tensorflow as tf
from tensorflow.python.framework import  graph_util
from  tensorflow.python.framework import importer
import test
import cv2
#load the pretrain model
def load_model(session,netmodel_path,param_path):

	new_saver = tf.train.import_meta_graph(netmodel_path)
	session.run(tf.global_variables_initializer())
	new_saver.restore(session, param_path)
	x= tf.get_collection('real_A')[0]
	y = tf.get_collection("fake_B")[0]
	return  x,y
#freeze the graph
def freeze_graph(model_root='model/model.cpkt',output_graph='model/freegraph.pb'):
	with tf.Session() as sess:
		x,y=load_model(sess,model_root+'.meta',model_root)


		output_graph_def = graph_util.convert_variables_to_constants(
		sess, tf.get_default_graph().as_graph_def(),[y.name.split(':')[0]])
		print "***********save***********"
		with tf.gfile.GFile(output_graph, "wb") as f:
			f.write(output_graph_def.SerializeToString())
		load_freeze_graph(output_graph,x,y)
#test the freeze result
def load_freeze_graph(graphpb_path,input_tensor,out_tensor):
	with tf.gfile.GFile(graphpb_path, 'rb') as f:
		graph_def_frozen = tf.GraphDef()
		graph_def_frozen.ParseFromString(f.read())
	#fix tensorflow freeze_graph bug
	for node in graph_def_frozen.node:
		if node.op == 'RefSwitch':
			node.op = 'Switch'
			for index in xrange(len(node.input)):
				node.input[index] = node.input[index] + '/read'
		elif node.op == 'AssignSub':
			node.op = 'Sub'
			if 'use_locking' in node.attr: del node.attr['use_locking']

	with tf.Graph().as_default() as graph:
		y,x= tf.import_graph_def(graph_def_frozen,return_elements=[out_tensor.name,input_tensor.name],name='import')
		with tf.Session(graph=graph) as sess:
			origin_image=cv2.imread('server/2.jpg')
			npA,npB=test.server(sess,origin_image,x,y,4)
			cv2.imwrite('npA.jpg',npA)
			cv2.imwrite('npB.jpg',npB)

freeze_graph()
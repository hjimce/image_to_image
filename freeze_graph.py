import tensorflow as tf
from  tensorflow.python.framework import  graph_util

def load_model(session,netmodel_path,param_path):

	new_saver = tf.train.import_meta_graph(netmodel_path)
	new_saver.restore(session, param_path)
	x= tf.get_collection('real_A')[0]
	y = tf.get_collection("fake_B")[0]
	return  x,y
def freeze_graph():
	output_graph='model/freegraph.pb'
	graph = tf.get_default_graph()
	input_graph_def = graph.as_graph_def()
	with tf.Session() as sess:
		model_root='model/model.cpkt'
		x,y=load_model(sess,model_root+'.meta',model_root)
		print y.name
		output_graph_def = graph_util.convert_variables_to_constants(
		sess, input_graph_def, y)
		with tf.gfile.GFile(output_graph, "wb") as f:
			f.write(output_graph_def.SerializeToString())
#def load_freeze():


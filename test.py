#coding=utf-8
import  tensorflow  as tf
import  os
import  cv2
from  data_encoder_decoeder import  load_image
import  numpy as np
def get_batch_data(imagepath):

    image=cv2.imread(imagepath)
    image=cv2.resize(image,(256,256))/127.5-1.
    return  image
def load_model(session,netmodel_path,param_path):

    new_saver = tf.train.import_meta_graph(netmodel_path)
    new_saver.restore(session, param_path)
    x= tf.get_collection('x')[0]#在训练阶段需要调用tf.add_to_collection('test_images',test_images),保存之
    y = tf.get_collection("y")[0]
    return  x,y

def load_images(data_root):
    filename_queue = tf.train.string_input_producer(data_root)
    image_reader = tf.WholeFileReader()
    key,image_file = image_reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file)
    return image, key

def test(data_root,model_root,batch_size):
    image_filenamest=os.listdir(data_root)
    image_filenames=[(data_root+'/'+i) for i in image_filenamest]
    batchs=len(image_filenames)/batch_size



    with tf.Session() as session:

        x,y=load_model(session,model_root+'.meta',model_root)

        print x
        for i in range(batchs):
            imagesf=image_filenames[i*batch_size:(i+1)*batch_size]
            images=[]
            for imgf in imagesf:
                A,B=load_image(imgf)
                image=cv2.resize(A,(256,256))/127.5-1.
                images.append(image)
            images=np.asarray(images,np.float32)

            y_np=session.run(y,feed_dict = {x:images})
            print y_np.shape
            imagesft=image_filenamest[i*batch_size:(i+1)*batch_size]
            for i,imgf in enumerate(imagesft):
                cv2.imwrite('test_result/'+imgf,((y_np[i,:,:,:]+1)*127.5).astype(np.uint8))




test('test','model/model.cpkt',4)

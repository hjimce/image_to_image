
#coding=utf-8
#tensorflow高效数据读取训练
import tensorflow as tf
import cv2
import  random
import os

#把train.txt文件格式，每一行：图片路径名   类别标签
#奖数据打包，转换成tfrecords格式，以便后续高效读取
def load_image(image_path):
    input_img = cv2.imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/2)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w]

    return img_A, img_B
def encode_to_tfrecords(ABdata_root,new_name,resize=None):
    imagefiles=os.listdir(ABdata_root)
    num_example=0
    writer=tf.python_io.TFRecordWriter(new_name)
    for imgf in imagefiles:
        Aimage,Bimage=load_image(os.path.join(ABdata_root,imgf))
        if resize is not None:
            Aimage=cv2.resize(Aimage,resize)#为了
            Bimage=cv2.resize(Bimage,resize)
        height,width,nchannel=Aimage.shape

        example=tf.train.Example(features=tf.train.Features(feature={
            'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            'nchannel':tf.train.Feature(int64_list=tf.train.Int64List(value=[nchannel])),
            'Aimage':tf.train.Feature(bytes_list=tf.train.BytesList(value=[Aimage.tobytes()])),
            'Bimage':tf.train.Feature(bytes_list=tf.train.BytesList(value=[Bimage.tobytes()]))
        }))
        serialized=example.SerializeToString()
        writer.write(serialized)
        num_example+=1
        print num_example


    writer.close()
#读取tfrecords文件
def decode_from_tfrecords(filename,num_epoch=None):
    filename_queue=tf.train.string_input_producer([filename],num_epochs=num_epoch)#因为有的训练数据过于庞大，被分成了很多个文件，所以第一个参数就是文件列表名参数
    reader=tf.TFRecordReader()
    _,serialized=reader.read(filename_queue)
    example=tf.parse_single_example(serialized,features={
        'height':tf.FixedLenFeature([],tf.int64),
        'width':tf.FixedLenFeature([],tf.int64),
        'nchannel':tf.FixedLenFeature([],tf.int64),
        'Aimage':tf.FixedLenFeature([],tf.string),
        'Bimage':tf.FixedLenFeature([],tf.string)
    })

    Aimage=tf.decode_raw(example['Aimage'],tf.uint8)
    Bimage=tf.decode_raw(example['Bimage'],tf.uint8)
    Aimage=tf.reshape(Aimage,tf.pack([
        tf.cast(example['height'], tf.int32),
        tf.cast(example['width'], tf.int32),
        tf.cast(example['nchannel'], tf.int32)]))
    Bimage=tf.reshape(Bimage,tf.pack([
        tf.cast(example['height'], tf.int32),
        tf.cast(example['width'], tf.int32),
        tf.cast(example['nchannel'], tf.int32)]))


    return tf.concat(2,[Aimage,Bimage])
#根据队列流数据格式，解压出一张图片后，输入一张图片，对其做预处理、及样本随机扩充
def get_batch(ABimage, batch_size,crop_size):
    #数据扩充变换
    distorted_image = tf.random_crop(ABimage, [crop_size, crop_size, 6])#随机裁剪
    #distorted_image = tf.image.random_flip_up_down(distorted_image)#上下随机翻转

    #生成batch
    #shuffle_batch的参数：capacity用于定义shuttle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该够大
    #保证数据打的足够乱
    images= tf.train.shuffle_batch([distorted_image],batch_size=batch_size,
                                                 num_threads=4,capacity=1000+3*128,min_after_dequeue=100)


    # 调试显示
    #tf.image_summary('images', images)
    return tf.cast(images,tf.float32)/127.5-1.
#这个是用于测试阶段，使用的get_batch函数
def get_test_batch(ABimage, batch_size,crop_size,ori_size):
        #数据扩充变换
    distorted_image=tf.image.central_crop(ABimage,float(crop_size)/ori_size)
    images=tf.train.batch([distorted_image],num_threads=4,batch_size=batch_size)
    return images




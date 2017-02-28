#coding=utf-8
#对一批训练数据，里面包含多个文件夹，每个文件夹下面存放的是相同类别的物体
# 根据这些文件夹生成列表、切分验证、训练集数据
import os
import shutil
import  random
from facemask import  getmask,getface,getrectimage,get_landmark,getContourStat
import  cv2
import  numpy as np

#因为caffe中,不允许文件名中有空格,所有需要重命名去除空格
def stdrename(imgfiles):
	for l in imgfiles:
		x_list=l.split(' ')
		y = ''.join(x_list)
		if l!=y:
			print 'rename'
			os.rename(l,y)

def GetFileList(FindPath,FlagStr=[]):
	FileList=[]
	FileNames=os.listdir(FindPath)
	if len(FileNames)>0:
		for fn in FileNames:
			if len(FlagStr)>0:
				if IsSubString(FlagStr,fn):
					fullfilename=os.path.join(FindPath,fn)
					FileList.append(fullfilename)
			else:
				fullfilename=os.path.join(FindPath,fn)
				FileList.append(fullfilename)


	if len(FileList)>0:
		FileList.sort()

	return FileList




def spiltdata(path_root,valratio=0.05):
	classify_temp=os.listdir(path_root)
	classify_file=[]
	for c in classify_temp:
		classify_file.append(os.path.join(path_root,c))



	for f in classify_file:
		imgfiles=GetFileList(f)
		stdrename(imgfiles)#caffe 文件名不允许有空格
	for c in classify_temp:
		imgfiles=os.listdir(os.path.join(path_root,c))
		nval=int(len(imgfiles)*valratio)
		print nval
		imgfvals=imgfiles[:nval]
	#验证数据文件列表
		for j in imgfvals:
			if os.path.exists(os.path.join('val',c)) is False:
				os.makedirs(os.path.join('val',c))
			newname=os.path.join('val',c)+'/'+j
			oldname=os.path.join(path_root,c)+'/'+j
			shutil.move(oldname,newname)
	#训练数据文件列表
		imgftrains=imgfiles[nval:]
		for j in imgftrains:
			if os.path.exists(os.path.join('train',c)) is False:
				os.makedirs(os.path.join('train',c))
			newname=os.path.join('train',c)+'/'+j
			oldname=os.path.join(path_root,c)+'/'+j
			shutil.move(oldname,newname)



def writetrainlist(path_root):
	oriimage_root=path_root+'/oriimage'
	classify_temp=os.listdir(oriimage_root)
	strlist=''
	for c in classify_temp:
		labelimage=os.path.join(oriimage_root,c)#原始图片作为类别标签
		croot=os.path.join(path_root,os.basename(c))
		for d in os.listdir(croot):
			trainimage=os.path.join(croot,d)

			strlist+=trainimage+' '+labelimage+'\n'



	txtlist=open(path_root+'.txt','w')
	txtlist.write(strlist)
	txtlist.close()
#无监督训练反卷积网络数据文件制作
def writeunsupervise(images_root):
	tempfile=GetFileList(images_root)
	stdrename(tempfile)
	classify_temp=os.listdir(images_root)
	strlist=''
	for c in classify_temp:
		labelimage=os.path.join(images_root,c)#原始图片作为类别标签
		strlist+=labelimage+' '+labelimage+'\n'#输入等于输出



	txtlist=open(images_root+'.txt','w')
	txtlist.write(strlist)
	txtlist.close()




def getmaskimage(origImage,mask):
	destimage = cv2.bitwise_and(origImage,origImage,mask = mask)
	return  destimage
def readlist(txtlistpath):
	lists=open(txtlistpath)
	sentence=[]
	for l in lists.readlines():
		sentence.append(l)

	random.shuffle(sentence)

	for i,l in enumerate(sentence):
		input,output=l.split()
		mask=getmask(input)
		inputimage=cv2.imread(input)
		outputimage=cv2.imread(output)

		if mask is not  None:
			inputimage=getmaskimage(inputimage,mask)
			outputimage=getmaskimage(outputimage,mask)

		inputnew="../light_for_train/A/"+str(i)+'.jpg'
		outputnew="../light_for_train/B/"+str(i)+'.jpg'
		cv2.imwrite(inputnew,inputimage)
		cv2.imwrite(outputnew,outputimage)

		#shutil.copy(input, inputnew)
		#shutil.copy(output, outputnew)
#对某个目录下的人脸图片文件，进行mask
def facemask(dataroot):
	stdcrop(dataroot)
	images=os.listdir(dataroot)
	for i,l in enumerate(images):
		l=dataroot+'/'+l
		mask=getmask(l)
		inputimage=cv2.imread(l)
		if mask is not  None:
			inputimage=getmaskimage(inputimage,mask)
		'''cv2.imshow("inpout",inputimage)
		cv2.imshow("out",outputimage)
		cv2.waitKey(0)'''
		newroot=dataroot+'crop'
		if os.path.exists(newroot) is False:
			os.makedirs(newroot)
		inputnew=newroot+'/'+str(i)+'.jpg'
		print inputnew

		cv2.imwrite(inputnew,inputimage)
#对某个目录下的文件图片，进行人脸裁剪
def stdcrop(dataroot):

	imgs=os.listdir(dataroot)
	for i in imgs:
		path=dataroot+'/'+i
		F=getface(path)
		if F is None:
			continue
		out=getrectimage(cv2.imread(path),F[0],F[1],F[2],F[3])
		cv2.imwrite(path,out)


#首先进行人脸标准裁剪，覆盖原始图片
def writelightlist(dataroot,origin_dataroot,Apath,Bpath,batchflag):
	oriimagelist=os.listdir(origin_dataroot)
	i=0
	strlist=''
	for o in oriimagelist:
		troot=os.path.join(dataroot,os.path.splitext(o)[0])
		oroot_image=os.path.join(origin_dataroot,o)
		F=getface(oroot_image)
		if F is None:
			continue

		inputimage=getrectimage(cv2.imread(oroot_image),F[0],F[1],F[2],F[3])



		rgbImg = cv2.cvtColor(inputimage, cv2.COLOR_BGR2RGB)
		landmark=get_landmark(rgbImg)
		mask=getContourStat(landmark,inputimage)


		input_maskimage=getmaskimage(inputimage,mask)
		#cv2.imshow('input',input_maskimage)
		#cv2.waitKey(0)

		for root, dirs, files in os.walk(troot, topdown=False):

			for troot_image in files:
				if os.path.splitext(troot_image)[1]!='.jpg':
					continue

				troot_image=os.path.join(root, troot_image)

				outputimage=getrectimage(cv2.imread(troot_image),F[0],F[1],F[2],F[3])
				output_maskimage=getmaskimage(outputimage,mask)




				inputnew=os.path.join(Apath,batchflag+str(i)+'.jpg')
				outputnew=os.path.join(Bpath,batchflag+str(i)+'.jpg')
				im_AB = np.concatenate([output_maskimage, input_maskimage], 1)
				cv2.imwrite(inputnew, im_AB)
				#cv2.imwrite(inputnew,output_maskimage)
				#cv2.imwrite(outputnew,input_maskimage)
				i+=1














#spiltdata('newtrain')
#writetrainlist('../data')
#writeunsupervise('../data/oriimage_train')
#writeunsupervise('../data/001')
#writetrainlist('newtrain_val')
'''writelightlist('b1','b1/origin','A','B','bone')
writelightlist('b2','b2/origin','A','B','btwo')
writelightlist('b3','b3/origin','A','B','bthree')
writelightlist('b4','b4/origin','A','B','bfour')'''
writelightlist('b5','b5/origin','A','B','bfive')

#stdcrop('../unsupervise/ori')#所有训练数据，第一步都要进行人脸的标准裁剪

#facemask("../unsupervise/ori")#只裁剪出人脸部分
#facemask("../data/realtest/mutil-light-ori/0")#只裁剪出人脸部分

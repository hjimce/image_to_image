#coding=utf-8
import numpy as np
import  matplotlib.pyplot as plt
import  cv2
import  dlib
import  os
#根据人脸框bbox，从一张完整图片裁剪出人脸,并保存问文件名cropimgname
#如果未检测到人脸,那么返回false,否则返回true
face_detector=dlib.get_frontal_face_detector()
landmark_predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def getface_cortour(imgpath):
	bgrImg = cv2.imread(imgpath)
	if bgrImg is None:
		return None
	rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
	return get_landmark(rgbImg)
def normalize(v):
	"""

	:rtype : object
	"""
	norm=np.linalg.norm(v, ord=1)
	if norm==0:
		norm=np.finfo(v.dtype).eps
	return v/norm

def get_landmark(rgbimage):

	facesrects = face_detector(rgbimage, 1)
	face_cortour=[]
	if len(facesrects) <=0:
		return None
	facerect=max(facesrects, key=lambda rect: rect.width() * rect.height())

	shape = landmark_predictor(rgbimage, facerect)


	#plt.imshow(rgbimage)
	for i in range(0,17):
		pt=shape.part(i)
		face_cortour.append(np.asarray([pt.x,pt.y]))
		#plt.plot(pt.x,pt.y,'ro')
		#plt.text(pt.x,pt.y,str(i))
	face_cortour=insertpoint(face_cortour)

	height,width,nc=rgbimage.shape

	for i,pt in enumerate(face_cortour):
		if pt[0]<0:
			pt[0]=0
		elif pt[0]>=width:
			pt[0]=width-1
		if pt[1]<0:
			pt[1]=0
		elif pt[1]>=height:
			pt[1]=height-1
		face_cortour[i]=pt
	'''for i in range(len(face_cortour)):
		plt.plot(face_cortour[i][0],face_cortour[i][1],'ro')
		plt.text(face_cortour[i][0],face_cortour[i][1],str(i))
	plt.show()'''



	return  face_cortour
#人脸外轮廓
def insertpoint(face_cortour):
	v=(face_cortour[0]+face_cortour[16])/2.-face_cortour[8]
	nor=normalize(v)
	lenght=np.linalg.norm(v, ord=1)
	center=lenght*1.6*nor+face_cortour[8]


	left1=np.asarray([face_cortour[2][0],2*face_cortour[0][1]-face_cortour[2][1]])
	left2=np.asarray([face_cortour[4][0],2*face_cortour[0][1]-face_cortour[4][1]])


	right1=np.asarray([face_cortour[14][0],2*face_cortour[16][1]-face_cortour[14][1]])
	right2=np.asarray([face_cortour[12][0],2*face_cortour[16][1]-face_cortour[12][1]])



	face_cortour.append(right1)
	#face_cortour.append(right2)

	face_cortour.append(center)

	face_cortour.append(left1)
	#face_cortour.append(left2)
	return face_cortour

def getContourStat(contour,image):
	mask = np.zeros((image.shape[0],image.shape[1]),dtype="uint8")
	print mask.shape
	#print contour.shape
	cv2.drawContours(mask, [np.asarray(contour,dtype=int)],0, 255,-1)

	'''cv2.imshow('mask',mask)
	cv2.waitKey(0)'''
	return mask
#根据人脸框bbox，从一张完整图片裁剪出人脸
def getrectimage(img,miny,maxy,minx,maxx):
    roi=img[miny:maxy,minx:maxx]
    rectshape=roi.shape
    maxlenght=max(rectshape[0],rectshape[1])
    img0=np.zeros((maxlenght,maxlenght,3),np.uint8)
    img0[int(maxlenght*.5-rectshape[0]*.5):int(maxlenght*.5+rectshape[0]*.5),
    int(maxlenght*.5-rectshape[1]*.5):int(maxlenght*.5+rectshape[1]*.5)]=roi
    return  img0
def getface(imgpath):
    bgrImg = cv2.imread(imgpath)
    if bgrImg is None:
        return  None
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    #img = io.imread('1.jpg')
    faces = face_detector(rgbImg, 1)
    if len(faces) <=0:
        return None
    face=max(faces, key=lambda rect: rect.width() * rect.height())
    [x1,x2,y1,y2]=[face.left(),face.right(),face.top(),face.bottom()]
    img = bgrImg
    height, weight =np.shape(img)[:2]
    x=int(x1)
    y=int(y1)
    w=int(x2-x1)
    h=int(y2-y1)
    scale=0.2
    miny=int(max(0,y-scale*h))
    minx=int(max(0,x-scale*w))
    maxy=int(min(height-1,y+(1+scale)*h))
    maxx=int(min(weight-1,x+(1+scale)*w))
    return  [miny,maxy,minx,maxx]

def getmask(imagepath):
	landmark=getface_cortour(imagepath)
	bgrImg = cv2.imread(imagepath)
	if landmark is not None:
		return  getContourStat(landmark,bgrImg)
	else:
		return None

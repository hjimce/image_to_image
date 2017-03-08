import re
import numpy
from matplotlib import pyplot as plt
import os
import  cv2
from facemask import  getmask,getface,getrectimage
import  random
def read_pgm(filename, byteorder='>'):
	print filename
	with open(filename, 'rb') as f:
		buffer = f.read()
	try:
		header, width, height, maxval = re.search(
		b"(^P5\s(?:\s*#.*[\r\n])*"
		b"(\d+)\s(?:\s*#.*[\r\n])*"
		b"(\d+)\s(?:\s*#.*[\r\n])*"
		b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
	except AttributeError:
		raise ValueError("Not a raw PGM file: '%s'" % filename)
	return numpy.frombuffer(buffer,
							dtype='u1' if int(maxval) < 256 else byteorder+'u2',
							count=int(width)*int(height),
							offset=len(header)
							).reshape((int(height), int(width)))
def cropsave(image):
	cv2.imwrite('../light_for_train/A/'+str(i)+'.jpg',image)
	cv2.imread('../light_for_train/A/'+str(i)+'.jpg',)
def makelist(dataroot="../ExtendedYaleB"):
	A=os.listdir(dataroot)
	files=[]
	for f in A:
		f=dataroot+'/'+f
		pgm=os.listdir(f)
		for p in pgm:
			if 'info' in p:
				files.append(f+'/'+p)
	random.shuffle(files)
	c=100000
	for f in files:
		basedir=os.path.dirname(f)
		lines=[]
		with open(f) as fl:
			for i,l in enumerate(fl.readlines()):
				if i==0:
					continue
				lines.append(l.split()[0])
		for l in lines[1:]:
			src=basedir+'/'+lines[0]
			tar=basedir+'/'+l
			srcimg = read_pgm(src, byteorder='<')
			tarimg = read_pgm(tar, byteorder='<')
			oroot='../light_for_train/B/'+str(c)+'.jpg'
			troot='../light_for_train/A/'+str(c)+'.jpg'
			cv2.imwrite(troot,tarimg)
			cv2.imwrite(oroot,srcimg)





			F=getface(oroot)
			if F is None:
				continue
			out=getrectimage(cv2.imread(oroot),F[0],F[1],F[2],F[3])
			cv2.imwrite(oroot,out)
			out=getrectimage(cv2.imread(troot),F[0],F[1],F[2],F[3])
			cv2.imwrite(troot,out)
			c+=1

			print oroot







#read_pgm('../ExtendedYaleB/yaleB39/yaleB39_P07A-020E-10.pgm', byteorder='<')
makelist()
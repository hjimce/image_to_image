import  cv2
import  numpy as np
import  os
def readmask(mask_path,thread=200):
	image=cv2.imread(mask_path)
	image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	ret1,image_input=cv2.threshold(image,thread,1,cv2.THRESH_BINARY)
	ret2,image_output=cv2.threshold(image,10,1,cv2.THRESH_BINARY)

	return  (image_input,image_output)
def masktexture(dataroot,datarootmask,datarootAB):
	if os.path.exists(datarootAB) is False:
		os.makedirs(datarootAB)

	masks=[readmask(datarootmask,i) for i in range(50,220,10)]

	images=os.listdir(dataroot)
	for imagef in images:
		image=cv2.imread(os.path.join(dataroot,imagef))
		print image.shape
		for i,(inputmask,outmask) in enumerate(masks):


			inputimage=image
			outputimage=image
			inputimage=cv2.bitwise_and(image,inputimage,mask = inputmask)
			outputimage=cv2.bitwise_and(image,outputimage,mask = outmask)
			im_AB = np.concatenate([inputimage, outputimage], 1)
			cv2.imwrite(os.path.join(datarootAB,str(i)+imagef), im_AB)

			'''cv2.imshow("image",inputimage)
			cv2.imshow("imageo",outputimage)
			cv2.waitKey(0)'''


masktexture("../texture/test","../texture/mask.jpg","../texture/ABtest")


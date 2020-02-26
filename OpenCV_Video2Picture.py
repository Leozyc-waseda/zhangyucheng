import cv2
vc=cv2.VideoCapture('/home/ogai/Desktop/bdd100k/videos/train/00a0f008-3c67908e.mov' )
c=1
if vc.isOpened():
	rval,frame=vc.read()
else:
	rval=False
while rval:
	rval,frame=vc.read()
	cv2.imwrite('/home/ogai/Desktop/kiit_dataset/cloudy' +str(c)+'.jpg',frame)
	c=c+1
	cv2.waitKey(1)
vc.release()

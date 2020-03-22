import cv2
from model import FacialExpressionModel
import numpy as np
import os

face_clf = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model_fer = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
emotions = ['angry','disgust','fear','happy','sad','surprise','neutral']

##############################################################################
def convert_frames_to_video(frame_array,pathOut,fps):
    #frame_array already ordered
    #for sorting the file names properly

    height, width, layers = frame_array[0].shape
    size = (width,height)
    
    try:
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    except:
        print('no video written')

    for i in range(len(frame_array)):
        
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def process_image(path2file):
	img = cv2.imread(path2file)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_clf.detectMultiScale(gray, 1.3, 5)

	for (x, y, w, h) in faces:
		fc = gray[y:y+h, x:x+w]

		roi = cv2.resize(fc, (48, 48))
		pred = model_fer.predict_all_emotion(roi[np.newaxis, :, :, np.newaxis])
		print(pred)
		ordered = np.argsort(pred)[0]
		print(ordered)
		first, second = ordered[-1], ordered[-2]

		print(first,second)
		if emotions[first] == 'happy' and emotions[second] == 'neutral': pred = 'confident'
		elif emotions[first] in ['fear','angry','sad']: pred = 'nervous'
		elif emotions[first] == 'neutral': pred = 'neutral'
		else: pred = emotions[first]

		cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

	cv2.imwrite(path2file[:-4]+'out.jpg',img)


def process_video(path2file):
	fps = 25
	vidcap = cv2.VideoCapture(path2file)
	frame_array = []
	while True:
		ret,img = vidcap.read()
		if not ret: break
		print(img.shape)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_clf.detectMultiScale(gray, 1.3, 5)

		for (x, y, w, h) in faces:
			fc = gray[y:y+h, x:x+w]

			roi = cv2.resize(fc, (48, 48))
			pred = model_fer.predict_all_emotion(roi[np.newaxis, :, :, np.newaxis])
			print(pred)
			ordered = np.argsort(pred)[0]
			first, second = ordered[-1], ordered[-2]
			if emotions[first] == 'happy' and emotions[second] == 'neutral': pred = 'confident'
			elif emotions[first] in ['fear','angry']: pred = 'nervous'
			elif emotions[first] == 'neutral': pred = 'neutral'
			else: pred = emotions[first]

			cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		frame_array.append(img)

	frame_array.append(img)
	convert_frames_to_video(frame_array,path2file[:-4]+'out.mp4',fps)

def main():
	#process_image('images/w.jpg')
	#process_image('images/z.jpg')
	process_video('edouard_philipe.mp4')
if __name__ == '__main__':
	main()




	
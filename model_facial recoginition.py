#CREATING AND TRAINING MODEL
import cv2
import numpy as np
import webbrowser
from os import listdir
from os.path import isfile,join
#importing datsets file 
data_path='C:/Users/WELCOME/Desktop/python/cvdata/'
onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]
training_data,labels=[],[]
for i,files in enumerate(onlyfiles):
    image_path=data_path+onlyfiles[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)#reading and converting colored images to grayscale 
    training_data.append(np.asarray(images,dtype=np.uint8))
    labels.append(i)
labels=np.asarray(labels,dtype=np.int32)#creating labels into numpy arrrays
#linear binary phase histogram face reoganizer(LBPH) 
model= cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(training_data),np.asarray(labels))
print('model training done!!')
#TESTING MODEL
face_classifier=cv2.CascadeClassifier(r'C:/Users/WELCOME/Desktop/python/haarcascade_frontalface_default.xml')
#DETECTING FACE FROM WEBCAM
def face_detector(img,size=0.5):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize(roi,(200,200))

    return img,roi
#Taking images from webcam
cap=cv2.VideoCapture(0)
counts= 0
counth=0
while True:
    ret,frame=cap.read()

    image,face=face_detector(frame)

    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)#Converting colored images to grayscale
        result=model.predict(face)

        if result[1]<500:
            confidence=int(100*((result[1]-1)/300))
            if((confidence>75):
                cv2.putText(image,"face found",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.imshow('Face Cropper',image)
                counts= counts+1
            
    except:
        cv2.putText(image,"locked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        cv2.imshow('Face Cropper',image)
        pass
    if cv2.waitKey(1)==13:
        break
cap.release()

cv2.destroyAllWindows()
    

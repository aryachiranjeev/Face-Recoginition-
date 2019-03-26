#Creating Datasets
import cv2
import numpy as np
#creating object face_classifier of cv2.cascade clsssifier for importing xml file
face_classifier=cv2.CascadeClassifier(r'C:/Users/WELCOME/Desktop/python/haarcascade_frontalface_default.xml')
#Extracting data using webcam
def face_extractor(img):
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray_img,1.3,5)#(img,scaling,no.of neigbours(3-6))
    #when no face
    if faces is():
        return None
    #if faces(x-cordinate,y-cordinate,w-width,h-height)
    #for changes produced in faces
    for(x,y,w,h) in faces:
        cropped_face=img[y:y+h,x:x+w]

    return cropped_face
#Starting Webcam
cap=cv2.VideoCapture(0)
#count no. of images
count=0
while True:
    ret,frame=cap.read()
    if face_extractor(frame) is not None:
        count=count+1
        face=cv2.resize(face_extractor(frame),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)#Converting colored images to grayscale
        # location to store data images  
        file_name_path='C:/Users/WELCOME/Desktop/python/cvdata/user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('face_cropper',face)

    else:
        print('face not found')
        pass

    if cv2.waitKey(1)==13 or count==1000:
        break

cap.release()
cv2.destroyAllWindows()
print('!!collecting samples complete!!')

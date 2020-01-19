import cv2
import numpy as np
import math

fl = open("model_structure.json",'r')


from keras.models import model_from_json
Modelarch = fl.read()
model = model_from_json(Modelarch) #loadarchitechture
model.load_weights('ASLCNN.h5') #load weights


#Start webcam
cap = cv2.VideoCapture(0)

     
while(1):
    
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
        #define region of interest
        roi=frame[100:300, 100:300]
        
        
        cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
   
        
        #mytest
        mtest=roi
        roi= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #transform to gray scale
 
        roi = cv2.resize(roi, (28,28))

        roi=np.squeeze(roi) #removes extra noise
        roi=roi.reshape((28,28,1))

        print(np.argmax(model.predict([[roi]])))
        #print(frame)
        

        #plt.imshow(mtest)      
        #show the windows
        cv2.imshow('mask',mtest)
        cv2.imshow('frame',frame)
        
        #stops loop on keypress
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    
cv2.destroyAllWindows()
cap.release()    
    


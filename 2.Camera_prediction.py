import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import PIL

fl = open("model_structure.json",'r')


from keras.models import model_from_json
Modelarch = fl.read()
model = model_from_json(Modelarch) #loadarchitechture
model.load_weights('ASLCNN.h5') #load weights

dict = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G", 7:"H", 8:'I', 10:"k", 11:"L", 12:"M", 13:"N", 14:"O", 15:"P", 16:"Q", 17:"R", 18:"S", 19:"T", 20:"U", 21:"V", 22:"W", 23:"X", 24:"Y"}

#Start webcam
cap = cv2.VideoCapture(0)

     
while(1):
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
        #define region of interest
        roi=frame[100:300, 100:300]
        
        
        cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
        #hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        #Test code
    
        #bk = mtest
        mtest=roi
        mtest= cv2.cvtColor(mtest, cv2.COLOR_BGR2GRAY)
        new_im = Image.fromarray(mtest)
        img = new_im.resize((28,28), PIL.Image.ANTIALIAS)
        np_im = np.array(img)
        np_im = np_im.reshape((28,28,1))
        alf=np.argmax(model.predict([[np_im]]))
        print(dict[alf])
        #print(frame)
        cv2.putText(frame,dict[alf], (50, 50) ,  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)
        
        ### mytest
        #mtest=roi
        #roi= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #transform to gray scale
 
#        roi = cv2.resize(roi, (28,28))

 #       roi=np.squeeze(roi) #removes extra noise
  #      roi=roi.reshape((28,28,1))

   #     alf=np.argmax(model.predict([[roi]]))
    #    print(dict[alf])
        #print(frame)
     #   cv2.putText(frame,dict[alf], (50, 50) ,  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)
        

        #plt.imshow(mtest)      
        #show the windows
        cv2.imshow('mask',roi)
        cv2.imshow('frame',frame)
        
        #stops loop on keypress
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    
cv2.destroyAllWindows()
cap.release()    


    


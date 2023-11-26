import cv2
import numpy as np
import os

dataset_path = "./data/"
classId = 0
faceData = []
labels = []
nameMap = {}
offset = 40
for f in os.listdir(dataset_path):
    if f.endswith(".npy"):

        nameMap[classId] = f[:-4] #mapping names for each face
        #x-value
        dataItem = np.load(dataset_path+f)
        m = dataItem.shape[0]
        faceData.append(dataItem)
        #y-value (target)
        target = classId * np.ones((m,))
        classId +=1 

        labels.append(target)



#merge inputs
X_train = np.concatenate(faceData,axis= 0)
#merge targets
y_train = np.concatenate(labels,axis=0).reshape((-1,1)) #one column vector

print(X_train.shape)
print(y_train.shape)
print(nameMap)

#KNN algorithm

def dist(p,q):
    return np.sqrt(np.sum((p-q)**2))

def knn(X,y,xt,k=5):
    m = X.shape[0]
    dlist = []

    for i in range(m):
        d = dist(X[i],xt)
        dlist.append((d,y[i]))

    dlist = sorted(dlist)
    dlist = np.array(dlist[:k])
    labels = dlist[:,1]

    labels, cnts = np.unique(labels, return_counts=True)
    idx = cnts.argmax()
    pred = labels[idx]

    return int(pred)

#predictions

cam = cv2.VideoCapture(0)

model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    success,img = cam.read()
    if not success:
        print("Camera read failed!")

    
    faces = model.detectMultiScale(img,1.3,5)

    for f in faces:
        x,y,w,h = f
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        cropped_face = img[y-offset:y+h+offset,x-offset:x+w+offset]
        cropped_face = cv2.resize(cropped_face,(100,100))
        
        
        #predict the name using  KNN ALGORITHM
        predictedClass = knn(X_train,y_train,cropped_face.flatten()) #flatten because shape (100,100,3) and (100,100)

        predictedName = nameMap[predictedClass]
        cv2.putText(img,predictedName,(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Prediction window",img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
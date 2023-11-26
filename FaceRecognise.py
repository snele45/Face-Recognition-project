import cv2
import numpy as np
import os

dataset_path = "./data/"
classId = 0
faceData = []
labels = []
nameMap = {}

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
X = np.concatenate(faceData,axis= 0)
#merge targets
y = np.concatenate(labels,axis=0).reshape((-1,1)) #one column vector

print(X.shape)
print(y.shape)
print(nameMap)
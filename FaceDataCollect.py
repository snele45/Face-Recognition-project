import cv2

cam = cv2.VideoCapture(0)

fileName = input("Enter the name of the person : ")
dataset_path = "./data/"

model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    success,img = cam.read()
    if not success:
        print("Camera read failed!")

    faces = model.detectMultiScale(img,1.3,5)

    for f in faces:
        x,y,w,h = f
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Image window", img)

    key = cv2.waitKey(1)
    if key == ord(q):
        break

cam.release()
cv2.destroyAllWindows()
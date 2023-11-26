import cv2

cam = cv2.VideoCapture(0)

fileName = input("Enter the name of the person : ")
dataset_path = "./data/"
offset = 40
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    success,img = cam.read()
    if not success:
        print("Camera read failed!")

    
    faces = model.detectMultiScale(img,1.3,5)

    faces = sorted(faces,key=lambda f:f[2]*f[3])
    if len(faces)>0:
        f = faces[-1]
        x,y,w,h = f
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        cropped_face = img[y-offset:y+h+offset,x-offset:x+w+offset]
        cropped_face = cv2.resize(cropped_face,(100,100))

    cv2.imshow("Image window", img)
    cv2.imshow("Cropped Face",cropped_face)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
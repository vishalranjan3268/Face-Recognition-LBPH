#!/usr/bin/python

# Import the required modules
import cv2,os
import numpy as np
from PIL import Image
import RPi.GPIO as GPIO
import time
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.IN)         #Read output from PIR motion sensor

pir = GPIO.input(11) 
print pir
# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "/home/pi/Desktop/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.face.createLBPHFaceRecognizer()
global image_paths
def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            #cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            #cv2.waitKey(10)
    # return the images list and labels list
    return images, labels

# Path to the Yale Dataset
path = '/home/pi/Desktop/vinay_new'
# Call the get_images_and_labels function and get the face images and the 
# corresponding labels
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))

# Append the images with the extension .sad into image_paths
#image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad.gif')]
#print image_paths


# Start USB WebCam


def detect(gray):
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=5,
      minSize=(30, 30),
      flags = cv2.CASCADE_SCALE_IMAGE
    )
    m = gray
    #print faces
    count = 0
    curr_path = []
    #print len(faces)
    for (x, y, w, h) in faces:
        count += 1
        #print count
        print x, y, w, h
        #cv2.imshow('', gray)
        #cv2.waitKey(0)
        m = gray[y : y+h+20, x : x+w+20]
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #gray = gray.resize(320, 243), gray.ANTIALIAS)
        m = cv2.resize(m, (768, 1024))
        #cv2.imshow('', m)
        #cv2.waitKey(0)
        lol = '/home/pi/alam' + str(count) + '.jpg'
        cv2.imwrite(lol, m)
        m = Image.open(lol)
        lol = '/home/pi/alam' + str(count) + '.gif'
        curr_path.append(lol)
        m.save(lol)
    #print len(faces), curr_path
    return len(faces), curr_path, gray

while True:
    pir = GPIO.input(11)
    print pir
    if pir == 1:
        video_capture = cv2.VideoCapture(0)
        s, img = video_capture.read()
        wid = "Randajan"
        cv2.namedWindow(wid)
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        break
while s:
    cv2.imshow(wid, img)
    s, img = video_capture.read()
    key = cv2.waitKey(1)
    if key == 1:
        cv2.destroyWindow(wid)
        #img = img.resize(320, 243), Image.ANTIALIAS)
    cv2.imwrite('/home/pi/alam.jpg', img)
    image = cv2.imread('/home/pi/alam.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    val, c_path, img = detect(image)
    print val
    if val > 0:
        s = False
        video_capture.release()
        break


kill = val
#no_of_faces_recognized = 0
for image_path in image_paths:
    kill -= 1
    for i in c_path:
        predict_image_pil = Image.open(i)
            #predict_image_pil.show()
        predict_image = np.array(predict_image_pil, 'uint8')
        faces = faceCascade.detectMultiScale(predict_image)
        for (x, y, w, h) in faces:
            nbr_predicted, conf = recognizer.predict(predict_image[y: y + h , x: x + w])
            nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
            print nbr_actual, nbr_predicted, conf
            if nbr_actual == nbr_predicted:
                print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
                if conf < 27.0:
                        #no_of_faces_recognized += 1
                    if conf < 27.0 and conf > 22.0:
                            #replace image
                        print image_path
                        predict_image_pil.save(image_path)
                    break
                #else:
                    #    print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)
        #cv2.imshow("Recognizing Face", predict_image[y: y+h+20, x: x+w+20])
        #cv2.waitKey(100)
        if conf < 25.0:
            break
    if conf < 25.0:
        break
#print no_of_faces_recognized

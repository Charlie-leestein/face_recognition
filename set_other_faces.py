# -*- codeing: utf-8 -*-
import sys
import os
import cv2
import dlib

input_dir = './input_img'
output_dir = './other_faces'
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Use the frontal_face_detector that comes with dlib as our feature extractor
detector = dlib.get_frontal_face_detector()

index = 1
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % index)
            img_path = path+'/'+filename
            # Read images from file
            img = cv2.imread(img_path)
            # Convert to grayscale image
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Use detector for face detection dets is the returned result
            dets = detector(gray_img, 1)

            # Use the enumerate function to iterate over the elements in a sequence and their subscripts
            # The subscript i is the face serial number
            # left: the distance between the left side of the face and the left border of the picture; right: the distance between the right side of the face and the left border of the picture
            # top: the distance between the top of the face and the upper border of the picture; bottom: the distance between the bottom of the face and the upper border of the picture
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                # img[y:y+h,x:x+w]
                face = img[x1:y1,x2:y2]
                # Resize image
                face = cv2.resize(face, (size,size))
                cv2.imshow('image',face)
                # save Picture
                cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
                index += 1

            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)

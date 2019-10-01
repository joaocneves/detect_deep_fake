
# import the necessary packages
from face_utils import FaceAligner
from face_utils import rect_to_bb
import argparse

import dlib
import numpy as np
import imutils
import cv2
import glob
import os

# --------------------- PARAMS ------------------------- #

DEBUG = 0

shape_predictor_model = 'shape_predictor_68_face_landmarks.dat'
dataset_path = '/media/jcneves/DATASETS/CASIA-WebFace/byid_original/'
output_path = '/media/jcneves/DATASETS/CASIA-WebFace/byid_alignedlib/'

# --------------------- INIT ------------------------- #

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_model)
fa = FaceAligner(predictor, desiredFaceWidth=224, desiredLeftEye=(0.3, 0.3))

os.makedirs(output_path, exist_ok=True)

dataset_files = [f for f in glob.glob(dataset_path + "**/*.jpg", recursive=True)]
print(len(dataset_files))
last_path = ''
last_image_path_out = ''

for idx, image_path in enumerate(dataset_files):

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_path)
    # image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale image
    if DEBUG:
        cv2.imshow("Input", image)
        cv2.waitKey(1)

    [rects, scores, tmp] = detector.run(gray, 2)

    if len(rects) > 0:  # check if exist some detection



        ix = scores.index(max(scores))
        rect = rects[ix]

        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        faceAligned, pose = fa.align(image, gray, rect)

        # display the output images
        if DEBUG:
            faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
            cv2.imshow("Original", faceOrig)
            cv2.imshow("Aligned", faceAligned)
            cv2.waitKey(1)

        if pose:

            path, file = os.path.split(image_path)
            if path != last_path:
                image_path_out = path.replace(dataset_path,output_path)
                last_path = path
                last_image_path_out = image_path_out
                os.makedirs(image_path_out, exist_ok=True)
            else:
                image_path_out = last_image_path_out

            image_path_out = image_path_out + '/' + file
            cv2.imwrite(image_path_out, faceAligned)


# import the necessary packages
from face_utils import FaceAligner
from face_utils import rect_to_bb
import argparse

from skimage.feature import greycomatrix
import numpy as np
import imutils
import cv2
import glob
import os

# --------------------- PARAMS ------------------------- #

DEBUG = 0

shape_predictor_model = 'shape_predictor_68_face_landmarks.dat'
dataset_path = 'D:\\FACE_DATASETS\\CASIA-WebFace\\byid_original\\'
output_path = 'D:\\FACE_DATASETS\\CASIA-WebFace\\byid_comatrix\\'
eyes_margin = 0.3

# --------------------- INIT ------------------------- #



os.makedirs(output_path, exist_ok=True)

dataset_files = [f for f in glob.glob(dataset_path + "**/*.jpg", recursive=True)]
#dataset_files = dataset_files[30000:]
print(len(dataset_files))
last_path = ''
last_image_path_out = ''

for idx, image_path in enumerate(dataset_files):

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_path)

    image_np_r = np.array(image[:, :, 2])
    image_np_g = np.array(image[:, :, 1])
    image_np_b = np.array(image[:, :, 0])

    gcm_r = greycomatrix(image_np_r, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)
    gcm_g = greycomatrix(image_np_g, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)
    gcm_b = greycomatrix(image_np_b, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)

    gcm_r = np.sum(gcm_r, axis=3)[:, :, 0]
    gcm_g = np.sum(gcm_g, axis=3)[:, :, 0]
    gcm_b = np.sum(gcm_b, axis=3)[:, :, 0]

    gcm = np.stack((gcm_r, gcm_g, gcm_b), axis=2)

    # show the original input image and detect faces in the grayscale image
    if DEBUG:
        gcm_norm = np.array(gcm, dtype='float32')/(np.max(gcm[:])/50)
        cv2.imshow("Input", np.array(gcm_norm*255, dtype='uint8'))
        cv2.waitKey(1)

    path, file = os.path.split(image_path)
    if path != last_path:
        image_path_out = path.replace(dataset_path, output_path)
        last_path = path
        last_image_path_out = image_path_out
        os.makedirs(image_path_out, exist_ok=True)
    else:
        image_path_out = last_image_path_out

    image_path_out = image_path_out + '/' + file
    np.savez_compressed(image_path_out, gcm)
import cv2
import os
from matplotlib import pyplot as plt
import mediapipe as mp
from protobuf_to_dict import protobuf_to_dict 
import argparse
import glob
from natsort import natsorted
import pickle
import numpy as np
import math
#=======================


class Landmarks(object):

    def __init__(self, img_path, save_path):
        self.img_path = img_path
        self.save_path = save_path
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        
    def resize_and_show(self, image):
        DESIRED_HEIGHT = 256
        DESIRED_WIDTH = 256
        h, w = image.shape[:2]
        if h < w:
            img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
        else:
            img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

        base = os.path.basename(self.img_path)
        cv2.imwrite(self.save_path + '/' + '{}'.format(base), img)
        

    def draw_face(self):
        # Run MediaPipe Face Mesh, 0.3 - note that "refine_landmarks=True" returns irises
        with self.mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.3) as face_mesh:
            image = cv2.imread(self.img_path, flags=cv2.IMREAD_COLOR) #open image
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                      image=image,
                      landmark_list=face_landmarks,
                      connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                      landmark_drawing_spec=None,
                      connection_drawing_spec=self.mp_drawing_styles
                      .get_default_face_mesh_tesselation_style())

                    self.mp_drawing.draw_landmarks(
                      image=image,
                      landmark_list=face_landmarks,
                      connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                      landmark_drawing_spec=None,
                      connection_drawing_spec=self.mp_drawing_styles
                      .get_default_face_mesh_contours_style())

                    self.mp_drawing.draw_landmarks(
                      image=image,
                      landmark_list=face_landmarks,
                      connections=self.mp_face_mesh.FACEMESH_IRISES,
                      landmark_drawing_spec=None,
                      connection_drawing_spec=self.mp_drawing_styles
                      .get_default_face_mesh_iris_connections_style())

                self.resize_and_show(annotated_image)
        
            else: 
                print("NONE!")
                return None
 
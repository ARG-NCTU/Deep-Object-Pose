#!/usr/bin/env python3

# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

"""
This file starts a ROS node to run DOPE, 
"""

from __future__ import print_function

import cv2
import glob
import numpy as np
import resource_retriever
import rospy
from PIL import Image
from PIL import ImageDraw
from dope.inference.cuboid import Cuboid3d
from dope.inference.cuboid_pnp_solver import CuboidPNPSolver
from dope.inference.detector import ModelData, ObjectDetector

class Draw(object):
    """Drawing helper class to visualize the neural network output"""

    def __init__(self, im):
        """
        :param im: The image to draw in.
        """
        self.draw = ImageDraw.Draw(im)

    def draw_line(self, point1, point2, line_color, line_width=2):
        """Draws line on image"""
        if point1 is not None and point2 is not None:
            self.draw.line([point1, point2], fill=line_color, width=line_width)

    def draw_dot(self, point, point_color, point_radius):
        """Draws dot (filled circle) on image"""
        if point is not None:
            xy = [
                point[0] - point_radius,
                point[1] - point_radius,
                point[0] + point_radius,
                point[1] + point_radius
            ]
            self.draw.ellipse(xy,
                              fill=point_color,
                              outline=point_color
                              )

    def draw_cube(self, points, color=(255, 0, 0)):
        """
        Draws cube with a thick solid line across
        the front top edge and an X on the top face.
        """

        # draw front
        self.draw_line(points[0], points[1], color)
        self.draw_line(points[1], points[2], color)
        self.draw_line(points[3], points[2], color)
        self.draw_line(points[3], points[0], color)

        # draw back
        self.draw_line(points[4], points[5], color)
        self.draw_line(points[6], points[5], color)
        self.draw_line(points[6], points[7], color)
        self.draw_line(points[4], points[7], color)

        # draw sides
        self.draw_line(points[0], points[4], color)
        self.draw_line(points[7], points[3], color)
        self.draw_line(points[5], points[1], color)
        self.draw_line(points[2], points[6], color)

        # draw dots
        self.draw_dot(points[0], point_color=color, point_radius=4)
        self.draw_dot(points[1], point_color=color, point_radius=4)

        # draw x on the top
        self.draw_line(points[0], points[5], color)
        self.draw_line(points[1], points[4], color)


class DopeNode(object):
    """ROS node that runs DOPE, and output DOPE results"""
    def __init__(self):
        self.models = {}
        self.pnp_solvers = {}
        self.draw_colors = {}
        self.dimensions = {}
        self.class_ids = {}
        self.model_transforms = {}
        self.meshes = {}
        self.mesh_scales = {}
        self.prev_num_detections = 0

        self.input_is_rectified = rospy.get_param('~input_is_rectified', True)
        self.downscale_height = rospy.get_param('~downscale_height', 500)

        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = rospy.get_param('~thresh_angle', 0.5)
        self.config_detect.thresh_map = rospy.get_param('~thresh_map', 0.01)
        self.config_detect.sigma = rospy.get_param('~sigma', 3)
        self.config_detect.thresh_points = rospy.get_param("~thresh_points", 0.1)

        # For each object to detect, load network model, create PNP solver, and start predict
        for model, weights_url in rospy.get_param('~weights').items():
            self.models[model] = \
                ModelData(
                    model,
                    resource_retriever.get_filename(weights_url, use_protocol=False)
                )
            self.models[model].load_net_model()

            try:
                M = np.array(rospy.get_param('~model_transforms')[model], dtype='float64')
                self.model_transforms[model] = tf.transformations.quaternion_from_matrix(M)
            except KeyError:
                self.model_transforms[model] = np.array([0.0, 0.0, 0.0, 1.0], dtype='float64')

            try:
                self.meshes[model] = rospy.get_param('~meshes')[model]
            except KeyError:
                pass

            try:
                self.mesh_scales[model] = rospy.get_param('~mesh_scales')[model]
            except KeyError:
                self.mesh_scales[model] = 1.0
            try:
                self.draw_colors[model] = tuple(rospy.get_param("~draw_colors")[model])
            except:
                self.draw_colors[model] = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))

            self.dimensions[model] = tuple(rospy.get_param("~dimensions")[model])
            self.class_ids[model] = rospy.get_param("~class_ids")[model]

            self.pnp_solvers[model] = \
                CuboidPNPSolver(
                    model,
                    cuboid3d=Cuboid3d(rospy.get_param('~dimensions')[model])
                )

        print("Running DOPE...")
        self.image_prediction()


    def image_prediction(self):
        N_of_complete = 0
        # dataset_l = ['LIVALO']
        dataset_l = ['image']
        # dataset_l = ['3m', 'andes', 'cocacola', 'crayola', 'folgers', 'heineken', 'hunts', 'kellogg', 'kleenex', 'kotex', \
        #              'libava', 'macadamia', 'milo', 'mm', 'pocky', 'raisins', 'stax', 'swissmiss', 'vanish', 'viva']
        
        for dataset in dataset_l:
          N = 0
        #   dirPath = r"/content/train/*/*.*.jpg"
          dirPath = r"/home/julie/catkin_ws/src/dope/dataset/LIVALO_train/train/Scene1/24.main.jpg"
          # dirPath = r"/content/dataset/virtual_object/virtual_object/images/" + dataset + "/*/*"
          dataset_path = glob.glob(dirPath)
          print("\nDataset total size:", len(dataset_path))

          for path in dataset_path:
            N_of_complete += 1
            print(path)
            img = cv2.imread(path)

            # Update camera matrix and distortion coefficients
            if self.input_is_rectified:
                # P = np.matrix(camera_info.P, dtype='float64')
                P = np.matrix([1194.2562255859375, 0.0, 320.0, 0.0, 0.0, 1194.2562255859375, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype='float64')
                P.resize((3, 4))
                camera_matrix = P[:, :3]
                dist_coeffs = np.zeros((4, 1))
            else:
                # camera_matrix = np.matrix(camera_info.K, dtype='float64')
                camera_matrix = np.matrix([1194.2562255859375, 0.0, 320.0, 0.0, 1194.2562255859375, 240.0, 0.0, 0.0, 1.0], dtype='float64')
                camera_matrix.resize((3, 3))
                # dist_coeffs = np.matrix(camera_info.D, dtype='float64')
                dist_coeffs = np.matrix([0.0, 0.0, 0.0, 0.0, 0.0], dtype='float64')
                dist_coeffs.resize((len(dist_coeffs), 1))

            # Downscale image if necessary
            height, width, _ = img.shape
            scaling_factor = float(self.downscale_height) / height
            if scaling_factor < 1.0:
                camera_matrix[:2] *= scaling_factor
                img = cv2.resize(img, (int(scaling_factor * width), int(scaling_factor * height)))

            for m in self.models:
                self.pnp_solvers[m].set_camera_intrinsic_matrix(camera_matrix)
                self.pnp_solvers[m].set_dist_coeffs(dist_coeffs)

            # Copy and draw image
            img_copy = img.copy()
            im = Image.fromarray(img_copy)
            draw = Draw(im)

            for m in self.models:
                # Detect object
                results = ObjectDetector.detect_object_in_image(
                    self.models[m].net,
                    self.pnp_solvers[m],
                    img,
                    self.config_detect
                )

                # Publish pose and overlay cube on image
                for i_r, result in enumerate(results):
                    if result["location"] is None:
                        continue

                    # Draw the cube
                    if None not in result['projected_points']:
                        points2d = []
                        for pair in result['projected_points']:
                            points2d.append(tuple(pair))
                        draw.draw_cube(points2d, self.draw_colors[m])

            # Store the image with results overlaid
            print("{}:[{}/{} ({:.0f}%)]".format(dataset, N_of_complete, len(dataset_path), 100. * N_of_complete / len(dataset_path)))
            # cv2.imwrite("/content/predict_img/{}_prediction_{}.png".format(dataset, N), np.array(im))
            cv2.imwrite("/home/julie/catkin_ws/src/dope/test/24.main.jpg", np.array(im))
            N += 1
            
        print("Predict Finished")

def main():
    """Main routine to run DOPE"""

    # Initialize ROS node
    rospy.init_node('dope')
    DopeNode()


if __name__ == "__main__":
    main()

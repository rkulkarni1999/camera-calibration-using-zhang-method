import cv2
import os
import numpy as np

# Paths to the individual images
path_to_original_image = 'D:\Classical and Deep Learning Approaches for Geometric Computer Vision\HW1\\rkulkarni1\original\corners3.png'
path_to_reprojected_image = 'D:\Classical and Deep Learning Approaches for Geometric Computer Vision\HW1\\rkulkarni1\\reprojected\\reprojected_corners3.png'

# Read the images
original_image = cv2.imread(path_to_original_image)
reprojected_image = cv2.imread(path_to_reprojected_image)

# Resize if necessary
if original_image.shape != reprojected_image.shape:
    reprojected_image = cv2.resize(reprojected_image, (original_image.shape[1], original_image.shape[0]))

# Overlay the images
overlay_image = cv2.addWeighted(original_image, 0.5, reprojected_image, 0.5, 0)

overlay_path = os.path.join("overlay/",'overlayed.png')
cv2.imwrite(overlay_path, overlay_image)
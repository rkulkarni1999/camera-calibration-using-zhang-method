import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import os
import scipy.optimize as optimize
import math

####################
# PLOTTING FUNCTIONS
####################
def plot_and_save_corners(img, corners, name):
    # Ensure the 'Results' directory exists
    os.makedirs('Results', exist_ok=True)
    # Make a copy of the image to avoid modifying the original image
    img_with_corners = copy.deepcopy(img)
    # Plot each corner
    for corner in corners:
        cv2.circle(img_with_corners, (int(corner[0]), int(corner[1])), 7, (0, 0, 255), -1)
    # Save the image
    cv2.imwrite(f"Results/{name}.png", img_with_corners)

###########################
# IMPORTANT FUNCTIONS
###########################
###############
# Read Images
############### 
def read_images():
    folder_img = "Calibration_Imgs/Calibration_Imgs"
    
    if not os.path.exists(folder_img):
        raise FileNotFoundError("Directory doesn't exist")

    image_list = sorted(os.listdir(folder_img))
    images = [cv2.imread(os.path.join(folder_img, img)) for img in image_list]

    print(f"No. of images: {len(images)}")
    return images

################################
# Finding corners of the Image
################################
def find_image_corners(images, checker_size):
    found_corners = []
    for idx, img in enumerate(images, start=1):
        ret, corners = cv2.findChessboardCorners(img, checker_size, None)
        if ret:
            corners = corners.squeeze(1)
            plot_and_save_corners(img, corners, f"corners{idx}")
            found_corners.append(corners)
        else:
            print(f"Corners not found in image {idx}")
            found_corners.append(None)

    return np.array(found_corners, dtype=object)

#################################################################
# Finding World Corners (World Coordinate system of the Image)
#################################################################
def find_world_corners(checker_size, checker_box_size):
    corners = []
    for ii in range(1,checker_size[1]+1):
        for jj in range(1,checker_size[0]+1):
            corners.append((ii*checker_box_size, jj*checker_box_size))
    corners = np.array(corners, np.float32)
    return corners

###################################################
# Computing Homography for all Images (3x3 Matrix) 
###################################################
def find_homography(img_corners, world_corners):
    if len(img_corners) != len(world_corners) or len(img_corners) < 4:
        raise ValueError("Insufficient or unequal number of points in img_corners and world_corners.")

    num_points = len(img_corners)
    h_matrix = []

    for i in range(num_points):
        x, y = img_corners[i]
        X, Y = world_corners[i]
        h_matrix.append([-X, -Y, -1, 0, 0, 0, x*X, x*Y, x])
        h_matrix.append([0, 0, 0, -X, -Y, -1, y*X, y*Y, y])
    
    h_matrix = np.array(h_matrix)
    _, _, V_T = np.linalg.svd(h_matrix, full_matrices=True)
    H = V_T[-1].reshape(3, 3)

    # Normalize the matrix to ensure the last element is 1
    H = H / H[-1, -1]

    return H


def compute_homography(images_corners, world_corners):
    H_all = []
    for img_corners in images_corners:
        H = find_homography(img_corners, world_corners)
        H_all.append(H)
    H_all = np.array(H_all)
    return H_all

#############################
# Computing the the B Matrix
#############################
def V_ij(H, i, j):
    H = H.T
    return np.array([
        H[i, 0] * H[j, 0],
        H[i, 0] * H[j, 1] + H[i, 1] * H[j, 0],
        H[i, 1] * H[j, 1],
        H[i, 2] * H[j, 0] + H[i, 0] * H[j, 2],
        H[i, 2] * H[j, 1] + H[i, 1] * H[j, 2],
        H[i, 2] * H[j, 2]
    ]).reshape(1, -1)
    

def compute_B(H_all):
    v = [V_ij(H, 0, 1) for H in H_all]
    v.extend([V_ij(H, 0, 0) - V_ij(H, 1, 1) for H in H_all])
    v = np.vstack(v)

    _, _, V_T = np.linalg.svd(v)
    b = V_T[-1]

    B = np.array([[b[0], b[1], b[3]],
                [b[1], b[2], b[4]],
                [b[3], b[4], b[5]]])

    print("B matrix \n", B)
    return B

#######################################
# Computing the Intrinsics : K Matrix
#######################################
def compute_K(B):
    B_00, B_01, B_02 = B[0, 0], B[0, 1], B[0, 2]
    B_11, B_12 = B[1, 1], B[1, 2]
    B_22 = B[2, 2]

    denominator = B_00 * B_11 - B_01 ** 2
    if denominator == 0:
        raise ValueError("Invalid B matrix: Division by zero encountered.")

    v0 = (B_01 * B_02 - B_00 * B_12) / denominator
    lambda_value = B_22 - (B_02 ** 2 + v0 * (B_01 * B_02 - B_00 * B_12)) / B_00
    alpha = np.sqrt(lambda_value / B_00)
    gamma = -B_01 * alpha ** 2 / lambda_value
    u0 = (gamma * v0 / np.sqrt((lambda_value * B_00) / denominator)) - (B_02 * alpha ** 2 / lambda_value)

    K = np.array([[alpha, gamma, u0],
                [0, np.sqrt((lambda_value * B_00) / denominator), v0],
                [0, 0, 1]])

    print("K matrix \n", K)
    return K

#################################
# Computing Extrinsics : R and T  
#################################
def compute_extrinsics(K, H_matrices):
    K_inv = np.linalg.pinv(K)  
    Rt_all = []

    for H in H_matrices:
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        norm_factor = 1 / np.linalg.norm(np.dot(K_inv, h1))

        r1 = norm_factor * np.dot(K_inv, h1)
        r2 = norm_factor * np.dot(K_inv, h2)
        r3 = np.cross(r1, r2)  
        t = norm_factor * np.dot(K_inv, h3)

        Rt = np.column_stack([r1, r2, r3, t])  
        Rt_all.append(Rt)

    return Rt_all

####################################
# Get parameters for Optimization
####################################
def get_parameters(K, k_distortion):
    return np.array([K[0,0], K[0,1], K[1,1], K[0,2], K[1,2], k_distortion[0,0], k_distortion[1,0]])

###########################################
# Defining Loss Function for the Optimizer
###########################################
# Getting the A matrix
def A_matrix(param):
    alpha, gamma, beta, u0, v0, k1, k2 = param
    A = np.array([[alpha, gamma, u0],
                [0, beta, v0],
                [0, 0, 1]])
    k_distortion = np.array([[k1],[k2]])
    return A, k_distortion

def rms_reprojection(K, K_distortion, Rt_all, images_corners, world_corners):
    alpha, gamma, beta, u0, v0, k1, k2 = get_parameters(K, K_distortion)
    error_all_images = []
    reprojected_corners_all = []

    for Rt, img_corners in zip(Rt_all, images_corners):
        H = np.dot(K, Rt)
        error_img = 0
        reprojected_corners_img = []

        for world_point_2d, corners in zip(world_corners, img_corners):
            world_point_homo = np.array([world_point_2d[0], world_point_2d[1], 0, 1])
            corners = np.array([corners[0], corners[1]], dtype=float)

            proj_coords = np.dot(H, world_point_homo)
            u, v = proj_coords[0] / proj_coords[2], proj_coords[1] / proj_coords[2]

            normalized_coords = np.dot(Rt, world_point_homo)
            x, y = normalized_coords[0] / normalized_coords[2], normalized_coords[1] / normalized_coords[2]
            r = np.sqrt(x**2 + y**2)

            u_hat = u + (u - u0) * (k1 * r**2 + k2 * r**4)
            v_hat = v + (v - v0) * (k1 * r**2 + k2 * r**4)
            
            corners_hat = np.array([u_hat, v_hat], dtype=float) 
            reprojected_corners_img.append((u_hat, v_hat))
            error = np.linalg.norm(corners - corners_hat)
            error_img += error

        reprojected_corners_all.append(reprojected_corners_img)
        error_all_images.append(error_img / len(img_corners))

    return np.array(error_all_images), np.array(reprojected_corners_all)

# the loss Function to be optimized
def loss_function(x0, Rt_all, images_corners, world_corners):
    K, k_distortion = A_matrix(x0)
    error_all_images, _ = rms_reprojection(K, k_distortion, Rt_all, images_corners, world_corners)
    
    return np.array(error_all_images)

# Reading images and defining constants
images            = read_images()
checkerboard_size = (9,6)
checkerbox_size   = 21.5

def main():
    
    # Finding chess board corners of all images. Corners can be visualized in Results
    images_corners = find_image_corners(images, checkerboard_size) 
    
    # Finding world corners for all images. 
    world_corners  = find_world_corners(checkerboard_size, checkerbox_size)
    
    # Finding homography for all images. 
    H_all = compute_homography(images_corners, world_corners)
    
    # Computing the B matrix
    B = compute_B(H_all)
    # Computing the Intrinsics : K Matrix
    K = compute_K(B)
    
    # Computing the Extrinsics : R and T
    RT_all = compute_extrinsics(K, H_all)
    print("R and T First Image")
    print(RT_all[0])
    
    # Defining Distortions
    k_distortion = np.array([[0],
                            [0]])
    param = get_parameters(K, k_distortion)
    print("Params")
    print(param)
    
    # Optimization 
    print("Optimizing...")
    res_params = optimize.least_squares(loss_function, x0=param, method="lm", args=[RT_all, images_corners, world_corners])
    
    res = res_params.x
    K_new, K_dist_new = A_matrix(res)
    print("Undistorted K Matrix")
    print(K_new)
    print("New Distortion")
    print(K_dist_new)

    # Calculating Errors
    print("Mean Error and Reprojection Error")
    error_all_images, reprojected_points = rms_reprojection(K_new, K_distortion_new, RT_all, images_corners, world_corners)
    mean_error = np.mean(error_all_images)
    K_distortion_new = np.array([K_distortion_new[0,0], K_distortion_new[1,0], 0, 0, 0], dtype = float)
    print("Reprojection error", mean_error)
    
    for i in range(len(images)):
        img = images[i]
        img = cv2.undistort(img, K_new, K_distortion_new)
        plot_and_save_corners(img, reprojected_points[i], "reprojected_corners" + str(i+1))
    
if __name__ == "__main__":
    main()

import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

# Load 3D points from CSV files
def load_points_from_csv(csv_files):
    points_3d_list = []
    for csv_file in csv_files:
        points = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        points_3d_list.append(points)
    return points_3d_list

# Load camera intrinsic and extrinsic parameters from XML files
def load_camera_parameters_from_xml(intrinsic_files, extrinsic_files):
    camera_parameters = []
    
    for intrinsic_file, extrinsic_file in zip(intrinsic_files, extrinsic_files):
        # Load intrinsic parameters
        fs_intrinsic = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_READ)
        if not fs_intrinsic.isOpened():
            raise IOError(f"Error opening intrinsic file: {intrinsic_file}")
        
        camera_matrix_node = fs_intrinsic.getNode("camera_matrix")
        distortion_coefficients_node = fs_intrinsic.getNode("distortion_coefficients")
        
        if camera_matrix_node.empty() or distortion_coefficients_node.empty():
            fs_intrinsic.release()
            raise ValueError(f"'camera_matrix' or 'distortion_coefficients' not found: {intrinsic_file}")
        
        intrinsic_params = camera_matrix_node.mat()
        distortion_coefficients = distortion_coefficients_node.mat()
        fs_intrinsic.release()

        # Load extrinsic parameters
        fs_extrinsic = cv2.FileStorage(extrinsic_file, cv2.FILE_STORAGE_READ)
        if not fs_extrinsic.isOpened():
            raise IOError(f"Error opening extrinsic file: {extrinsic_file}")
        
        rvec_node = fs_extrinsic.getNode("rvec")
        tvec_node = fs_extrinsic.getNode("tvec")
        
        if rvec_node.empty() or tvec_node.empty():
            fs_extrinsic.release()
            raise ValueError(f"'rvec' or 'tvec' not found: {extrinsic_file}")

        rvec = np.array(rvec_node.mat().flatten())
        tvec = np.array(tvec_node.mat().flatten())
        extrinsic_params = np.hstack((rvec, tvec))
        fs_extrinsic.release()

        camera_parameters.append((intrinsic_params, distortion_coefficients, extrinsic_params))
    return camera_parameters

# Visualize 2D projection of the 3D points in each image
def visualize_2d_projection(points_3d_list, intrinsic_files, extrinsic_files):
    if len(points_3d_list) != len(intrinsic_files) or len(intrinsic_files) != len(extrinsic_files):
        raise ValueError("The number of files in points_3D_list, intrinsic_files and extrinsic_files must be the same.")
    
    for i in range(len(points_3d_list)):
        points_3d = np.array(points_3d_list[i])
        intrinsic_file = intrinsic_files[i]
        extrinsic_file = extrinsic_files[i]

        # Convert 3D points from cm to meters
        points_3d /= 100.0
        
        # Load camera intrinsic and extrinsic parameters
        fs_intrinsic = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_READ)
        intrinsic_params = fs_intrinsic.getNode("camera_matrix").mat()
        distortion_coefficients = fs_intrinsic.getNode("distortion_coefficients").mat()
        fs_intrinsic.release()
        
        fs_extrinsic = cv2.FileStorage(extrinsic_file, cv2.FILE_STORAGE_READ)
        rvec = np.array(fs_extrinsic.getNode("rvec").mat().flatten())
        tvec = np.array(fs_extrinsic.getNode("tvec").mat().flatten())
        fs_extrinsic.release()

        # Reproject 3D points to 2D
        projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, intrinsic_params, distortion_coefficients)
        projected_points_2d = projected_points[:, 0, :]

        # Plot the projected points
        plt.figure(figsize=(10, 8))
        plt.scatter(projected_points_2d[:, 0], projected_points_2d[:, 1], color='red', s=10, label='Projected Points')
        plt.title(f'Projection of 3D Points onto 2D Plane for Image {i+1}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
        plt.legend()
        plt.show()

# Fuse multiple 3D poses
def fuse_poses_3d(points_3d_list, camera_parameters):
    pose3d_pool = []
    angle_pool = []

    for i, points_3d in enumerate(points_3d_list):
        intrinsic_params, distortion_coefficients, extrinsic_params = camera_parameters[i]
        rvec = extrinsic_params[:3]
        tvec = extrinsic_params[3:]

        # Convert 3D points from cm to meters
        points_3d /= 100.0
        
        # Reproject 2D points to 3D using camera parameters
        projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, intrinsic_params, distortion_coefficients)
        pose3d_pool.append(np.hstack((points_3d, projected_points[:, 0, :])))
        
        # Compute angles for fusion weights
        if len(points_3d) > 12:  # Ensure there are enough points
            lsh, rsh, lhip, rhip = points_3d[5], points_3d[6], points_3d[11], points_3d[12]
            msh = (lsh + rsh) / 2.0
            mhip = (lhip + rhip) / 2.0
            sh = rsh - lsh
            spine = mhip - msh
            person_dir = np.cross(sh, spine)
            cam_loc = tvec
            person_cam = msh - cam_loc
            v1 = person_dir / np.linalg.norm(person_dir)
            v2 = person_cam / np.linalg.norm(person_cam)
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) / np.pi * 180.0
            if angle > 90:
                angle = 180 - angle
            angle_pool.append(angle)
        else:
            angle_pool.append(0)

    # Fuse multiple views
    final_pose3d_pool = []
    num_points = max(len(p) for p in pose3d_pool)

    for j in range(num_points):
        cluster_points = []
        cluster_angles = []
        
        for i in range(len(pose3d_pool)):
            if j < len(pose3d_pool[i]):
                cluster_points.append(pose3d_pool[i][j])
                cluster_angles.append(angle_pool[i])

        if cluster_points:
            cluster_points = np.array(cluster_points)
            cluster_angles = np.array(cluster_angles)

            if len(cluster_points) == 1:
                final_pose3d_pool.append(cluster_points[0][:3])
            else:
                weights = 90 - cluster_angles
                mean_pose3d = np.sum(cluster_points[:, :3] * weights.reshape(-1, 1), axis=0) / (np.sum(weights) + 1e-8)
                final_pose3d_pool.append(mean_pose3d)

    return np.array(final_pose3d_pool)

# Load ground truth data
def load_ground_truth(json_file):
    with open(json_file, 'r') as f:
        ground_truth = json.load(f)
    return ground_truth

# 3D visualization of fused points
def visualize_3d_points(points_3d_list, title):
    if points_3d_list.size == 0:
        print("No points to display.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if points_3d_list.ndim == 1:
        points_3d_list = np.expand_dims(points_3d_list, axis=0)

    for i, points in enumerate(points_3d_list):
        if points.ndim == 1:
            points = np.expand_dims(points, axis=0)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=f'Points Set {i+1}')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.legend()
    plt.show()

# Main script
def main():
    shot = "00000005"
    points_3d_files_list = [f'./images/3D_Points/{shot}/cam_{i+1}.csv' for i in range(7)]
    intrinsic_files = [f'./camera_parameters/intrinsic/intr_{i+1}.xml' for i in range(7)]
    extrinsic_files = [f'./camera_parameters/extrinsic/extr_{i+1}.xml' for i in range(7)]

    ground_truth_file = f'./images/ground_truth/{shot}.json'

    # Load data
    points_3d_list = load_points_from_csv(points_3d_files_list)
    camera_parameters = load_camera_parameters_from_xml(intrinsic_files, extrinsic_files)
    ground_truth = load_ground_truth(ground_truth_file)

    # Visualize the 3D points projected to 2D for each image
    # visualize_2d_projection(points_3d_list, intrinsic_files, extrinsic_files)

    # Fuse 3D poses
    fused_points_3d_list = fuse_poses_3d(points_3d_list, camera_parameters)

    # Visualize the fused points
    visualize_3d_points(fused_points_3d_list, "3D Fused Points")

if __name__ == "__main__":
    main()
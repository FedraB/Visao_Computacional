import numpy as np
import cv2
import json
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Load ground truth data
def load_ground_truth(json_file):
    with open(json_file, 'r') as f:
        ground_truth = json.load(f)

    gt_points_3d = []

    # Convert positionID to world coordinates (X, Y, Z)
    for person in ground_truth:
        position_id = person["positionID"]
        
        # Calculate X and Y coordinates based on the grid (X-first, 2.5cm spacing)
        X = -3.0 + 0.025 * (position_id % 480)
        Y = -9.0 + 0.025 * (position_id // 480)
        Z = 0  # Since the points are located on the ground plane (Z=0)
        
        # Store the 3D ground truth point
        gt_points_3d.append([X, Y, Z])

    return np.array(gt_points_3d)

# Visualize 2D projection of the 3D points in each image
def visualize_3d_to_2d(points_3d_list, intrinsic_files, extrinsic_files):
    if len(points_3d_list) != len(intrinsic_files) or len(intrinsic_files) != len(extrinsic_files):
        raise ValueError("The number of files in points_3D_list, intrinsic_files and extrinsic_files must be the same.")
    
    for i in range(len(points_3d_list)):
        points_3d = np.array(points_3d_list[i])
        intrinsic_file = intrinsic_files[i]
        extrinsic_file = extrinsic_files[i]

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
def fuse_poses_3d(points_3d_list, camera_parameters, distance_threshold):
    """
    Function to fuse multiple 3D poses from different cameras using angle-weighted fusion.
    """
    # Initialize pools to store fused 3D poses and the corresponding angles between the skeletons and the cameras.
    pose3d_pool = []
    angle_pool = []

    # Reproject and store the 3D points for each camera.
    for i, points_3d in enumerate(points_3d_list):
        # Unpack the intrinsic, distortion, and extrinsic parameters for the current camera.
        intrinsic_params, distortion_coefficients, extrinsic_params = camera_parameters[i]
        rvec = extrinsic_params[:3]  # Rotation vector from the extrinsic parameters.
        tvec = extrinsic_params[3:]  # Translation vector from the extrinsic parameters.

        # Reproject the 3D points from the world coordinate system into the camera's coordinate system.
        projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, intrinsic_params, distortion_coefficients)

        # Append the reprojected points to the pose3d_pool, adding the projected 2D points as well.
        pose3d_pool.append(np.hstack((points_3d, projected_points[:, 0, :])))

        # For each person in the 3D points, calculate the angle between the skeleton and the camera.
        for person in points_3d:  # Iterate over each detected person.
            if len(person) >= 13:  # Ensure there are enough points (13+ points) to define body structure.
                # Extract points for left shoulder (lsh), right shoulder (rsh), left hip (lhip), and right hip (rhip).
                lsh, rsh, lhip, rhip = person[5], person[6], person[11], person[12]

                # Calculate midpoints of the shoulders and hips for the center of the body.
                msh = (lsh + rsh) / 2.0  # Midpoint of the shoulders.
                mhip = (lhip + rhip) / 2.0  # Midpoint of the hips.

                # Calculate body vectors for the shoulder line (sh) and the spine (spine).
                sh = rsh - lsh  # Vector between the shoulders.
                spine = mhip - msh  # Vector between the mid-hip and mid-shoulder.

                # Compute the body direction by taking the cross product of shoulder and spine vectors.
                person_dir = np.cross(sh, spine)

                # Compute the vector between the camera and the person (camera to shoulder midpoint).
                cam_loc = tvec  # Camera's location in the scene.
                person_cam = msh - cam_loc  # Vector from camera to the midpoint of the shoulders.

                # Normalize both vectors (body direction and camera-person vector).
                v1 = person_dir / np.linalg.norm(person_dir)
                v2 = person_cam / np.linalg.norm(person_cam)

                # Compute the angle between the body direction and the camera-person vector.
                angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) / np.pi * 180.0

                # Adjust the angle to always be between 0 and 90 degrees (ensure symmetry).
                if angle > 90:
                    angle = 180 - angle
                # Append the computed angle to the angle pool for later weighting.
                angle_pool.append(angle)
            else:
                # If there are not enough points to compute angles, append a default angle of 0.
                angle_pool.append(0)

    # Combine all 3D points from all cameras into a single pool for clustering.
    all_points_3d = np.vstack([pose[:, :3] for pose in pose3d_pool])

    # Use hierarchical clustering (single linkage) to group 3D points that are close to each other.
    Z = linkage(all_points_3d, method='single', metric='euclidean')

    # Label each point based on the cluster it belongs to, using the provided distance threshold.
    labels = fcluster(Z, t=distance_threshold, criterion='distance')

    final_pose3d_pool = []

    # Iterate over each unique cluster identified by the hierarchical clustering.
    for cluster_id in np.unique(labels):
        # Select the 3D points belonging to the current cluster.
        cluster_points = all_points_3d[labels == cluster_id]

        # If the cluster contains only a single point, keep it as is.
        if len(cluster_points) == 1:
            final_pose3d_pool.append(cluster_points[0])
        else:
            # For clusters with multiple points, fuse them using an angle-weighted average.
            cluster_angles = np.array(angle_pool)[labels == cluster_id]  # Retrieve the angles for the current cluster.
            weights = 90 - cluster_angles  # Compute weights inversely proportional to the angle (closer to 90 means less weight).
            
            # Compute the weighted average of the points in the cluster.
            mean_pose3d = np.sum(cluster_points * weights.reshape(-1, 1), axis=0) / (np.sum(weights) + 1e-8)
            final_pose3d_pool.append(mean_pose3d)

    # Return the final set of fused 3D poses as a numpy array.
    return np.array(final_pose3d_pool)

# Visualize 3D points
def visualize_3d_points(points_3d_list, title):

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
    plt.show()

# Finds the intersection between the fused points and the ground truth points
def find_intersection_with_ground_truth(fused_points_3d, ground_truth_3d_points, threshold):
    """
    Finds the intersection between fused 3D points and ground truth 3D points, and calculates precision, recall, and F1-Score.
    """
    
    intersection_points = []  # List to store matching pairs of fused and ground truth points.

    # Iterate through each fused 3D point to check if it has a corresponding match in the ground truth.
    for fused_point in fused_points_3d:
        # Calculate the Euclidean distance from the fused point to all ground truth points.
        distances = np.linalg.norm(ground_truth_3d_points - fused_point, axis=1)
        
        # Find the minimum distance and the index of the ground truth point closest to the current fused point.
        min_distance = np.min(distances)
        min_idx = np.argmin(distances)

        # If the minimum distance is within the specified threshold, it is considered a valid match (intersection).
        if min_distance <= threshold:
            intersection_points.append((fused_point, ground_truth_3d_points[min_idx]))

    # Calculate the number of intersection points (matches).
    num_intersection_points = len(intersection_points)
    
    # Proportion of fused points that have a match in the ground truth.
    precision = num_intersection_points / len(fused_points_3d) if len(fused_points_3d) > 0 else 0
    
    # Proportion of ground truth points that have a match in the fused points.
    recall = num_intersection_points / len(ground_truth_3d_points) if len(ground_truth_3d_points) > 0 else 0

    # F1-Score is the harmonic mean of precision and recall.
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    # Return the matched points (intersection), along with precision, recall, and F1-Score.
    return intersection_points, precision, recall, f1_score

# Rotate points around Z-axis
def rotate_z(points, angle):
    cos_angle = np.cos(np.radians(angle))
    sin_angle = np.sin(np.radians(angle))
    rotation_matrix = np.array([[cos_angle, -sin_angle, 0],
                                [sin_angle, cos_angle, 0],
                                [0, 0, 1]])
    return np.dot(points, rotation_matrix.T)

def visualize_intersections(intersection_points, fused_points_3d, ground_truth_3d_points, title="Interseção entre Pontos Fundidos e Ground Truth"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convertendo para array para facilitar a manipulação
    intersection_fused_points = np.array([point[0] for point in intersection_points])
    intersection_ground_truth_points = np.array([point[1] for point in intersection_points])

    # Plot de todos os pontos 3D fundidos
    ax.scatter(fused_points_3d[:, 0], fused_points_3d[:, 1], fused_points_3d[:, 2], c='blue', label='Pontos Fundidos', s=20)

    # Plot de todos os pontos 3D do ground truth
    ax.scatter(ground_truth_3d_points[:, 0], ground_truth_3d_points[:, 1], ground_truth_3d_points[:, 2], c='green', label='Ground Truth', s=20)

    # Plot dos pontos de interseção (em vermelho)
    ax.scatter(intersection_fused_points[:, 0], intersection_fused_points[:, 1], intersection_fused_points[:, 2], c='red', label='Interseções Fundidos', s=50, marker='x')
    ax.scatter(intersection_ground_truth_points[:, 0], intersection_ground_truth_points[:, 1], intersection_ground_truth_points[:, 2], c='yellow', label='Interseções Ground Truth', s=50, marker='o')

    # Configurações de visualização
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

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
    ground_truth_3d_points = load_ground_truth(ground_truth_file)
    
    # Threshold in meters
    threshold = 1

    # Visualize 3D coordinates
    # visualize_3d_points(ground_truth_3d_points, "Ground Truth 3D Points")

    # Visualize the 2D projection of the 3D points for each image
    # visualize_3d_to_2d(points_3d_list, intrinsic_files, extrinsic_files)

    # Fuse 3D poses
    fused_points_3d_list = fuse_poses_3d(points_3d_list, camera_parameters, threshold)

    # Setting Z-coordinates to zero and mirroring the fused points
    fused_points_3d_zero_z = fused_points_3d_list.copy()
    fused_points_3d_zero_z[:, 2] = 0
    fused_points_3d_zero_z[:, 0] *= -1
    fused_points_3d_zero_z = rotate_z(fused_points_3d_zero_z, -135)

    intersection_points, precision, recall, f1_score = find_intersection_with_ground_truth(fused_points_3d_zero_z, ground_truth_3d_points, threshold)

    print("Recall: ", recall, " | Precision: ", precision, " | F1 Score: ", f1_score)
    visualize_intersections(intersection_points, fused_points_3d_zero_z, ground_truth_3d_points)

if __name__ == "__main__":
    main()

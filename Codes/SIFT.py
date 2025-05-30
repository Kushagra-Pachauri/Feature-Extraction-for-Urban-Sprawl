import os
import time
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def extract_sift_features(image):
    """
    Extract SIFT features from an image.
    """
    print("Step: Converting image to grayscale for SIFT feature extraction.")
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    print("Step: Initializing SIFT detector.")
    sift = cv2.SIFT_create()

    print("Step: Detecting keypoints and computing descriptors.")
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    print(f"Step: Found {len(keypoints)} keypoints.")
    
    # Optionally, visualize keypoints on the image (if needed for debugging)
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
    
    return keypoints, descriptors, image_with_keypoints

def classify_land_cover(image, sift_descriptors, n_clusters=3):
    """
    Classify land cover using K-means clustering with SIFT and color features.
    """
    print("Step: Preprocessing image for K-means clustering.")
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image")

    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    print("Step: Resizing image to manageable size.")
    resized_image = cv2.resize(image, (1024, 768))

    print("Step: Flattening and normalizing image data.")
    flat_image = resized_image.reshape((-1, 3)).astype(float)

    mean_color = np.mean(flat_image, axis=0)
    std_color = np.std(flat_image, axis=0)

    height, width, _ = resized_image.shape
    color_features = []
    for row in range(height):
        for col in range(width):
            pixel = resized_image[row, col]
            color_dist = np.linalg.norm(pixel - mean_color)
            norm_row = row / height
            norm_col = col / width
            color_features.append([pixel[0], pixel[1], pixel[2], color_dist, norm_row, norm_col])

    color_features = np.array(color_features)

    print("Step: Scaling features for clustering.")
    scaler = StandardScaler()
    scaled_color_features = scaler.fit_transform(color_features)
    
    # Flatten SIFT descriptors to use them in clustering
    if sift_descriptors is not None:
        sift_descriptors = np.array(sift_descriptors)
        # Flatten the SIFT descriptors (if any)
        flattened_sift_features = sift_descriptors.flatten().reshape(1, -1)
    else:
        flattened_sift_features = np.array([])

    # Combine color features with SIFT descriptors
    if flattened_sift_features.size > 0:
        combined_features = np.concatenate([scaled_color_features, flattened_sift_features.T], axis=1)
    else:
        combined_features = scaled_color_features

    print("Step: Applying K-means clustering.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(combined_features)
    labels = kmeans.labels_.reshape((height, width))

    print("Step: Assigning cluster colors.")
    cluster_colors = []
    for cluster in range(n_clusters):
        cluster_mask = (labels == cluster)
        cluster_pixels = resized_image[cluster_mask]
        cluster_colors.append(np.mean(cluster_pixels, axis=0))

    sorted_indices = np.argsort([np.mean(color) for color in cluster_colors])
    precise_colors = [
        [0, 255, 0],     # Green for vegetation
        [0, 0, 255],     # Red for urban areas
        [255, 0, 0]      # Blue for barren land/water
    ]

    segmented_image = np.zeros_like(resized_image)
    for i, cluster in enumerate(sorted_indices):
        mask = labels == cluster
        segmented_image[mask] = precise_colors[i]
    
    return segmented_image

def process_satellite_images(input_folder, output_folder):
    """
    Process satellite images with SIFT feature extraction and advanced segmentation.
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    print(f"Total images to process: {len(image_files)}")
    start_time = time.time()

    def process_image(image_path, filename):
        try:
            print(f"Loading image {filename}...")
            image = cv2.imread(image_path)

            if image is None:
                print(f"Warning: Could not read image {filename}")
                return

            height, width = image.shape[:2]
            print(f"Processing image {filename} (Size: {width}x{height})")

            # Extract SIFT features
            keypoints, descriptors, image_with_keypoints = extract_sift_features(image)
            print(f"Extracted {len(keypoints)} keypoints from {filename}")

            # Use SIFT features for classification and segmentation
            segmented_image = classify_land_cover(image, descriptors)

            output_path = os.path.join(output_folder, f'segmented_sift_{filename}')
            cv2.imwrite(output_path, segmented_image)

            # Optionally, save the image with keypoints visualized
            keypoints_output_path = os.path.join(output_folder, f'keypoints_{filename}')
            cv2.imwrite(keypoints_output_path, image_with_keypoints)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    with ThreadPoolExecutor(max_workers=1) as executor:
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(input_folder, filename)
            future = executor.submit(process_image, image_path, filename)
            try:
                future.result(timeout=60)  # Timeout for processing each image
            except TimeoutError:
                print(f"Timeout: Processing for {filename} took too long, skipping...")

            elapsed_time = time.time() - start_time
            avg_time_per_image = elapsed_time / i
            estimated_remaining_time = avg_time_per_image * (len(image_files) - i)

            print(f"Progress: {i / len(image_files) * 100:.2f}%")
            print(f"Estimated time remaining: {estimated_remaining_time:.2f} seconds")

    total_processing_time = time.time() - start_time
    print(f"Total processing time: {total_processing_time:.2f} seconds")

if __name__ == "__main__":
    input_folder = "Clean Images"  # Path to the folder containing input images
    output_folder = "Segmented_SIFT"  # Path to the folder where processed images will be saved
    process_satellite_images(input_folder, output_folder)
    print("Image processing complete!")

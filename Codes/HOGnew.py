import os
import time
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def extract_hog_features(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    """
    Extract HOG features from an image.
    """
    print("Step: Converting image to grayscale.")
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    print("Step: Resizing image for HOG feature extraction.")
    resized = cv2.resize(gray, (128, 128))

    print("Step: Computing HOG features.")
    features, _ = hog(
        resized,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        visualize=True
    )
    return features

def classify_land_cover(image, hog_features, n_clusters=3):
    """
    Classify land cover using K-means clustering with HOG and color features.
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
    
    # Combine HOG features with color features
    combined_features = np.concatenate([scaled_color_features, hog_features.T], axis=1)

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
    Process satellite images with HOG feature extraction and advanced segmentation.
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

            # Extract HOG features
            hog_features = extract_hog_features(image)
            print(f"Extracted {len(hog_features)} HOG features from {filename}")

            # Perform segmentation with HOG features integrated
            segmented_image = classify_land_cover(image, hog_features)

            output_path = os.path.join(output_folder, f'segmented_hog_{filename}')
            cv2.imwrite(output_path, segmented_image)

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
    output_folder = "Segmented_HOG"  # Path to the folder where processed images will be saved
    process_satellite_images(input_folder, output_folder)
    print("Image processing complete!")

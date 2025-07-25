import cv2
import numpy as np
from skimage import exposure

def load_image(image_path):
    """Load an image from file."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def detect_noise(image):
    """Detect noise level in the image."""
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the Laplacian to detect edges
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    noise = laplacian.var()
    return noise

def detect_edges(image):
    """Detect edges in the image."""
    edges = cv2.Canny(image, 100, 200)
    return edges

def detect_anomalies(image):
    """Detect anomalies in the image that may suggest manipulation."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Equalize the histogram to enhance contrast
    equalized = cv2.equalizeHist(gray)
    return equalized

def detect_illumination(image):
    """Detect illumination inconsistencies."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value_channel = hsv_image[:, :, 2]
    return value_channel

def display_images(images, titles):
    """Display a list of images with corresponding titles."""
    for i in range(len(images)):
        cv2.imshow(titles[i], images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def analyze_image(image_path):
    image = load_image(image_path)

    noise = detect_noise(image)
    edges = detect_edges(image)
    anomalies = detect_anomalies(image)
    illumination = detect_illumination(image)

    print(f"Noise Level: {noise}")

    display_images([image, edges, anomalies, illumination],
                   ['Original Image', 'Edges', 'Anomalies', 'Illumination Inconsistencies'])

if __name__ == "__main__":
    image_path = 'images/61DMgCXRRGL._SY879_.jpg'
    analyze_image(image_path)

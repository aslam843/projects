import numpy as np
from matplotlib import pyplot as plt
import os
import threading
import datetime
import time
import pymongo
import cv2
from PIL import Image
from rembg import remove
from concurrent.futures import ThreadPoolExecutor, as_completed
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# MongoDB setup
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["ImageMatching_Target_Test_Setup"]

# directory = r'E:\Image_Matching\Image_Matching_Score_Generation\Inputs3\Images1'
# directory = r'E:\Target_Image_Matching_Sample\Images_Walmart'
# directory = r'E:\Target_Image_Matching_Sample\Images_Amazon'
directory = r'D:\python\Image Matching\TT Part1 tool\\'
# directory = r'F:\Target Images\Batch 1\\'
# directory = r'F:\Testing_Resize\\'
# directory = r'F:\Target_Image_Matching_2_Lakh\Batch 1\\'

# Initialize feature detector
method = 'ORB'  # 'ORB' or 'SIFT'
sift = cv2.ORB_create() if method == 'ORB' else cv2.xfeatures2d.SIFT_create()

# Load the pre-trained MobileNet-SSD model and classes
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "bag", "jar", "shoe", "Tshirt", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
           "socks", "pillow", "bedsheet"]


# Console clearing function
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


# Add this function to create and save processed images
def save_processed_image(image, path, step):
    # Ensure the folder for saving processed images exists
    processed_dir = os.path.join(os.path.dirname(path), 'resized')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Save the image with a unique name based on the step (resize/normalize)
    filename = os.path.basename(path)
    save_path = os.path.join(processed_dir, f"{step}_{filename}")
    cv2.imwrite(save_path, image)
    print(f"{step.capitalize()} version saved at {save_path}")


# Define the normalization function (already in your code)
def normalize_image(image):
    # Normalize the image to the range 0-255
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_image


# Function for object detection using MobileNet-SSD
def detect_objects_tf(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    detected_objects = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            # Calculate the bounding box for the detected object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and label on the image
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {confidence:.2f}", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Store the detected objects
            detected_objects.append((label, confidence, (startX, startY, endX, endY)))

            # **Print the detected object label and confidence**
            print(f"Detected object: {label}, Confidence: {confidence:.2f}")

    return image, detected_objects


def fallback_matching(img1, img2):
    """
    Perform fallback image-to-image matching when descriptors are None or incompatible.
    """
    try:
        print("Debug: Fallback to direct image-to-image matching due to descriptor issues.")

        # Normalize the full images without cropping
        img1_normalized = normalize_image(img1)
        img2_normalized = normalize_image(img2)

        # Detect features and descriptors in the full images
        kp1, des1 = sift.detectAndCompute(img1_normalized, None)
        kp2, des2 = sift.detectAndCompute(img2_normalized, None)

        if des1 is None or des2 is None:
            print("Debug: Descriptors are still None in fallback. Unable to proceed with matching.")
            return 0  # Return 0% match as no descriptors were found

        # Descriptor Matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test to filter good matches
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        # Calculate the match percentage
        percent = (len(good) / min(len(kp1), len(kp2))) * 100
        print(f"Debug: Fallback matching percentage: {percent}%")
        return percent
    except Exception as e:
        print(f"Error during fallback matching: {e}")
        return 0  # Return 0% match if fallback fails

def process_client_image(client_image, clientsource, source, mycol):
    try:
        print(f"Processing client image: {client_image}")

        # Step 1: Read client image in color
        img1 = cv2.imread(clientsource + client_image)  # Read client image as color (BGR format)

        # Step 2: Convert to RGB for TensorFlow processing
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        # Step 3: Detect objects using MobileNet-SSD model in client image
        img1_with_objects, client_detected_objects = detect_objects_tf(img1_rgb)

        # Display image with detected objects in client image
        # plt.imshow(img1_with_objects)
        # plt.title(f"Object Detection: {client_image}")
        # plt.axis('off')
        # plt.show()

        # Step 4: Perform comparison and matching for all images in 'Comp' folder
        for root, dirs, filenames in os.walk(source):
            for f in filenames:
                try:
                    print(f"\nProcessing comparison image: {f}")
                    img2 = cv2.imread(source + f)

                    # Step 5: Convert comparison image to RGB for TensorFlow processing
                    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

                    # Step 6: Detect objects using MobileNet-SSD model in comparison image
                    img2_with_objects, comp_detected_objects = detect_objects_tf(img2_rgb)

                    # Display image with detected objects in comparison image
                    # plt.imshow(img2_with_objects)
                    # plt.title(f"Object Detection: {f}")
                    # plt.axis('off')
                    # plt.show()

                    # Step 7: Check if both client and comparison images have exactly 1 detected object
                    if len(client_detected_objects) == 1 and len(comp_detected_objects) == 1:
                        print(
                            "Debug: One object detected in both client and comparison images. Proceeding with straightforward matching.")

                        # Log detected bounding boxes
                        client_box = client_detected_objects[0][2]
                        comp_box = comp_detected_objects[0][2]
                        print(f"Debug: Client detected object bounding box: {client_box}")
                        print(f"Debug: Comparison detected object bounding box: {comp_box}")

                        # Step 8: Crop both images to the detected object region
                        (startX_client, startY_client, endX_client, endY_client) = client_box
                        (startX_comp, startY_comp, endX_comp, endY_comp) = comp_box
                        client_cropped = img1[startY_client:endY_client, startX_client:endX_client]
                        comp_cropped = img2[startY_comp:endY_comp, startX_comp:endX_comp]

                        # Log dimensions of cropped images
                        print(
                            f"Debug: Dimensions of cropped client image: {client_cropped.shape if client_cropped is not None else 'None'}")
                        print(
                            f"Debug: Dimensions of cropped comparison image: {comp_cropped.shape if comp_cropped is not None else 'None'}")

                        # Step 9: Normalize both client and comparison cropped images
                        client_cropped = normalize_image(client_cropped)  # Normalize client image
                        comp_cropped = normalize_image(comp_cropped)  # Normalize comparison image

                        # Log normalization step
                        print("Debug: Normalized cropped client and comparison images.")

                        preprocessdesource = clientsource.replace('Client\\', '')

                        # Log the preprocessing directory (if needed)
                        print(f"Debug: Preprocessing source directory: {preprocessdesource}")

                        # Feature detection and description for the cropped client image
                        kp1, des1 = sift.detectAndCompute(client_cropped, None)
                        kp2, des2 = sift.detectAndCompute(comp_cropped, None)

                        # Log keypoints and descriptors
                        print(f"Debug: Keypoints in client image: {len(kp1)}")
                        print(f"Debug: Keypoints in comparison image: {len(kp2)}")
                        print(f"Debug: Descriptors in client image: {des1.shape if des1 is not None else 'None'}")
                        print(f"Debug: Descriptors in comparison image: {des2.shape if des2 is not None else 'None'}")


                        if des1 is None or des2 is None or des1.shape[1] != des2.shape[1]:
                            print("Debug: Descriptor issue - Either descriptors are None or shapes are incompatible. Skipping matching.")
                            percent = fallback_matching(img1, img2) #new logic to handle the skipping
                            #continue old logic which was skipping the image

                        else:
                            # Descriptor Matching
                            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
                            matches = bf.knnMatch(des1, des2, k=2)

                            # Log number of matches before Lowe's ratio test
                            print(f"Debug: Total matches before Lowe's ratio test: {len(matches)}")

                            # Apply Lowe's ratio test to filter good matches
                            good = []
                            for m, n in matches:
                                if m.distance < 0.75 * n.distance:
                                    if m not in good:  # Avoid duplicates
                                        good.append([m])

                            # Log filtered good matches
                            print(f"Debug: Good matches after Lowe's ratio test: {len(good)}")

                            percent = (len(good) / min(len(kp1), len(kp2))) * 100

                        # Log the final matching percentage
                        print(f"Debug: Matching percentage between {client_image} and {f}: {percent}%")

                    # If more than one object is detected in both images, perform matching for all objects
                    elif len(client_detected_objects) > 1 and len(comp_detected_objects) > 1:
                        print("Multiple objects detected in both client and comparison images. Performing direct image-to-image matching.")

                        # Normalize without resizing
                        client_cropped = normalize_image(img1)
                        comp_cropped = normalize_image(img2)

                        # Feature detection and description for the normalized client and comparison images
                        kp1, des1 = sift.detectAndCompute(client_cropped, None)
                        kp2, des2 = sift.detectAndCompute(comp_cropped, None)

                        if des1 is None or des2 is None or des1.shape[1] != des2.shape[1]:
                            print(f"Descriptors not found or incompatible shapes in {client_image} and {f}.")
                            percent = fallback_matching(img1, img2)  # new logic to handle the skipping
                            # continue old logic which was skipping the image

                        else:
                            # Descriptor Matching
                            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
                            matches = bf.knnMatch(des1, des2, k=2)

                            good = []
                            for m, n in matches:
                                if m.distance < 0.75 * n.distance:
                                    if m not in good:  # Avoid duplicates
                                        good.append([m])

                            percent = (len(good) / min(len(kp1), len(kp2))) * 100
                        print(f"Matching percentage between {client_image} and {f}: {percent}%")

                    # If one object detected in client image and multiple objects detected in comparison image
                    elif len(client_detected_objects) == 1 and len(comp_detected_objects) > 1:
                        print("One object in client image, multiple objects in comparison image.")

                        best_match_score = 0
                        best_comp_box = None

                        # Get the client object's box
                        client_label, client_confidence, client_box = client_detected_objects[0]
                        (startX_client, startY_client, endX_client, endY_client) = client_box

                        # Crop the client image to the detected object region
                        client_cropped = img1[startY_client:endY_client, startX_client:endX_client]
                        client_cropped = normalize_image(client_cropped)

                        # Feature detection and description for the cropped client image
                        kp1, des1 = sift.detectAndCompute(client_cropped, None)

                        if des1 is None:
                            print("No descriptors found for client image.")
                            return

                        # Iterate through all detected objects in the comparison image
                        for (comp_label, comp_confidence, comp_box) in comp_detected_objects:
                            (startX_comp, startY_comp, endX_comp, endY_comp) = comp_box

                            # Crop the comparison image to the detected object region
                            comp_cropped = img2[startY_comp:endY_comp, startX_comp:endX_comp]
                            comp_cropped = normalize_image(comp_cropped)

                            # Feature detection and description for the cropped comparison object
                            kp2, des2 = sift.detectAndCompute(comp_cropped, None)

                            if des2 is None:
                                continue

                            # Descriptor Matching
                            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
                            matches = bf.knnMatch(des1, des2, k=2)

                            # Apply Lowe's ratio test to filter good matches
                            good = []
                            for m, n in matches:
                                if m.distance < 0.75 * n.distance:
                                    if m not in good:  # Avoid duplicates
                                        good.append([m])

                            percent = (len(good) / min(len(kp1), len(kp2))) * 100
                            print(f"Matching percentage between client object and {comp_label}: {percent}%")

                            if percent > best_match_score:
                                best_match_score = percent
                                best_comp_box = comp_box

                        if best_match_score == 0:
                            print("No good match found for client object.")
                        else:
                            print(f"Best matching object in comparison image: {best_match_score}%")

                    # If one object detected in comparison image and multiple objects detected in client image
                    elif len(comp_detected_objects) == 1 and len(client_detected_objects) > 1:
                        print("One object in comparison image, multiple objects in client image.")

                        best_match_score = 0
                        best_client_box = None

                        # Get the comparison object's box
                        comp_label, comp_confidence, comp_box = comp_detected_objects[0]
                        (startX_comp, startY_comp, endX_comp, endY_comp) = comp_box

                        # Crop the comparison image to the detected object region
                        comp_cropped = img2[startY_comp:endY_comp, startX_comp:endX_comp]
                        comp_cropped = normalize_image(comp_cropped)

                        # Feature detection and description for the cropped comparison object
                        kp2, des2 = sift.detectAndCompute(comp_cropped, None)

                        if des2 is None:
                            print("No descriptors found for comparison image.")
                            return

                        # Iterate through all detected objects in the client image
                        for (client_label, client_confidence, client_box) in client_detected_objects:
                            (startX_client, startY_client, endX_client, endY_client) = client_box

                            # Crop the client image to the detected object region
                            client_cropped = img1[startY_client:endY_client, startX_client:endX_client]
                            client_cropped = normalize_image(client_cropped)

                            # Feature detection and description for the cropped client image
                            kp1, des1 = sift.detectAndCompute(client_cropped, None)

                            if des1 is None:
                                continue

                            # Descriptor Matching
                            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
                            matches = bf.knnMatch(des1, des2, k=2)

                            good = []
                            for m, n in matches:
                                if m.distance < 0.75 * n.distance:
                                    if m not in good:  # Avoid duplicates
                                        good.append([m])

                            percent = (len(good) / min(len(kp1), len(kp2))) * 100
                            print(
                                f"Matching percentage between client object {client_label} and comparison object: {percent}%")

                            if percent > best_match_score:
                                best_match_score = percent
                                best_client_box = client_box

                        if best_match_score == 0:
                            print("No good match found for comparison object.")
                        else:
                            print(f"Best matching object in client image: {best_match_score}%")

                    else:
                        print(
                            f"Object counts don't match or no objects detected in {client_image} and {f}. Performing direct image-to-image matching.")

                        # Normalize without resizing
                        client_cropped = normalize_image(img1)
                        comp_cropped = normalize_image(img2)

                        # Feature detection and description for the normalized client and comparison images
                        kp1, des1 = sift.detectAndCompute(client_cropped, None)
                        kp2, des2 = sift.detectAndCompute(comp_cropped, None)

                        if des1 is None or des2 is None or des1.shape[1] != des2.shape[1]:
                            print(f"Descriptors not found or incompatible shapes in {client_image} and {f}.")
                            percent = fallback_matching(img1, img2)  # new logic to handle the skipping
                            # continue old logic which was skipping the image

                        else:
                            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
                            matches = bf.knnMatch(des1, des2, k=2)

                            good = []
                            for m, n in matches:
                                if m.distance < 0.75 * n.distance:
                                    if m not in good:  # Avoid duplicates
                                        good.append([m])

                            percent = (len(good) / min(len(kp1), len(kp2))) * 100
                        print(f"Matching percentage between {client_image} and {f}: {percent}%")

                    # Store result in MongoDB
                    ts = time.time()
                    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                    mydict = {
                        "client_product_id": "Client",
                        "client_image_name": client_image,
                        "client_image_path": clientsource + client_image,
                        "comp_product_id": "Comp",
                        "comp_image_name": f,
                        "comp_image_path": source + f,
                        "score": percent,
                        "date": st,
                        "defining_status": 0
                    }
                    x = mycol.insert_one(mydict)

                except cv2.error as e:
                    if "(-215:Assertion failed) inv_scale_x > 0" in str(e):
                        print(f"Resize error encountered for {f}: {e}. Attempting direct image-to-image matching.")
                        score = perform_direct_matching(img1, img2)
                        if score == "-":
                            print(f"Direct matching also failed for {client_image} and {f}. Logging with score '-'.")
                            log_skipped_image(f, source, "Comp", "-", mycol)
                        else:
                            log_image_match(client_image, f, score, clientsource, source, mycol)
                    else:
                        print(f"Unhandled OpenCV error for image {f}: {e}")
                        log_skipped_image(f, source, "Comp", "-", mycol)
                except Exception as e:
                    print(f"Error processing comparison image {f}: {e}")
                    continue
    except Exception as e:
        print(f"Error processing client image {client_image}: {e}")


# Helper function for direct image-to-image matching
def perform_direct_matching(img1, img2):
    try:
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None or des1.shape[1] != des2.shape[1]:
            print("Descriptors not found or incompatible shapes.")
            percent = fallback_matching(img1, img2)  # new logic to handle the skipping
            # return "-" old logic which was skipping the image
        else:
            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)

            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    if m not in good:  # Avoid duplicates
                        good.append([m])

            percent = (len(good) / min(len(kp1), len(kp2))) * 100
        print(f"Direct matching percentage: {percent}%")
        return percent

    except Exception as e:
        print(f"Direct matching failed: {e}")
        return "-"


# Helper function to log skipped images in MongoDB
def log_skipped_image(image_name, source, product_id, score, mycol):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    mydict = {
        "product_id": product_id,
        "image_name": image_name,
        "image_path": os.path.join(source, image_name),
        "score": score,
        "date": st,
        "defining_status": 0
    }
    mycol.insert_one(mydict)
    print(f"Logged skipped image: {image_name} with score '{score}'")


# Helper function to log matched images in MongoDB
def log_image_match(client_image, comp_image, score, clientsource, source, mycol):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    mydict = {
        "client_product_id": "Client",
        "client_image_name": client_image,
        "client_image_path": clientsource + client_image,
        "comp_product_id": "Comp",
        "comp_image_name": comp_image,
        "comp_image_path": source + comp_image,
        "score": score,
        "date": st,
        "defining_status": 0
    }
    mycol.insert_one(mydict)
    print(f"Logged image match between {client_image} and {comp_image} with score '{score}'")


def main(max_workers=50):
    image_count = 0  # Initialize image_count at the start of the function
    for filename in os.listdir(directory):
        # comPath = os.path.join(directory, filename, "Comp")
        comPath = os.path.join(directory, filename, "Amazon")
        clientPath = os.path.join(directory, filename, "Client")
        source = comPath + "\\"
        clientsource = clientPath + "\\"

        collectionDefaultName = "ImageMatching_Target_Test_Setup"
        # collectionDefaultName = "Aspect_Issue_Review"
        mycol = mydb[collectionDefaultName]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for root, dirs, Cfilenames in os.walk(clientsource):
                for clientImage in Cfilenames:
                    futures.append(executor.submit(process_client_image, clientImage, clientsource, source, mycol))
                    # Increment image count
                    image_count += 1

                    # Clear console after processing every 50 images
                    if image_count % 50 == 0:
                        clear_console()
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception occurred: {e}")


if __name__ == "__main__":
    main(max_workers=50)

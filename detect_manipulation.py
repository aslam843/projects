import cv2

def detect_manipulation(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (21, 21), 0)
    difference = cv2.absdiff(gray_image, blurred)
    _, thresholded = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        print("Manipulated Image")
    else:
        print("Real Image")

# Provide the path to the image you want to analyze
image_path = "images/allen_solly_mens_regular_fit_polo.jpg"
detect_manipulation(image_path)
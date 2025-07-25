# import the necessary packages
from imutils import paths
import argparse
import cv2

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
	help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

# loop over the input images
for imagePath in paths.list_images(args["images"]):
	# load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
	# method
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	text = "Not Blurry"

	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
	if fm < args["threshold"]:
		text = "Blurry"

	# show the image
	cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	cv2.imshow("Image", image)
	key = cv2.waitKey(0)

#Command > python blur_detection.py -i images -t 150
#The detect_blur_and_bright_spot function takes an image path and a threshold value as input. 
#It performs the following steps:
	#Reads the image and converts it to grayscale.
	#Applies binary thresholding to the grayscale image to detect bright spots.
	#Applies the Laplacian filter to the grayscale image to detect edges and calculate the variance.
	#Calculates the maximum intensity value and the variances of the binary and Laplacian images.
	#Initializes result variables for blur and bright spot conditions.
	#Checks the variance of the Laplacian image against the threshold to determine if the image is blurry.
	#Checks the variance of the binary image to determine if it contains bright spots.
	#Adds labels indicating the blur and bright spot conditions to the image.
	#Displays the labeled image
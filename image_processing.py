from PIL import Image, ImageFilter
filename = "images/cat.png"
with Image.open(filename) as img:
	img.load()

img.show()
print("Format:", img.format, "Size:", img.size, "Mode:", img.mode)

#cropped_img = img.crop((50, 50, 100, 100))
#print("cropped_img Size:", cropped_img.size)
#cropped_img.show()
#cropped_img.save("cropped_image.jpg")


#low_res_img = cropped_img.resize((cropped_img.width // 4, cropped_img.height // 4))
#low_res_img.show()

#low_res_img = cropped_img.reduce(4)
#low_res_img.show()

#converted_img = img.transpose(Image.FLIP_TOP_BOTTOM)
#converted_img.show()

#rotated_img = img.rotate(45)
#rotated_img.show()

#rotated_img = img.rotate(45, expand=True)
#rotated_img.show()

#cmyk_img = img.convert("CMYK")
#gray_img = img.convert("L")  # Grayscale

#cmyk_img.show()
#gray_img.show()

#red, green, blue = img.split()
#zeroed_band = red.point(lambda _: 0)
#red_merge = Image.merge("RGB", (red, zeroed_band, zeroed_band))
#green_merge = Image.merge("RGB", (zeroed_band, green, zeroed_band))
#blue_merge = Image.merge("RGB", (zeroed_band, zeroed_band, blue))
#red_merge.show()
#green_merge.show()
#blue_merge.show()

#blur_img = img.filter(ImageFilter.BLUR)
#blur_img.show()

#img.filter(ImageFilter.BoxBlur(5)).show()
#img.filter(ImageFilter.BoxBlur(20)).show()
#img.filter(ImageFilter.GaussianBlur(20)).show()

#sharp_img = img.filter(ImageFilter.SHARPEN)
#sharp_img.show()

#smooth_img = img.filter(ImageFilter.SMOOTH)
#smooth_img.show()

#img_gray = img.convert("L")
#edges = img_gray.filter(ImageFilter.FIND_EDGES)
#edges.show()
#img_gray_smooth = img_gray.filter(ImageFilter.SMOOTH)
#edges_smooth = img_gray_smooth.filter(ImageFilter.FIND_EDGES)
#edges_smooth.show()

#edge_enhance = img_gray_smooth.filter(ImageFilter.EDGE_ENHANCE)
#edge_enhance.show()


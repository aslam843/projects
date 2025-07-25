from moviepy.editor import *

clip1 = VideoFileClip("Animated_Video_1.mp4").subclip(0,5)

clip2 = VideoFileClip("Animated_Video_2.mp4").subclip(0,5)

clip2 = clip2.set_position((45,150))

final_video = concatenate_videoclips([clip1,clip2])
final_video.write_videofile("New_Animated_Video_2.mp4",codec='libx264')
print("Done")


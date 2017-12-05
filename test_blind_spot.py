from blind_spot import process_frame_for_video
from moviepy.editor import VideoFileClip

test_out_file2 = 'BlindSpot_out.mp4'
clip_test2 = VideoFileClip('BlindSpot.mp4')
clip_test_out2 = clip_test2.fl_image(process_frame_for_video)
clip_test_out2.write_videofile(test_out_file2, audio=False)
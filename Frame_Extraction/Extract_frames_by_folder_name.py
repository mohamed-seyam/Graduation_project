import os
from pathlib import Path


# video_directory = './test'
# frames_directory = "test_frames"

#####################################################################
## directory :: the input directory name of videos to be extracted ##
video_directory = './Optical_Flow_dataset'
#####################################################################


##################################################################
## frames_dir :: the output directory name for extracted frames ##
frames_directory = "extracted_frames_optical_flow"
##################################################################


def mkdir_ifnotexists(dir):
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True, exist_ok=True)
    else:
        print(f'{dir} Directory Already Exists')
        
def video2frames(video_file_path, video_frames_path, video_name):
    mkdir_ifnotexists(video_frames_path)
    print(video_file_path)
    print(video_frames_path)
    cmd = "ffmpeg -i %s -start_number 0 -vsync 0 %s/%s_%%05d.png" % (f'{video_directory}/{video_file_path}',f'./{video_frames_path}', video_name)
    os.system(cmd)



mkdir_ifnotexists(frames_directory)


allfiles = os.listdir(video_directory)
videos = [ fname for fname in allfiles if fname.endswith('.avi')]
# print(videos)


for i, video in enumerate(videos):
    print('{}/{} - {}'.format(i+1, len(videos), video))
    head_tail = os.path.split(video)
    video_name= os.path.splitext(os.path.basename(head_tail[1]))
    ## frame_pth :: This line create frames folder for each video ## "test_frames/video0_frames"
    ## {video_name[0]}_frames :: the name of the frames directory
    frame_pth = f'{frames_directory}/{video_name[0]}_frames'  
    video2frames(video, frame_pth , video_name[0])
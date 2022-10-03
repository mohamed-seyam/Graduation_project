import os
from pathlib import Path

video_directory = './data/'
frames_directory = "frames"

def mkdir_ifnotexists(dir):
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True, exist_ok=True)
    else:
        print(f'{dir} Directory Already Exists')

def video2frames(video_file_path, video_name):
    cmd = "ffmpeg -i %s -start_number 0 -vsync 0 %s/%s_%%05d.png" % (f'{video_directory}/{video_file_path}', frames_directory, video_name)
    os.system(cmd)

mkdir_ifnotexists(frames_directory)

allfiles = os.listdir(video_directory)
videos = [ fname for fname in allfiles if fname.endswith('.avi')]

for i, video in enumerate(videos):
    print('{}/{} - {}'.format(i+1, len(videos), video))
    head_tail = os.path.split(video)
    video_name= os.path.splitext(os.path.basename(head_tail[1]))
    video2frames(video, video_name[0])
import tkinter as tk
import tkinter.font as font
import cv2
from PIL import Image, ImageTk
from datetime import datetime
from tkinter import filedialog
from model import FCBFormer
import torch
from torchvision import transforms
import numpy as np


class VideoManager:
    def __init__(self, video):
        self.video = cv2.VideoCapture(video)
        self.frame_count = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.count = 0
        self.prev_frame = None
        
    def get_next_frame(self):
        if self.count < self.frame_count:
            _, frame = self.video.read()
            frame = inference(frame)
            self.prev_frame = frame
            self.count += 1
            return frame, False
        else:
            return self.prev_frame, True

    def restart(self):
        self.count = 0
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    def get_frame(self):
        return True, self.prev_frame
    
    def close(self):
        self.video.release()

vid = None
stop_play = False

device = "cuda" if torch.cuda.is_available() else "cpu"
model = FCBFormer(352).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device)["model_state_dict"])
model.eval()

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((352, 352), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

@torch.no_grad()
def inference(img):
    ori_size = img.shape[:2]
    input_img = transform_test(img).unsqueeze(0).to(device)
    res = (model(input_img).sigmoid() > 0.2).cpu().squeeze().numpy().astype("uint8")
    res = cv2.resize(res, ori_size[::-1])
    color = np.array([0, 255, 0], dtype='uint8')
    masked_img = np.where(res[..., None], color, img)
    out = cv2.addWeighted(img, 0.8, masked_img, 0.2, 0)
    return out
    

def open_camera():
    if vid is None or not vid.isOpened():
        return
    
    _, frame = vid.read()
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(opencv_image)
    photo_image = ImageTk.PhotoImage(image=captured_image)
    label_widget.photo_image = photo_image
    label_widget.configure(image=photo_image)
    label_widget.after(10, open_camera)
  
  
window = tk.Tk()
window.title('Poly Segmentation')
window.geometry('1280x730')
window.resizable(width=0, height=0)

word_size = font.Font(family='Helvetica', size=10, weight='bold')

def release_camera():
    global vid
    camera_btn["state"] = "normal"
    camera_btn["bg"]="white"
    video_btn["bg"]="white"
    
    snapshot_btn["state"] = "disabled"
    snapshot_btn["bg"]="#D0D0D0"
    start_btn["state"]="disabled"
    start_btn["bg"]="#D0D0D0"
    restart_btn["state"]="disabled"
    restart_btn["bg"]="#D0D0D0"
    stop_btn["state"]="disabled"
    stop_btn["bg"]="#D0D0D0"
    upload_btn["state"]="disabled"
    upload_btn["bg"]="#D0D0D0"
    
    
    if vid is not None and isinstance(vid, VideoManager):
        vid.close()
        label_widget.config(image="")
        vid = None

def release_video():
    global vid
    video_btn["state"]="normal"
    video_btn["bg"]="white"
    camera_btn["bg"]="white"
    
    snapshot_btn["state"] = "disabled"
    snapshot_btn["bg"]="#D0D0D0"
    start_btn["state"]="disabled"
    start_btn["bg"]="#D0D0D0"
    restart_btn["state"]="disabled"
    restart_btn["bg"]="#D0D0D0"
    stop_btn["state"]="disabled"
    stop_btn["bg"]="#D0D0D0"
    upload_btn["state"]="disabled"
    upload_btn["bg"]="#D0D0D0"
    
    if vid is not None and vid.isOpened():
        vid.release()
        vid = None
        label_widget.config(image="")

def screenshot():
    timestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(timestr)
    filename = "C:\\Users\\User\\Desktop\\polyp\\" + timestr + ".jpg"
    if vid is None:
        return
    
    if isinstance(vid, VideoManager):
        return_value, image = vid.get_frame()
    else:
        return_value, image = vid.read()
    
    if return_value:
        try:
            cv2.imwrite(filename, image)
        except Exception as e:
            print(str(e))
            
   
def hide_window():
   # hiding the tkinter window while taking the screenshot
   window.after(1000, screenshot)


def oas():
    global vid
    sfname = filedialog.askopenfilename(title='Choose a video file',
                                        filetypes=[
                                            ('video Files','*.mp4'),
                                            ])

    vid = VideoManager(sfname)
    get_single_frame()
    start_btn["state"]="normal"
    start_btn["bg"]="white"
    snapshot_btn["state"]="normal"
    snapshot_btn["bg"]="white"
    upload_btn["state"]="disabled"
    upload_btn["bg"]="#D0D0D0"
    

def get_single_frame():
    global vid
    
    if vid is None and isinstance(vid, VideoManager):
        return
    frame, end = vid.get_next_frame()
    
    if frame is not None and not end:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label_widget.imgtk = imgtk
        label_widget.configure(image=imgtk)
        return True
    return False
        
    
def get_frames():
    global stop_play
    if stop_play:
        return
    res = get_single_frame()
    if not res:
        return
    window.after(30, get_frames)
    
def start():
    global stop_play
    start_btn["state"] = "disable"
    start_btn["bg"] = "#D0D0D0"
    stop_btn["state"]="normal"
    stop_btn["bg"]="white"
    stop_play = False
    restart_btn["state"]="normal"
    restart_btn["bg"]="white"
    get_frames()
    

def stop():
    global stop_play
    stop_play = True
    start_btn["state"] = "normal"
    start_btn["bg"] = "white"
    stop_btn["state"]= "disabled"
    stop_btn["bg"] = "#D0D0D0"

def restart():
    if vid is not None and isinstance(vid, VideoManager):
        vid.restart()
        get_single_frame()
        stop()
    restart_btn["state"] = "disable"
    restart_btn["bg"] = "#D0D0D0"
        
def choose_camera():
    global vid
    video_btn["state"] = "disabled"
    upload_btn["state"] = "disabled"
    camera_btn["bg"]="#FFFF93"
    video_btn["bg"]="#D0D0D0"
    upload_btn["bg"]="#D0D0D0"
    snapshot_btn["state"]="normal"
    snapshot_btn["bg"]="white"
    # start_btn["state"]="normal"
    # start_btn["bg"]="white"
    # restart_btn["state"]="normal"
    # restart_btn["bg"]="white"
    # stop_btn["state"]="normal"
    # stop_btn["bg"]="white"
    
    
    if vid is None or not vid.isOpened():
        vid = cv2.VideoCapture(0)
    

def choose_video():
    camera_btn["state"] = "disabled"
    snapshot_btn["state"] = "disabled"
    video_btn["bg"]="#FFFF93"
    camera_btn["bg"]="#D0D0D0"
    start_btn["state"]="disabled"
    start_btn["bg"]="#D0D0D0"
    restart_btn["state"]="disabled"
    restart_btn["bg"]="#D0D0D0"
    stop_btn["state"]="disabled"
    stop_btn["bg"]="#D0D0D0"
    upload_btn["state"]="normal"
    upload_btn["bg"]="white"
    snapshot_btn["bg"]="#D0D0D0"
    

def activate_video():
    if camera_btn["state"] == "disabled":
        release_camera()
    elif video_btn["state"] == "normal":
        choose_video()
        
        
def activate_camera():
    global vid
    if video_btn["state"] == "disabled":
        release_video()
    elif camera_btn["state"] == "normal":
        choose_camera()
        open_camera()
        

def close():
   window.quit()
    

label_widget = tk.Label(window)
label_widget.place(x=500, y=300
                   , anchor='center')

start_btn = tk.Button(
    text="Start",
    width=30,
    height=5,
    bg="#D0D0D0",
    fg="black",
    state="disabled",
    command=start
)

restart_btn = tk.Button(
    text="Restart",
    width=30,
    height=5,
    bg="#D0D0D0",
    fg="black",
    state="disabled",
    command=restart
)

stop_btn = tk.Button(
    text="Stop",
    width=30,
    height=5,
    bg="#D0D0D0",
    fg="black",
    state="disabled",
    command=stop
)

upload_btn = tk.Button(
    text="Upload",
    width=30,
    height=5,
    bg="#D0D0D0",
    fg="black",
    state="disabled",
    command=oas
)

snapshot_btn = tk.Button(
    text="Snapshot",
    width=30,
    height=5,
    bg="#D0D0D0",
    fg="black",
    state="disabled",
    command=hide_window
)


close_btn = tk.Button(
    text="Close",
    width=30,
    height=5,
    bg="white",
    fg="black",
    command=close
)

video_btn = tk.Button(
    text="Video",
    width=30,
    height=5,
    bg="white",
    fg="black",
    command=activate_video
)

camera_btn = tk.Button(
    text="Camera",
    width=30,
    height=5,
    bg="white",
    fg="black",
    command=activate_camera
)


start_btn['font'] = word_size
start_btn.place(x=80, y=600)

restart_btn['font'] = word_size
restart_btn.place(x=380, y=600)

stop_btn['font'] = word_size
stop_btn.place(x=680, y=600)


video_btn['font'] = word_size
video_btn.place(x=980, y=80)

camera_btn['font'] = word_size
camera_btn.place(x=980, y=210)

upload_btn['font'] = word_size
upload_btn.place(x=980, y=340)


snapshot_btn['font'] = word_size
snapshot_btn.place(x=980, y=470)

close_btn['font'] = word_size
close_btn.place(x=980, y=600)



window.mainloop()
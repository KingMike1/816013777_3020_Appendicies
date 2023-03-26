import tkinter as tk
from PIL import Image, ImageTk
import functools

import cv2              # used so that OpenCv can be accessed (OpenCv for visualizations and tracking)

from absl import flags
import sys
FLAGS = flags.FLAGS

# Yolo  uses the command line interface, therefore a flag setting for using YOLOv3 in tensorflow must be imported.
FLAGS(sys.argv)

import os
import time                             # used to calculate the frames per second
import numpy as np                      # we need numpy
# import cv2                              # used so that OpenCv can be accessed (OpenCv for visualizations and tracking)
import matplotlib.pyplot as plt         # used for the colour map
import tensorflow as tf                 # import the TensorFlow Framework

# -------------------------------------- Importation of Necessary Files From Folders------------------------------------

from yolov3_tf2.models import YoloV3                 # From inside yolo v3_tf2 folder, models.py, import YoloV3
from yolov3_tf2.dataset import transform_images      # Used for resizing the Yolo image format
from yolov3_tf2.utils import convert_boxes           # Used to convert the bounding boxes back to the DEEP SORT format

from deep_sort import preprocessing                  # Used for none maximum suppression
from deep_sort import nn_matching                    # For setting up the deep association matrix
from deep_sort.detection import Detection            # This assist with object detections
from deep_sort.tracker import Tracker                # Used for writing the Track information
from tools import generate_detections as gdet        # Used for the features generation encoder
from _collections import deque
from collections import Counter
import datetime
import math

# --------------------------Initialization of Yolo and Loading of Yolo Class Names and Weights--------------------------

class_names = [c.strip() for c in open('./data/labels/obj.names').readlines()]  # load class names into a list
yolo = YoloV3(classes=len(class_names))     # input the number of classes. This is equal to the length of the class file
yolo.load_weights('./weights/yolov3_custom.tf')    # Load weights into these models

max_cosine_distance = 0.5                   # Used to determine whether the detected objects are the same or not
#                                             Cosine_distance > 0.5 means the features are very similar
nn_budget = None                      # Used to for a creating and storing a gallery. Features are extracted using a DNN
nms_max_overlap = 0.8                 # To avoid too many detections for the same object

# -------------------------------------------Generating DeepSORT Encoder Functions--------------------------------------

model_filename = 'model_data/mars-small128.pb'                      # This is a pre-trained CNN for tracking pedestrians
encoder = gdet.create_box_encoder(model_filename, batch_size=1)     # Encoder for feature generations
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)    # Association Matrix
tracker = Tracker(metric)                                          # Pass the association matrix to the DeepSORT tracker
# ----------------------------------------------------------------------------------------------------------------------

name1 = './data/video/cctv_inside.mp4'
name2 = './data/video/cctv_street.mp4'
name3 = './data/video/cctv_alley.mp4'
name4 = './data/video/cctv_mall.mp4'

# --------------------------------------------------Defining Functions--------------------------------------------------

# This function computes the center of the bounding boxes
def tlbr_center_point(box):
    x1, y1, x2, y2 = box
    center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))  # minus y coordinates to get proper xy format
    return center_point


def vector_angle(center_point, previous_center_point):
    x = center_point[0] - previous_center_point[0]
    y = center_point[1] - previous_center_point[1]
    return math.degrees(math.atan2(y, x))


# Check to see if the center points of the bounding boxes intersect with the ROI gate line
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
# ----------------------------------------------------------------------------------------------------------------------

# def trackcalls(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         wrapper.has_been_called = True
#         return func(*args, **kwargs)
#     wrapper.has_been_called = False
#     return wrapper

def trackcalls(func):
    @functools.wraps(func)
    def wrapper(*args):
        wrapper.has_been_called = True
        return func(*args)
    wrapper.has_been_called = False
    return wrapper

def app():
    present_date = datetime.datetime.now().date()
    count_dictionary = {}  # initiate dictionary for storing counts

    total_counter = 0
    Exit_count = 0
    Entered_count = 0

    class_counter = Counter()  # stores counts of each detected class
    already_counted = deque(maxlen=50)  # temporary memory for storing counted IDs
    intersect_info = []  # initialize intersection list

    memory = {}

    points = [deque(maxlen=30) for _ in range(1000)]  # data points cannot be larger than 30

    # ---------------------------Capturing of Videos and Assigning of File Names to Save The Outputs--------------------

    if cmd.has_been_called:
        vid = cv2.VideoCapture('./data/video/cctv_inside.mp4')
    elif cmd1.has_been_called:
        vid = cv2.VideoCapture('./data/video/cctv_street.mp4')
    elif cmd2.has_been_called:
        vid = cv2.VideoCapture('./data/video/cctv_alley.mp4')
    elif cmd3.has_been_called:
        vid = cv2.VideoCapture('./data/video/cctv_mall.mp4')
    else:
        cmd4()

    codec = cv2.VideoWriter_fourcc(*'XVID')                             # record outputs as an avi file.

    # Define the video fps, width and height
    vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
    vid_width, vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output / Save the video in the video folder inside the data folder.
    out = cv2.VideoWriter('./data/video/results.avi', codec, vid_fps, (vid_width, vid_height))

    while True:  # Make a while loop to capture all the frames from the videos
        _, img = vid.read()  # read the frames one at a time
        if img is None:  # At the end of the video if there is no image
            print('Video Completed')  # Print video completed and then break the while loop
            break

        # Transform images captured to be placed into Yolo Prediction Models
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert colour codes from BGR (in OpenCv) to RGB (in TF)
        img_in = tf.expand_dims(img_in, 0)  # Expand dimensions using tensor expand dimensions
        img_in = transform_images(img_in, 416)  # Resize image for the YoloV3

        t1 = time.time()  # Start the timer

        """
            Pass to the Yolo predictions ( This returns numpy np arrays that contains the boxes, scores, classes and nums)
            e.g. boxes, 3D shape (1, 100, 4) returns a 3D shape. max 100 bboxes per image and 4 numpy arrays. For the 
            scores, it will return a 2D shaped. 
            The detected objects confidence scores will be returned in a shutter form. Remaining objects will have 0.
            e.g. classes, 2D shape (1, 100) Detected objects classes
            e.g. nums, 1D shape(1,)
        """
        boxes, scores, classes, nums = yolo.predict(img_in)

        classes = classes[0]  # first row of the numpy empty arrays
        names = []

        for i in range(len(classes)):  # loop through the classes
            names.append(class_names[int(classes[i])])  # Give class names with the integer of the corresponding classes
        names = np.array(names)  # return everything into a numpy array
        converted_boxes = convert_boxes(img, boxes[0])  # Convert boxes into a list. Use first row of the boxes
        #                                                     Helps scale back to the original size of the image

        # ------------------------------------Use of DeepSORT Detection Functions-------------------------------------------

        features = encoder(img, converted_boxes)  # Use encoder to generate the features vector for each detected object

        # Pass the detected boxes, scores, names and features into the detection functions
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(converted_boxes, scores[0], names, features)]

        # For non-maximum suppression, we need box, scores and classes
        boxs = np.array([d.tlwh for d in detections])  # Bounding box
        scores = np.array([d.confidence for d in detections])  # Confidence
        classes = np.array([d.class_name for d in detections])  # Class name

        # The indices tell us which boxes can be disregarded.
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]  # Passed back to detections to remove the redundancy
        # ------------------------------------------------------------------------------------------------------------------

        # ------------------------------------------------Call Tracker------------------------------------------------------
        # Call tracker
        tracker.predict()  # Used to propagate the track distributions one time step forward based on Kalman Filtering

        # Updates the  detections. This updates the feature set and Kalman parameters. Additionally, it updates the target
        # disappearance and the new target appearance.
        tracker.update(detections)
        # ------------------------------------------------------------------------------------------------------------------

        # ----------------------------------------------Visualizing the Result----------------------------------------------

        cmap = plt.get_cmap('tab20b')  # generating color map. Dictionary that Maps numbers to colours
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        """ Multiple Object Tracking Gate (ROI) """
        # ROI: Draw yellow gate (counting or trip) line
        line = [(0, int(0.5 * img.shape[0])), (int(img.shape[1]), int(0.5 * img.shape[0]))]
        cv2.line(img, line[0], line[1], (0, 255, 255), thickness=2)
        # ------------------------------------------------------------------------------------------------------------------

        # Looping all the results from the tracker
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:  # If Kalman Filtering could not assign a track
                continue  # and if there is no update then continue (skip the track)
            bbox = track.to_tlbr()  # This bbox format will be used for x, y dimensions
            track_class_name = track.class_name  # most common detection class for track
            class_name = track.get_class()  # Get the corresponding classes
            color = colors[int(track.track_id) % len(colors)]  # Assign colour based on colour code previously created.
            color = [i * 255 for i in color]  # Convert back to standard RGB scale

            """ 
                If the counting line intersects the line drawn from the current tracked position to the most recent tracked 
                position, the tracked object is counted. As long as objects keep the same tracking ID after crossing the
                line, they will counted.
            """
            # Center of the bounding box
            center_point = tlbr_center_point(bbox)

            # get center point respective to bottom-left
            origin_center_point = (center_point[0], img.shape[0] - center_point[1])

            if track.track_id not in memory:
                memory[track.track_id] = deque(maxlen=2)

            memory[track.track_id].append(center_point)
            previous_center_point = memory[track.track_id][0]

            origin_previous_center_point = (previous_center_point[0], img.shape[0] - previous_center_point[1])

            # Change Color of Line to Green when object crosses
            cv2.line(img, center_point, previous_center_point, (0, 255, 0), 2)

            # Add to counter and get intersection details
            if intersect(center_point, previous_center_point, line[0],
                         line[1]) and track.track_id not in already_counted:
                class_counter[track_class_name] += 1
                total_counter += 1
                # last_track_id = track.track_id

                # draw red line
                cv2.line(img, line[0], line[1], (0, 255, 0), 2)

                # To avoid double counting (temporary memory for storing counted IDs)
                already_counted.append(track.track_id)  # Set already counted for ID to true.

                # Compute the time at which the intersection occurs
                intersection_time = datetime.datetime.now() - datetime.timedelta(
                    microseconds=datetime.datetime.now().microsecond)
                angle = vector_angle(origin_center_point, origin_previous_center_point)
                intersect_info.append([track_class_name, origin_center_point, angle, intersection_time])

                if class_name == 'person':  # check to see if the detected class is a person

                    # ----------------------------------Directional Counts--------------------------------------------------
                    """ 
                        Upon intersection with the line, the angle of the tracked object is computed with regards to the
                        the positive x-axis. An angle is considered upwards if it is within +180 degrees and downwards if 
                        it is within -180 degrees.
                     """
                    if angle > 0:
                        Exit_count += 1
                    if angle < 0:
                        Entered_count += 1
                    # ------------------------------------------------------------------------------------------------------

            # --------------------------------------Drawing of Bounding Boxes-----------------------------------------------
            """
                This section draws the bounding boxes around the detected objects (tracked objects). The class name and ID
                is also placed above the bounding boxes of each of the detected objects.
            """
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])

            # Draw bounding box around image.
            # Put bbox in a rectangle format ((top left, bottom left), (top right, bottom right), thickness)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            y1_ = int(bbox[1] - 30)
            # Put class name and tracker id rectangle above bbox rectangle
            cv2.rectangle(img, (x1, y1_), (x1 + (len(class_name) + len(str(track.track_id))) * 17, y1), color, -1)

            # Put the text inside_of the rectangle
            y1_2 = int(bbox[1] - 10)
            cv2.putText(img, class_name + "-" + str(track.track_id), (x1, y1_2), 0, 0.75, (255, 255, 255), 2)

            # -----------------------------------Visualize Historical Tracking Trajectory-------------------------------
            """To remove trajectory, uncomment this section of code"""
            middle = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            points[track.track_id].append(middle)  # This deque helps store all the center points

            # Drawing of the motion path
            for k in range(1, len(points[track.track_id])):
                # if there is no track for previous tracker ID or there is no track for current tracker ID the break loop
                if points[track.track_id][k - 1] is None or points[track.track_id][k] is None:
                    continue
                thickness = int(
                    np.sqrt(64 / float(k + 1) * 2))  # keep the close distance lines thinner and further ones thicker
                cv2.line(img, points[track.track_id][k - 1], (points[track.track_id][k]), color, thickness)

            # ----------------------------------------------------------------------------------------------------------
            """
                Delete the memory of old tracks. Must be larger than the number of tracked objects in the frame
            """
            if len(memory) > 50:
                del memory[list(memory)[0]]

            # --------------------------------------Display the Overall Count Results---------------------------------------
            """
                Output the Total total amount of persons that entered and exited the area. Also output the total amount of 
                persons counted.
            """
            cv2.putText(img, "Total: {} (Exited: {}, Entered: {})".format(str(total_counter), str(Exit_count),
                                                                          str(Entered_count)), (0, 130), 0, 1,
                        (0, 0, 255), 2)

            # --------------------------------------------------------------------------------------------------------------
            # -----------------------------------Storing Counting Results to text File--------------------------------------
            # Calculate current time
            now = datetime.datetime.now()
            approx_now = now - datetime.timedelta(microseconds=now.microsecond)  # round of to the nearest second
            present_minute = now.time().minute

            if present_minute == 0 and len(count_dictionary) > 1:
                count_dictionary = {}  # reset counts every hour
            else:
                # write counts to file for every set interval of the hour
                write_interval = 1  # write counts to file every 1 minutes of the hour
                if present_minute % write_interval == 0:  # write to file once only every write_interval minutes
                    if present_minute not in count_dictionary:
                        count_dictionary[present_minute] = True
                        total_filename = 'Total counts for {}, {}.txt'.format(present_date, class_name)
                        counts_folder = './counts/'
                        if not os.access(counts_folder + str(present_date) + '/total', os.W_OK):
                            os.makedirs(counts_folder + str(present_date) + '/total')
                        total_count_file = open(counts_folder + str(present_date) + '/total/' + total_filename, 'a')
                        print('{} writing...'.format(approx_now))
                        print('Writing the total count ({}) and directional counts to file.'.format(total_counter))
                        total_count_file.write('{}, {}, {}, {}\n'.format(str(approx_now),
                                                                         str(total_counter), Exit_count, Entered_count))
                        total_count_file.close()


        fps = 1. / (time.time() - t1)  # for every frame print out the fps
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30), 0, 1, (0, 0, 255), 2)  # put at top left hand corner
        cv2.resizeWindow('output', 1024, 768)  # resize output window
        cv2.imshow('output', img)  # show output
        # Save the image for evey frame in the video.
        out.write(img)

        if cv2.waitKey(1) == ord('q'):
            break  # break the while loop
    vid.release()  # release the video
    out.release()  # release the output

    # Terminate all windows
    cv2.destroyAllWindows()



root = tk.Tk()
root.geometry('500x400')
root.resizable(width=False, height=False)   # Fixed Size Window
root.title('Person Tracker Remote')

@trackcalls
def cmd():
    print('Opening the video stream: cctv.inside.mp4')
    app()


@trackcalls
def cmd1():
    print('Opening the video stream: cctv.street.mp4')
    app()


@trackcalls
def cmd2():
    print('Opening the video stream: cctv.alley.mp4')
    app()


@trackcalls
def cmd3():
    print('Opening the video stream: cctv.mall.mp4')
    app()


def cmd4():
    print('Exiting Program')
    # time.sleep(5)
    sys.exit()


main_photo2 = Image.open("./data/app_screens/main_app_screen.png")
resized_image = main_photo2.resize((500,400),Image.ANTIALIAS)
converted_image = ImageTk.PhotoImage(resized_image)


def home_page():
    home_frame = tk.Frame(main_frame)

    lb = tk.Label(home_frame, image=converted_image, width=500, height=400,bg="black", fg="yellow")
    lb.pack()

    home_frame.pack(side=tk.LEFT)


def videos_page():
    videos_frame = tk.Frame(main_frame)

    lb = tk.Label(videos_frame, text='Select desired Pre-Recorded Video.', font=('Bold', 12))
    lb.pack()

    videos_frame.pack(pady=20)


def about_page():
    about_frame = tk.Frame(main_frame)

    lb = tk.Label(about_frame, text='Person Tracker Remote is a highly efficient and\n'
                                    'expandable system that offers a visual interface\n'
                                    'to track and count the number of people entering\n'
                                    'and departing an area thus supporting security\n'
                                    'of that area.', font=('Bold', 12))
    lb.pack()

    about_frame.pack(pady=20)


# Function to hide indicator
def hide_indicators():
    home_indicate.config(bg='#c3c3c3')
    videos_indicate.config(bg='#c3c3c3')
    about_indicate.config(bg='#c3c3c3')
    # exit_indicate.config(bg='#c3c3c3')


def delete_pages():
    for frame in main_frame.winfo_children():
        frame.destroy()


# Function to show selected button
def indicate(lb, page):
    hide_indicators()
    lb.config(bg='#158aff')
    delete_pages()
    page()


options_panel = tk.Frame(root, bg='#c3c3c3')

# Page Action Buttons in Panel
home_bttn = tk.Button(options_panel, text='Home', font=('Bold', 15),
                      fg='#158aff', bd=3, relief='raised',overrelief='groove', bg='#c3c3c3',cursor='hand2',
                      command=lambda: indicate(home_indicate, home_page))

home_bttn.place(x=10,y=50)

home_indicate = tk.Label(options_panel, text='', bg='#158aff')
home_indicate.place(x=3, y=50, width=5, height=40)


videos_bttn = tk.Button(options_panel, text='Videos', font=('Bold', 15),
                      fg='#158aff', bd=3, relief='raised',overrelief='groove', bg='#c3c3c3',cursor='hand2',
                      command=lambda: [indicate(videos_indicate, videos_page),
                                        bttn(45,60,'cctv_inside.mp4','#ffcc66',"#141414",cmd),
                                        bttn(45,130,'cctv_street.mp4','#25dae9',"#141414",cmd1),
                                        bttn(45,195,'cctv_alley.mp4','#f86263',"#141414",cmd2),
                                        bttn(45,260,'cctv_mall.mp4','#ffa157',"#141414",cmd3)])

videos_bttn.place(x=10,y=100)

videos_indicate = tk.Label(options_panel, text='', bg='#c3c3c3')
videos_indicate.place(x=3, y=100, width=5, height=40)

about_bttn = tk.Button(options_panel, text='About', font=('Bold', 15),
                       fg='#158aff', bd=3, relief='raised',overrelief='groove', bg='#c3c3c3', cursor='hand2',
                       command=lambda: indicate(about_indicate, about_page))

about_bttn.place(x=10,y=150)

about_indicate = tk.Label(options_panel, text='', bg='#c3c3c3')
about_indicate.place(x=3, y=150, width=5, height=40)


exit_bttn = tk.Button(options_panel, text='Exit', font=('Bold', 15),
                      fg='#158aff', bd=3, relief='raised',overrelief='groove', bg='#c3c3c3',cursor='hand2',
                      command=cmd4)

exit_bttn.place(x=10,y=200)

# exit_indicate = tk.Label(options_panel, text='', bg='#c3c3c3')
# exit_indicate.place(x=3, y=200, width=5, height=40)


options_panel.pack(side=tk.LEFT)
options_panel.pack_propagate(False)
options_panel.configure(width=100, height=400)

main_frame =tk.Frame(root, highlightbackground='black',
                     highlightthickness=2)

main_frame.pack(side=tk.LEFT)
main_frame.pack_propagate(False)
main_frame.configure(width=400, height=500)

main_photo2_label = tk.Label(main_frame,image=converted_image, width=500, height=400,
                             bg="black", fg="yellow")
main_photo2_label.pack()


def bttn(x,y,text,bcolor,fcolor,cmd):

    def on_enter(e):
        click_button['background']=bcolor
        click_button['foreground']=fcolor

    def on_leave(e):
        click_button['background']=fcolor
        click_button['foreground']=bcolor

    click_button=tk.Button(main_frame,width=42,height=2,text=text,
                        fg=bcolor,
                        bg=fcolor,
                        activeforeground=fcolor,
                        activebackground=bcolor,
                        bd=3,   # fore button boarder
                        relief='raised',
                        overrelief='groove',
                        command=cmd)

    click_button.bind("<Enter>", on_enter)
    click_button.bind("<Leave>", on_leave)

    click_button.place(x=x,y=y)


root.mainloop()

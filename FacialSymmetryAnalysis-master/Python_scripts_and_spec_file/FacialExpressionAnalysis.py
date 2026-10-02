# ================= Importing all required libraries and modules ==========================
import PySimpleGUI as sg
import os.path
import numpy as np
import cv2
import pandas as pd 
import io
import matplotlib.pyplot as plt 
from keras.models import Sequential 
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D
from fer import FER
from fer import Video
from PIL import Image
import blobconverter
import depthai as dai
import re
import mediapipe as mp
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
import copy

# ================= 'Text' Class for writing text on the live stream =================
class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.color, 1, self.line_type)

# ============= The Camera libraries for connecting the camera. You can ignore this part and only check when necessary ========
# ===============================      Starts Here          ===========================
openvinoVersion = "2021.4"
p = dai.Pipeline()
p.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

# Set resolution of mono cameras
resolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

# THE_720_P => 720
resolution_num = int(re.findall("\d+", str(resolution))[0])

def populate_pipeline(p, name, resolution):
    cam = p.create(dai.node.MonoCamera)
    socket = dai.CameraBoardSocket.LEFT if name == "left" else dai.CameraBoardSocket.RIGHT
    cam.setBoardSocket(socket)
    cam.setResolution(resolution)

    # ImageManip for cropping (face detection NN requires input image of 300x300) and to change frame type
    face_manip = p.create(dai.node.ImageManip)
    face_manip.initialConfig.setResize(300, 300)
    # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
    face_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(face_manip.inputImage)

    # NN that detects faces in the image
    face_nn = p.create(dai.node.MobileNetDetectionNetwork)
    face_nn.setConfidenceThreshold(0.2)
    face_nn.setBlobPath(blobconverter.from_zoo("face-detection-retail-0004", shaves=6, version=openvinoVersion))
    face_manip.out.link(face_nn.input)

    # Send mono frames to the host via XLink
    cam_xout = p.create(dai.node.XLinkOut)
    cam_xout.setStreamName("mono_" + name)
    face_nn.passthrough.link(cam_xout.input)

    # Script node will take the output from the NN as an input, get the first bounding box
    # and send ImageManipConfig to the manip_crop
    image_manip_script = p.create(dai.node.Script)
    image_manip_script.inputs['nn_in'].setBlocking(False)
    image_manip_script.inputs['nn_in'].setQueueSize(1)
    face_nn.out.link(image_manip_script.inputs['nn_in'])
    image_manip_script.setScript("""

import time
def limit_roi(det):
    if det.xmin <= 0: det.xmin = 0.001
    if det.ymin <= 0: det.ymin = 0.001
    if det.xmax >= 1: det.xmax = 0.999
    if det.ymax >= 1: det.ymax = 0.999

while True:
    face_dets = node.io['nn_in'].get().detections
    # node.warn(f"Faces detected: {len(face_dets)}")
    for det in face_dets:
        limit_roi(det)
        # node.warn(f"Detection rect: {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
        cfg = ImageManipConfig()
        cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
        cfg.setResize(48, 48)
        cfg.setKeepAspectRatio(False)
        node.io['to_manip'].send(cfg)
        # node.warn(f"1 from nn_in: {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
    """)

    # This ImageManip will crop the mono frame based on the NN detections. Resulting image will be the cropped
    # face that was detected by the face-detection NN.
    manip_crop = p.create(dai.node.ImageManip)
    face_nn.passthrough.link(manip_crop.inputImage)
    image_manip_script.outputs['to_manip'].link(manip_crop.inputConfig)
    manip_crop.initialConfig.setResize(48, 48)
    manip_crop.inputConfig.setWaitForMessage(False)

    # Send ImageManipConfig to host so it can visualize the landmarks
    config_xout = p.create(dai.node.XLinkOut)
    config_xout.setStreamName("config_" + name)
    image_manip_script.outputs['to_manip'].link(config_xout.input)

    crop_xout = p.create(dai.node.XLinkOut)
    crop_xout.setStreamName("crop_" + name)
    manip_crop.out.link(crop_xout.input)

    # Second NN that detects landmarks from the cropped 48x48 face
    landmarks_nn = p.create(dai.node.NeuralNetwork)
    landmarks_nn.setBlobPath(blobconverter.from_zoo("landmarks-regression-retail-0009", shaves=6, version=openvinoVersion))
    manip_crop.out.link(landmarks_nn.input)

    landmarks_nn_xout = p.create(dai.node.XLinkOut)
    landmarks_nn_xout.setStreamName("landmarks_" + name)
    landmarks_nn.out.link(landmarks_nn_xout.input)

populate_pipeline(p, "right", resolution)
populate_pipeline(p, "left", resolution)

# =========================================         Ends Here    ===============================================

# ======================== Haar Cascade Library for Face and Eyes Detection ==========================
face_cascade=cv2.CascadeClassifier('D:\\Desktop\\Livrables\\my_creation\\fea\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('D:\\Desktop\\Livrables\\my_creation\\fea\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml')

# ============================== MediaPipe Library (The 468 facial landmarks) =================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh() 

# =========================== Emotion Prediction Library ========================================
emo_detector = FER(mtcnn=True)


# ======================== Assigning color variable and initialiazing the 'Text' Class ======================
leftColor = (255,0,0)
rightColor = (0,255,0)
textHelper = TextHelper()


# ======================= Getting screen parametres (screen window size) ========================
screen_width, screen_height = sg.Window.get_screen_size()

# =========== Initialization ==========
StopFlag = 0

# ============== For the ReadCsv Menu in the GUI =================
def draw_figure(window,filename):
    """
    Draws the previously created "figure" in the supplied Image Element
    :param element: an Image Element
    :param figure: a Matplotlib figure
    :return: The figure canvas
    """
    element = window['plot']
    figure = plt.figure()

    #Read the file using the filename given in the constructor
    csv_data = pd.read_csv(filename,sep=",")
    csv_data = csv_data.drop(csv_data.columns[0],axis=1)

    #Plot the data for all columns:
    fig, ax = plt.subplots()
    for col in csv_data.columns:
        ax.plot(csv_data.index, csv_data[col], label=col)
    ax.set_ylabel('Emotion Score')
    ax.set_xlabel('Frame')
    ax.set_title('Emotion Evolution over all Frames')
    ax.legend()
        
    # Save the plot as an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)

    #update PySimpleGUI image
    window['plot'].update(data=buf.getvalue())
    event,values = window.read()

# Variable to handle double-clicking the camera button
clicked_ = False

# Window configuration
X = 479
Y = 260
window_x = (screen_width // 2) - (X // 2)
window_y = (screen_height // 2) - (Y // 2)


# Display or mask circle and line used to help for face alignment
def toggle_visual_aid(window):
    global show_help
    show_help = not show_help

# ================= For the 'Help' button ==============
text_to_display = (
    "Use the Button to Navigate to and from the ''Main Menu\". "
    "For the Symmetry analysis, the 'Calculate Asymmetry' Button will appear "
    "automatically when you are in a good position and orientation with the camera. Please look "
    "straight at the camera to ensure this. When this button appears, click on "
    "it to see the asymmetry analysis results and the plots."
)

# Creation of the 4 layouts that the window will display
def Principal_layout():
    num_cameras = 0

    # Try opening cameras with increasing indices until no camera is available
    while True:
        cap = cv2.VideoCapture(num_cameras)
        if not cap.isOpened():
            break
        cap.release()
        num_cameras += 1
    # print("Number of available laptop cameras: ", num_cameras)
    
    # We set a maximum of 4 cameras
    if num_cameras > 4 :  
        num_cameras = 4
    
    # Dynamically add toggle button for each camera in Principal_layout  
    camera_buttons = [sg.Button(f"OAK-D Camera", key=f'_oak_d_')]
    for i in range(num_cameras):
        camera_buttons.append(sg.Button(f"Other Camera ({i+1})",key=f'_CAMERA_{i+1}_'))
    
    # Principal_layout structure
    layout = [
        [sg.Button("LIVE DETECTION",size=(59,2))],
        camera_buttons,
        [sg.Button("UPLOAD VIDEO",size=(59,2))],
        [sg.Button("READ CSV",size=(59,2))],
        [sg.Button("Exit",button_color='red',size=(59,2))],
        [sg.Text("Upgraded Version (3D) by Mubarak", font=("Helvetica", 9, "italic")), sg.Text("                Previous version by Enzo & Walid", font=("Helvetica", 9, "italic"))]
        ]
    return layout

# LiveDetection_layout structure
def LiveDetection_layout(): 
    layout = [
        [sg.Text("Live Detection & Facial Analysis in 3D", font=("Courier New", 13),  text_color= "White")],
        [sg.Image(filename='',key='image')],
        [sg.Button("HOME",visible=True,key="MAIN",button_color='green'),sg.Button("START"),sg.Button("STOP",key="STOP"),sg.Button("HELP", key="_TOGGLE_VISUAL_AID_"),sg.Button("Calc_SYM", key="_CALC_SYM_",visible=False)]
        ]
    return layout

# UploadVideo_layout structure
def UploadVideo_layout():
    layout = [
        [sg.Text("Select video folder")],
        [sg.InputText(key='_FILEVIDEO_',disabled=True), sg.FileBrowse(file_types=(("Video Files", "*.mp4;*.avi"),))],
        [sg.Button("HOME",visible=True,key='_MAIN_',button_color='green'),sg.Button("PLAY",key='_PLAY_'),sg.Button("STOP",key='_STOP_')],
        [sg.Image(filename='',key='_IMAGE_',size=(480, 640))]
        ]
    return layout

# ReadCSV_layout structure
def ReadCSV_layout():
    layout = [
        [sg.Text("Select .csv file to plot:"),sg.Input(key='-FILECSV-', visible=False), sg.FileBrowse(file_types=(("CSV Files","*.csv"),))],
        [sg.Image(filename='',key='plot')],
        [sg.Button("HOME",key="__MAIN__",button_color='green'),sg.Button("Display Data",visible=True)]
    ]
    return layout

# Call the principal Layout and display it in the middle of your screen
window = sg.Window("Facial Expression Analysis", Principal_layout(), size=(X,Y),location=(window_x,window_y))

# Initializing recording and working flag (these two are used in handling the 'Play and Stop' of live detection in the loop)
recording = False
working = False


# ======================================= The Main event loop: ===========================================
while True :

    # Check values and events (main event loop, continous loop)      
    event, values = window.read(timeout = 20)
    
    # Display the image of the webcam and analyze it if recording is true
    if (recording and not working): 
        working = True   # Working is later set to 'False', check below. 

        frame = queues[0].get().getCvFrame() # 1 for the Right camera of OAK-D and Zero for the left.
        # Remember that the center camera has isssues with streaming an Image, so we use left or right.  

        # To avoid displaying the rotated image but only the rotated image will be used for the symmetry analysis.
        display_img = copy.deepcopy(frame)

        # Now I need to display the normal image (un-rotated) with the landmarks. To avoid displaying the rotated image.
        result_ = face_mesh.process(display_img)

        if result_.multi_face_landmarks:
            for facial_landmarks in result_.multi_face_landmarks:            
                for i in range(0, 468):
                    pt_ = facial_landmarks.landmark[i]

                    norm_x_ = int(pt_.x * 300)
                    norm_y_ = int(pt_.y * 300)

                    cv2.circle(display_img, (norm_x_, norm_y_), 1, (255, 0, 0), -1)
        
        # For detecting faces and drawing rectangular box. I am using Haar Cascade Again == 
        gray_for_align = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Creating faces variables
        faces= face_cascade.detectMultiScale (gray_for_align, 1.1, 4)

        # Draw rectangles around detected faces
        for(x , y,  w,  h) in faces:
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw rectangle
        
        # Get the dominant emotion for the image frame (the display_img does not work well here, so I used the 'frame' itself)
        dominant_emotion, emotion_score = emo_detector.top_emotion(frame)

        # Displaying the Dominant Emotion on the Image
        y = 0
        y_delta = 18
        strings = [
            f"Dominant Emotion: {dominant_emotion}",
            "Score:{}".format(emotion_score),
        ]
        for s in strings:
            y += y_delta
            textHelper.putText(display_img, s, (10, y))

        imgbytes = cv2.imencode('.png', display_img)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)
            
        # ========> (1) Aligning the Image to upright position first (Using the center of the two eyes) ======>
        
        # Converting the image into grayscale
        # ==== I am commenting the next two lines because I have it above for face detection =====
        #  
        # gray_for_align = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # faces= face_cascade.detectMultiScale (gray_for_align, 1.1, 4)

        for(x , y,  w,  h) in faces:

            # Creating regions of interest
            roi_gray=gray_for_align[y:(y+h), x:(x+w)]
            roi_color=frame[y:(y+h), x:(x+w)]

            # Creating variable eyes
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            index=0

            if len(eyes) != 2:
                window['_CALC_SYM_'].update(visible=False) # The appaearance and dissapearance on GUI!!! 
                # === The algorithm here sometimes detects face in non-face parts of the body but when this happens, eyes detection
                # will never occur, so the reason for this condition here and since the eyes are our main focus for this alignment.====
                break
            else:

                # Creating for loop in order to separate the eyes
                for (ex , ey,  ew,  eh) in eyes:
                    if index == 0:
                        eye_1 = (ex, ey, ew, eh)
                        # print(eye_1)
                    elif index == 1:
                        eye_2 = (ex, ey, ew, eh)

                    # cv2.rectangle(roi_color, (ex,ey) ,(ex+ew, ey+eh), (0,0,255), 3)
                    index = index + 1

                # Indicating the Left and the right eye
                if eye_1[0] < eye_2[0]:
                    left_eye = eye_1
                    right_eye = eye_2
                else:
                    left_eye = eye_2
                    right_eye = eye_1

                # Calculating coordinates of a central points of the rectangles (a point in the eye)
                left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
                left_eye_x = left_eye_center[0] 
                left_eye_y = left_eye_center[1]
                    
                right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
                right_eye_x = right_eye_center[0]
                right_eye_y = right_eye_center[1]


                # Indicating the direction of rotation and third point to make a right angle triangle 
                if left_eye_y > right_eye_y:
                    A = (right_eye_x, left_eye_y)
                    # Integer -1 indicates that the image will rotate in the clockwise direction
                    direction = -1 
                else:
                    A = (left_eye_x, right_eye_y)
                    # Integer 1 indicates that image will rotate in the counter clockwise  
                    # direction
                    direction = 1

                # Calculating the sides of the triangle and angle
                delta_x = right_eye_x - left_eye_x
                delta_y = right_eye_y - left_eye_y
                # print(delta_y)
                
                if delta_x != 0:
                    window['_CALC_SYM_'].update(visible=True)
                    angle=np.arctan(delta_y/delta_x)
                    angle = (angle * 180) / np.pi #+ve angles denotes CCW rotation and vice versa.

                    # Width and height of the image
                    h, w = frame.shape[:2]
                    # Calculating a center point of the image
                    # Integer division "//"" ensures that we receive whole numbers
                    center = (w // 2, h // 2)
                    # Defining a matrix M and calling (cv2.getRotationMatrix2D method)
                    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
                    # Applying the rotation to our image using the
                    # cv2.warpAffine method
                    rotated_frame = cv2.warpAffine(frame, M, (w, h))

                    #  If the 'Calculate Symmetry Button is Clicked, then execute the code below'
                    if event == '_CALC_SYM_':

                        # <========= Use the rotated frame for symmetry processing ========>
                        # left_for_landmark = rotated_frame; 
                        result = face_mesh.process(rotated_frame)

                        #  ===== (2) Getting the 3D points_cloud. This should go under the else condition, that is
                        # ====== in the case there is a face alignment.

                        if result.multi_face_landmarks: # If you find a face 
                    
                            for facial_landmarks in result.multi_face_landmarks:
                                            
                                landmarks_xyz = []
                                        
                                for i in range(0, 468):

                                    pt = facial_landmarks.landmark[i]
                                    x = pt.x
                                    y = pt.y
                                    z = pt.z 

                                    norm_x = int(pt.x * 300)
                                    norm_y = int(pt.y * 300)

                                    # cv2.circle(left_for_landmark, (norm_x, norm_y), 1, (255, 0, 0), -1)

                                    numb = (x,y,z)

                                    landmarks_xyz.append(numb)

                                
                            # <==================== Relection begins here ===================>
                            spatials = landmarks_xyz
                            # print(spatials)

                            # ======= Calculate the Reflection after getting the 3D values =========
                            id_mat = np.eye(3)
                                            
                            # === The v-MATRIX ====
                            v_mat = np.array([[1], 
                                                [0], 
                                                [0]])
                                    
                            # === The computation ====
                            numerator = np.matmul(v_mat, v_mat.transpose())
                            denominator = np.matmul(v_mat.transpose(), v_mat)

                            division = numerator / denominator
                            multi = 2 * division

                            h_mat = np.subtract(id_mat, multi) # === 3*3 Matrix ==== 

                            # === Now using the 3D coordinate values ===
                            points_list = np.array(spatials)  # 468*3

                            reflec = np.matmul(h_mat, points_list.transpose()) # === 3*3 x 3*468 = 3*468

                            # ===== Transposing back  the result for easy indexing =====
                            reflec = reflec.transpose()  # 468*3

                            # ======= The alignment begins here. First, the translation ========
                            translation = np.mean(points_list, axis=0) - np.mean(reflec, axis=0)  # mean of 468*3 - 468*3 = (1*3), Broadcasting here too
                            aligned_points = reflec + translation # This does broadcasting (element-wise addition)
                            reflec = aligned_points   # 468 * 3  

                            # ===========> Then the Symmetry Analysis Code Should follow because the 'calculate sym' button has been clicked ==========>
                            # I also want to see the last rotated image that is being used for asymmetry processing.
                        
                            cv2.imshow("Last Image Frame", rotated_frame)
                                
                            # ======> The ICP should come before the plotting. The task here is to align the reflected points with the original ======>
                            # Function for ICP alignment
                            def best_fit_transform(A, B):
                                '''
                                Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
                                Input:
                                A: Nxm numpy array of corresponding points
                                B: Nxm numpy array of corresponding points
                                Returns:
                                T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
                                R: mxm rotation matrix
                                t: mx1 translation vector
                                '''

                                assert A.shape == B.shape

                                # get number of dimensions
                                m = A.shape[1]

                                # translate points to their centroids
                                centroid_A = np.mean(A, axis=0)
                                centroid_B = np.mean(B, axis=0)
                                AA = A - centroid_A
                                BB = B - centroid_B

                                # rotation matrix
                                H = np.dot(AA.T, BB)
                                U, S, Vt = np.linalg.svd(H)
                                R = np.dot(Vt.T, U.T)

                                # special reflection case
                                if np.linalg.det(R) < 0:
                                    Vt[m-1,:] *= -1
                                    R = np.dot(Vt.T, U.T)

                                # translation
                                t = centroid_B.T - np.dot(R,centroid_A.T)

                                # homogeneous transformation
                                T = np.identity(m+1)
                                T[:m, :m] = R
                                T[:m, m] = t

                                return T, R, t


                            def nearest_neighbor(src, dst):
                                '''
                                Find the nearest (Euclidean) neighbor in dst for each point in src
                                Input:
                                    src: Nxm array of points
                                    dst: Nxm array of points
                                Output:
                                    distances: Euclidean distances of the nearest neighbor
                                    indices: dst indices of the nearest neighbor
                                '''

                                assert src.shape == dst.shape

                                neigh = NearestNeighbors(n_neighbors=1)
                                neigh.fit(dst)
                                distances, indices = neigh.kneighbors(src, return_distance=True)
                                return distances.ravel(), indices.ravel()

                            # =========== The ICP function =============
                            def icp(A, B, init_pose=None, max_iterations=20, tolerance=1e-3):
                                '''
                                The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
                                Input:
                                    A: Nxm numpy array of source mD points
                                    B: Nxm numpy array of destination mD point
                                    init_pose: (m+1)x(m+1) homogeneous transformation
                                    max_iterations: exit algorithm after max_iterations
                                    tolerance: convergence criteria
                                Output:
                                    T: final homogeneous transformation that maps A on to B
                                    distances: Euclidean distances (errors) of the nearest neighbor
                                    i: number of iterations to converge
                                '''

                                assert A.shape == B.shape

                                # get number of dimensions
                                m = A.shape[1]

                                # make points homogeneous, copy them to maintain the originals
                                src = np.ones((m+1,A.shape[0]))  # arrays of ones in dimension 4*468 (x,y,z,1) for each point
                                dst = np.ones((m+1,B.shape[0]))  # arrays of ones in dimension 4*468
                                src[:m,:] = np.copy(A.T)         # Extracting the first 3 rows (x,y,z) and the columns and copying the original values into it
                                dst[:m,:] = np.copy(B.T)

                                # apply the initial pose estimation
                                if init_pose is not None:
                                    src = np.dot(init_pose, src)
                                    # dst = np.dot(init_pose, dst)

                                prev_error = 0

                                for i in range(max_iterations):
                                    # find the nearest neighbors between the current source and destination points
                                    distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
                                    # ==> It is worth noting here that the array orderliness index of the destination doesn't change ==>

                                    # compute the transformation between the current source and nearest destination points
                                    T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)
                                    initial_T = T #This is the transformation between the last transformed src points and its nearest dst points. Here, the indices is used. 
                                    # This transformation is then applied to the src points to continue with the alignment.  

                                    # update the current source
                                    src = np.dot(T, src)
                                    # dst = np.dot(T, dst)      # ===== Here is this dude, the aligned points from the transformation above =====

                                    # check error
                                    mean_error = np.mean(distances)
                                    if np.abs(prev_error - mean_error) < tolerance:
                                        break
                                    prev_error = mean_error

                                # calculate final transformation
                                T,_,_ = best_fit_transform(A, src[:m,:].T)
                                # ===> This is the transformation betwen the un-aligned src points and after it is being aligned. That is, the transformation
                                # that will move the src points from their initial pos and orientation to the final pose and orien made with the alignment. 

                                # return src, initial_T, T, distances, i
                                # return src, indices, i    ===> The indices have been established with the symmetrical face.
                                return src, i

                            # =========== Now calling the ICP ==========
                            reflec, i = icp(reflec, points_list)
                            reflec = reflec.T       # Transpose back to 468 * 4
                            reflec = reflec[:, :3]  # Extract the x, y and z columns (468 *3 )

                            # The indices array for the correspondence (Ground Truth)
                            indi = [0,   1,   2,   248, 4,   5,   6,   249, 8,   9,   10,  11,  12,  13,  14,  15,  16,  17, 
                                    18,  19,  250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265,
                                    266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283,
                                    284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301,
                                    302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319,
                                    320, 321, 322, 323,  94, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336,
                                    337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354,
                                    355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372,
                                    373, 374, 375, 376, 377, 378, 379, 151, 152, 380, 381, 382, 383, 384, 385, 386, 387, 388,
                                    389, 390, 164, 391, 392, 393, 168, 394, 395, 396, 397, 398, 399, 175, 400, 401, 402, 403,
                                    404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 195, 419, 197,
                                    420, 199, 200, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435,
                                    436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453,
                                    454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467,   3,   7,  20,  21,
                                    22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                                    40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,
                                    58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
                                    76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,
                                    95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                                    113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
                                    131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148,
                                    149, 150, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 169, 170,
                                    171, 172, 173, 174, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
                                    190, 191, 192, 193, 194, 196, 198, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211,
                                    212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229,
                                    230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247]
                            # ===> When calling the ICP here, the reflec is used as src because it is being transformed by ICP while the points_list (original) remain the same. 


                            # ====> Ignore the below blocks of code from where I indicate 'starts' to where I indicate 'end' because I used them to establish the ground truth correspondence.
                            # After establsihing, I then compare the result with that of ICP correspondence for a symmetrical face. 'Starts Here'

                            # <============= Testing that the orderliness is not changing and it's symmetrical to establish the ground truth ==============>
                            # ===============================> The figure for testing, comment after testing ============================>

                            # fig = plt.figure(figsize=(10, 8))
                            # ax = fig.add_subplot(111, projection='3d')

                            # Plot the original cloud points in green
                            # ax.scatter(points_list[:, 0], points_list[:, 1], points_list[:, 2], c='g', s = 5, alpha=0.5, label='original points')

                            # Plot the reflected cloud points in red
                            # ax.scatter(reflec[:, 0], reflec[:, 1], reflec[:, 2], c='g',  s = 5, alpha=0.5, label='reflected points') #return color to red

                            # Customize the plot
                            # ax.set_xlabel('X-axis')
                            # ax.set_ylabel('Y-axis')
                            # ax.set_zlabel('Z-axis')

                            # ax.set_title('The Original Face (Before Reflection)')
                            # ax.set_title('The Reflected Face')

                            # Set the aspect ratio to 'auto' to prevent distortion
                            # ax.set_aspect('auto')

                            # <================ The Landmarks on the Symmetry Plane  ==================>
                            # sym_plane_lndmk = [1,2,3,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,95,152,153,165,169,176,196,198,200,201]

                            # <================ The Landmarks on the Left Face  ==================>
                            # left_face_lndmk = [4, 8, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 
                            #                     47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 
                            #                     74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 
                            #                     102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 
                            #                     124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 
                            #                     147, 148, 149, 150, 151, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 166, 167, 168, 170, 171, 172, 173, 
                            #                     174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 197, 199, 
                            #                     202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 
                            #                     225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248]

                            # # <================ The Landmarks on the Right Face  ==================>
                            # right_face_lndmk = [249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 
                            #                     273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 
                            #                     297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 
                            #                     321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 
                            #                     345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 
                            #                     369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 
                            #                     393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 
                            #                     417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 
                            #                     441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 
                            #                     465, 466, 467, 468]

                            # print(len(left_face_lndmk))

                            # for j in left_face_lndmk:

                                # face_x = points_list[j-1,0]
                                # face_y = points_list[j-1,1]
                                # face_z = points_list[j-1,2] 

                                # face_x = reflec[j-1,0]
                                # face_y = reflec[j-1,1]
                                # face_z = reflec[j-1,2] 
                                
                                # Write the landmark number for each landmark on the plot
                                # ax.text(face_x, face_y, face_z, f"{j}", fontsize=4, fontweight='extra bold', color = 'blue')

                            # for j in right_face_lndmk:

                                # face_x = points_list[j-1,0]
                                # face_y = points_list[j-1,1]
                                # face_z = points_list[j-1,2] 

                                # face_x = reflec[j-1,0]
                                # face_y = reflec[j-1,1]
                                # face_z = reflec[j-1,2]
                                
                                # Write the landmark number for each landmark on the plot
                                # ax.text(face_x, face_y, face_z, f"{j}", fontsize=4, fontweight='extra bold', color = 'red')


                            # Remove the grid
                            # ax.grid(False)

                            # Hide the x-axis, y-axis, and z-axis
                            # ax.xaxis.set_visible(False)
                            # ax.yaxis.set_visible(False)
                            # ax.zaxis.set_visible(False)

                            # Hide the entire plot
                            # ax.axis('off')
                            # ======================================> 'Ends Here'. The End Ground Truth  =====================================> 

                            # ========> Next, Calculating the Error for Each Correspondence and showing the Vectors ========>
                            # ==================> Now the Code for plotting: This will Follow the ICP alignment ====================>

                            # (1) Create a 3D figure and axes: 4 different Figures
                            # ===============================> The First Figure (The Initial one) ============================>
                            fig = plt.figure(figsize=(10, 8))
                            ax = fig.add_subplot(111, projection='3d')

                            # Plot the original cloud points in green
                            ax.scatter(points_list[:, 0], points_list[:, 1], points_list[:, 2], c='g', s = 5, alpha=0.5, label='original points')

                            # Plot the reflected cloud points in red
                            ax.scatter(reflec[:, 0], reflec[:, 1], reflec[:, 2], c='r',  s = 5, alpha=0.5, label='reflected points')

                            # Customize the plot
                            ax.set_xlabel('X-axis label')
                            ax.set_ylabel('Y-axis label')
                            ax.set_zlabel('Z-axis label')
                            ax.set_title('Alignment with the mean error and standard deviation')

                            # Set the aspect ratio to 'auto' to prevent distortion
                            ax.set_aspect('auto')

                            # Remove the grid
                            ax.grid(False)

                            # Hide the x-axis, y-axis, and z-axis
                            ax.xaxis.set_visible(False)
                            ax.yaxis.set_visible(False)
                            ax.zaxis.set_visible(False)


                            # ===================> The Second Figure ===============================>
                            fig2 = plt.figure(figsize=(10, 8))
                            ax2 = fig2.add_subplot(111, projection='3d')

                            # Plot the original cloud points in green
                            ax2.scatter(points_list[:, 0], points_list[:, 1], points_list[:, 2], c='g', s = 5, alpha=0.5, label='original points')

                            # Plot the reflected cloud points in red
                            ax2.scatter(reflec[:, 0], reflec[:, 1], reflec[:, 2], c='r',  s = 5, alpha=0.5, label='reflected points')

                            # Customize the plot
                            ax2.set_xlabel('X-axis label')
                            ax2.set_ylabel('Y-axis label')
                            ax2.set_zlabel('Z-axis label')
                            ax2.set_title('The Asymmetry along the X-axis')

                            # Set the aspect ratio to 'auto' to prevent distortion
                            ax2.set_aspect('auto')

                            # Remove the grid
                            ax2.grid(False)

                            # Hide the x-axis, y-axis, and z-axis
                            ax2.xaxis.set_visible(False)
                            ax2.yaxis.set_visible(False)
                            ax2.zaxis.set_visible(False)


                            # ===================> The Third Figure ===============================>
                            fig3 = plt.figure(figsize=(10, 8))
                            ax3 = fig3.add_subplot(111, projection='3d')

                            # Plot the original cloud points in green
                            ax3.scatter(points_list[:, 0], points_list[:, 1], points_list[:, 2], c='g', s = 5, alpha=0.5, label='original points')

                            # Plot the reflected cloud points in red
                            ax3.scatter(reflec[:, 0], reflec[:, 1], reflec[:, 2], c='r', s = 5, alpha=0.5, label='reflected points')

                            # Customize the plot
                            ax3.set_xlabel('X-axis label')
                            ax3.set_ylabel('Y-axis label')
                            ax3.set_zlabel('Z-axis label')
                            ax3.set_title('The Asymmetry along the Y-axis')

                            # Set the aspect ratio to 'auto' to prevent distortion
                            ax3.set_aspect('auto')

                            # Remove the grid
                            ax3.grid(False)

                            # Hide the x-axis, y-axis, and z-axis
                            ax3.xaxis.set_visible(False)
                            ax3.yaxis.set_visible(False)
                            ax3.zaxis.set_visible(False)


                            # ===================> The Fourth Figure ===============================>
                            fig4 = plt.figure(figsize=(10, 8))
                            ax4 = fig4.add_subplot(111, projection='3d')

                            # Plot the original cloud points in green
                            ax4.scatter(points_list[:, 0], points_list[:, 1], points_list[:, 2], c='g', s = 5, alpha=0.5, label='original points')

                            # Plot the reflected cloud points in red
                            ax4.scatter(reflec[:, 0], reflec[:, 1], reflec[:, 2], c='r',  s = 5, alpha=0.5, label='reflected points')

                            # Customize the plot
                            ax4.set_xlabel('X-axis label')
                            ax4.set_ylabel('Y-axis label')
                            ax4.set_zlabel('Z-axis label')
                            ax4.set_title('The Asymmetry along the Z-axis')

                            # Set the aspect ratio to 'auto' to prevent distortion
                            ax4.set_aspect('auto')

                            # Remove the grid
                            ax4.grid(False)

                            # Hide the x-axis, y-axis, and z-axis
                            ax4.xaxis.set_visible(False)
                            ax4.yaxis.set_visible(False)
                            ax4.zaxis.set_visible(False)

                            # ==========> After showing those 4 figures with the alignment, then we begin plotting the vectors and errors =======>
                            x_values = []
                            y_values = []
                            z_values = []
                            distances = []

                            for j in range(468):

                                corrspnd_indi = indi[j]      # The indices of the correspondent points (established with the ground truth)
                                corrspnd_pnts = points_list[corrspnd_indi, :] # Extract the row indices and all columns of this row (x y z)
                                reflec_pnts = reflec[j]       # This doesn't need indices and it is in ascending order as from the landmarks (only the values changes and not the order)

                                face_x = corrspnd_pnts[0]
                                face_y = corrspnd_pnts[1]
                                face_z = corrspnd_pnts[2]

                                reflec_x = reflec_pnts[0]
                                reflec_y = reflec_pnts[1]
                                reflec_z = reflec_pnts[2]

                                # Do the Subtraction
                                x_val = face_x - reflec_x
                                y_val = face_y - reflec_y
                                z_val = face_z - reflec_z

                                # Estimate the Error (the distance)
                                dist = np.sqrt((face_x-reflec_x)**2 + (face_y-reflec_y)**2 + (face_z-reflec_z)**2)

                                # Append the results into a list
                                x_values.append(x_val)
                                y_values.append(y_val)
                                z_values.append(z_val)

                                distances.append(dist)

                                # Drawing a line to connect the original landmarks and their reflected correspondence (Do this for the 4 Plots)
                                ax.plot([face_x, reflec_x], [face_y, reflec_y], [face_z, reflec_z], linestyle='-', color='black')
                                ax2.plot([face_x, reflec_x], [face_y, reflec_y], [face_z, reflec_z], linestyle='-', color='black')
                                ax3.plot([face_x, reflec_x], [face_y, reflec_y], [face_z, reflec_z], linestyle='-', color='black')
                                ax4.plot([face_x, reflec_x], [face_y, reflec_y], [face_z, reflec_z], linestyle='-', color='black')

                                # Write the distance (error) value on the connecting line for each landmark and its correspondence (On Fig. 1)
                                # ax.text((face_x + reflec_x) / 2, (face_y + reflec_y) / 2, (face_z + reflec_z) / 2, f"{distances[j]:.2f}", fontsize=6, fontweight='bold')


                                # Show the vector length as distance/error and the direction (determined by Python) on the first plot (Figure 1)
                                ax.quiver(face_x, face_y, face_z, 2*x_val, 2*y_val, 2*z_val, color='blue', linewidth=1, arrow_length_ratio=1.0)

                                # Drawing the vector differences (along each axes) on the face (original) vectors
                                ax2.quiver(face_x, face_y, face_z, 2*x_val, 0, 0, color='blue', linewidth=1, arrow_length_ratio=1.0)
                                ax3.quiver(face_x, face_y, face_z, 0, 2*y_val, 0, color='blue', linewidth=1, arrow_length_ratio=1.0)
                                ax4.quiver(face_x, face_y, face_z, 0, 0, 2*z_val, color='blue', linewidth=1, arrow_length_ratio=1.0)

                                # ======>  I scaled the vector twice equally to enhance the visualization =====>


                            # Calculate the mean error and standard deviation
                            mean_err = np.mean(distances)

                            std_dv = np.std(distances)

                            # print(mean_err, std_dv)

                            # <=========  Write the mean error and the standard deviation ======>
                            ax.text(0.5, 0.2, 0, f"Mean error =  {mean_err:.6f}", fontsize=10, fontweight='extra bold', color = 'purple')
                            ax.text(0.5, 0.25, 0, f"Error Standard Deviation =  {mean_err:.6f}", fontsize=10, fontweight='extra bold', color = 'purple')

                            # Enable interactive mode
                            plt.ion()

                            # Show the plot
                            plt.show()

                            # Enable pan and zoom
                            # pan_zoom(fig)

                            # Keep the plot open until you close it
                            plt.ioff()
                            plt.show()      

            # # Mean Error and Standard Deviation
            # AVG_distances = mean_err
            # STD_Distances = std_dv

                else:  # Break if there is a zero division in obtaining the angle (The chance is very slim because the object is already in good pos and orientation with the camera)
                    # window['_CALC_SYM_'].update(visible=False)
                    break
        
        # Set 'working' back to 'False' at the end of each image frame
        working=False                    

    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    # ====== Select the camera to use =======
    if event == '_oak_d_':

        window["_oak_d_"].update(button_color=("white", "blue"))

        if clicked_ == False:
            
            # ========> This only needs to be initialized ones and at the first click ===========>
            device = dai.Device(p.getOpenVINOVersion())
            cams = device.getConnectedCameras()
            depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
            if not depth_enabled:
                raise RuntimeError("Unable to run this experiment on device without depth capabilities! (Available cameras: {})".format(cams))
            device.startPipeline(p)
            
            # Start pipeline
            queues = []
            for name in ["left", "right"]:
                queues.append(device.getOutputQueue(name="mono_"+name, maxSize=4, blocking=False))
                queues.append(device.getOutputQueue(name="crop_"+name, maxSize=4, blocking=False))
                queues.append(device.getOutputQueue(name="landmarks_"+name, maxSize=4, blocking=False))
                queues.append(device.getOutputQueue(name="config_"+name, maxSize=4, blocking=False)) 

        else:

            # Start pipeline
            queues = []
            for name in ["left", "right"]:
                queues.append(device.getOutputQueue(name="mono_"+name, maxSize=4, blocking=False))
                queues.append(device.getOutputQueue(name="crop_"+name, maxSize=4, blocking=False))
                queues.append(device.getOutputQueue(name="landmarks_"+name, maxSize=4, blocking=False))
                queues.append(device.getOutputQueue(name="config_"+name, maxSize=4, blocking=False))
            
        clicked_ = True      # Set click to true

    if event == '_CAMERA_1_' :
        #Read image of the first cam
        cap = cv2.VideoCapture(0)

    if event == '_CAMERA_2_' :
        #Read image of the second cam
        cap = cv2.VideoCapture(1)

    if event == '_CAMERA_3_' :
        #Read image of the third cam
        cap = cv2.VideoCapture(2)

    if event == '_CAMERA_4_' :
        #Read image of the forth cam
        cap = cv2.VideoCapture(3)

    ### ### ### live detection events ### ### ###

    elif event == 'LIVE DETECTION':
        window.close()
        # Giving new size of the window and place it in the middle of the screen
        X = 600 #Prev - 800
        Y = 400 #Prev - 700
        window_x = (screen_width // 2) - (X // 2)
        window_y = (screen_height // 2) - (Y // 2)
        window = sg.Window("Live Detection", LiveDetection_layout(), size=(X,Y),location=(window_x,window_y))
        
    elif event == 'START':
        recording = True
        # print('Live Capture Begins')
        window['MAIN'].update(visible=True)

    elif event == 'STOP' or StopFlag == 1:
        recording = False
        # print(recording)
        img = np.full((300, 300), 255)  # Prev - 480, 640
        # this is faster, shorter and needs less includes
        imgbytes = cv2.imencode('.png', img)[1].tobytes()
        window['image'].update(data=imgbytes)
        window['MAIN'].update(visible=True)

        if StopFlag == 1:
            StopFlag = 0
            window.close()
            # Giving new size of the window and place it in the middle of the screen
            X = 479  #prev - 479  #This is in the case that the 'Home Button' is pressed during 'live Capture'
            Y = 260  #prev - 260
            window_x = (screen_width // 2) - (X // 2)
            window_y = (screen_height // 2) - (Y // 2)
            window = sg.Window("Facial Expression Analysis", Principal_layout(), size=(X,Y),location=(window_x,window_y))

    # Back to main button of LiveDetection_layout
    elif event == 'MAIN': 
        if recording == True:
            StopFlag = 1
        
        if StopFlag == 0:

            window.close()
            # Giving new size of the window and place it in the middle of the screen
            X = 479
            Y = 260 
            window_x = (screen_width // 2) - (X // 2)
            window_y = (screen_height // 2) - (Y // 2)
            window = sg.Window("Facial Expression Analysis", Principal_layout(), size=(X,Y),location=(window_x,window_y))

    elif event == '_TOGGLE_VISUAL_AID_':
        sg.popup(text_to_display, title="Usage Info.", font=("Tahoma", 12), text_color='Black')

    ### ### ### Upload Video events ### ### ###
    elif event == 'UPLOAD VIDEO':
        window.close()
        window = sg.Window("Upload Video", UploadVideo_layout(), size=(X,Y),location=(window_x,window_y))

    elif event == '_PLAY_':
        #Get video file
        location_videofile = values['_FILEVIDEO_']

        # Input the video for processing
        input_video = Video(location_videofile)

        # The Analyze() function will run analysis on every frame of the input video. 
        # It will create a rectangular box around every image and show the emotion values next to that.
        # Finally, the method will publish a new video that will have a box around the face of the human with live emotion values.
        vid_df = input_video.analyze(emo_detector,annotate_frames=True,display=False, save_video=False, output="pandas")

        # We will now convert the analysed information into a panda dataframe.
        # This will help us import the data as a .CSV file to perform analysis over it later
        vid_df = input_video.get_first_face(vid_df)
        vid_df = input_video.get_emotions(vid_df)

        # Extract the name of the input video file
        video_filename = os.path.basename(location_videofile)

        # Create a new filename for the output CSV file
        csv_filename = os.path.splitext(video_filename)[0] + "_results.csv"
        # save the new file in .csv format
        vid_df.to_csv(csv_filename)

    # back to main butoon of UploadVideo_Layout
    elif event == '_MAIN_': 
        window.close()
        # Giving new size of the window and place it in the middle of the screen
        X = 479
        Y = 260
        window_x = (screen_width // 2) - (X // 2)
        window_y = (screen_height // 2) - (Y // 2)
        window = sg.Window("Facial Expression Analysis", Principal_layout(), size=(X,Y),location=(window_x,window_y))
    

    ### ### ### Read CSV events ### ### ###
    elif event == 'READ CSV':
        window.close()
        X = 650
        Y = 600
        window = sg.Window("Live Detection", ReadCSV_layout(), size=(X,Y),location=(window_x,window_y-150))

    elif event == "Display Data":
        filename = values['-FILECSV-']
        print(filename)
        draw_figure(window, filename)
    
    elif event == '__MAIN__':
        window.close()
        # Giving new size of the window and place it in the middle of the screen
        X = 479
        Y = 260
        window_x = (screen_width // 2) - (X // 2)
        window_y = (screen_height // 2) - (Y // 2)
        window = sg.Window("Facial Expression Analysis", Principal_layout(), size=(X,Y),location=(window_x,window_y))

# cap.release() # In the case that the Webcam is used. 

window.close()
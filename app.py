from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sqlite3 import DatabaseError
from flask import Flask, render_template, Response, request, redirect, url_for, flash, make_response, jsonify
import cv2
import os
import numpy as np
from threading import Thread
import Capture_Image
import csv
import Train_Image
import Recognize
import pandas as pd
from time import time
import time
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit, send
from socket import socket
from distutils.log import debug
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
from multiprocessing.pool import ThreadPool
import sys
from asyncio.windows_events import NULL
from wsgiref.validate import validator
from pymongo import MongoClient
import json
from bson import ObjectId
from bson.json_util import dumps
import datetime
from datetime import datetime
import pytz
import dlib
import speech_recognition as sr
import pyaudio
import wave
import threading
import nltk
from multiprocessing import Process
import glob


app = Flask(__name__, template_folder='./templates')
app.config['SECRET_KEY'] = 'mysecret'
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, resources={f"/api/*": {"origins": "*"}})
app.config['CORS HEADERS'] = "Content-Type"
client = MongoClient(
    "mongodb://localhost:27017")
db = client.get_database('AutoProctor')
collection = db.user


def train_image():
    Train_Image.TrainImages()


pool = ThreadPool(processes=1)
count = 0
mouth_opened = 0

#Timer for saving files for No face detection code
def check_time_difference():
    # set timezone to Indian Standard Time (IST)
    ist = pytz.timezone('Asia/Kolkata')

    # datetime object containing current date and time in IST
    now = datetime.now(ist)

    print("now =", now)

    # DD-MM-YYYY H-M-S in IST
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    print("date and time in IST =", dt_string)

    # specify the directory and partial filename to search
    directory = "C:\\Users\\Siddhesh Gadkar\\Desktop\\Sem8 workings\\Flask-Auto-Proctor\\warnings"
    partial_filename = "warn"

    # create a glob pattern to match the partial filename
    glob_pattern = os.path.join(directory, f"*{partial_filename}*")

    # find all files that match the glob pattern
    matching_files = glob.glob(glob_pattern)

    # get the most recent file creation time
    if matching_files:
        # get the most recent file in the list
        most_recent_file = max(matching_files, key=os.path.getctime)
        most_recent_time = os.path.getctime(most_recent_file)
        print(
            f"The most recent file matching '{partial_filename}' was created on {most_recent_time}")

        # convert the timestamp to a datetime object and format it as a string
        dt = datetime.fromtimestamp(most_recent_time)
        formatted_date = dt.strftime('%d-%m-%Y-%H-%M-%S')

        # get the current date and time as a string in the desired format
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")

        # calculate the difference between the two dates in seconds
        date1 = datetime.strptime(formatted_date, '%d-%m-%Y-%H-%M-%S')
        date2 = datetime.strptime(dt_string, '%d-%m-%Y-%H-%M-%S')
        difference = (date2 - date1).total_seconds()

        # print the results
        if(difference > 10):
            return True
        else:
            return False
    else:
        return True


@app.route('/audio')
def audio():
    def read_audio(stream, filename):
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 2
        fs = 44100  # Record at 44100 samples per second
        seconds = 10  # Number of seconds to record at once
        filename = filename
        frames = []  # Initialize array to store frames

        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()
        # Stop and close the stream
        stream.stop_stream()
        stream.close()

    def convert(i):
        if i >= 0:
            sound = 'record' + str(i) + '.wav'
            r = sr.Recognizer()

            with sr.AudioFile(sound) as source:
                r.adjust_for_ambient_noise(source)
                print("Converting Audio To Text and saving to file..... ")
                audio = r.listen(source)
            try:
                # API call to google for speech recognition
                value = r.recognize_google(audio)
                os.remove(sound)
                if str is bytes:
                    result = u"{}".format(value).encode("utf-8")
                else:
                    result = "{}".format(value)

                with open("test.txt", "a") as f:
                    f.write(result)
                    f.write(" ")
                    f.close()

            except sr.UnknownValueError:
                print("")
            except sr.RequestError as e:
                print("{0}".format(e))
            except KeyboardInterrupt:
                pass

    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100

    def save_audios(i):
        stream = p.open(format=sample_format, channels=channels, rate=fs,
                        frames_per_buffer=chunk, input=True)
        filename = 'record'+str(i)+'.wav'
        read_audio(stream, filename)

    for i in range(60//20):  # Number of total seconds to record/ Number of seconds per recording
        t1 = threading.Thread(target=save_audios, args=[i])
        x = i-1
        # send one earlier than being recorded
        t2 = threading.Thread(target=convert, args=[x])
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        if i == 2:
            flag = True
    if flag:
        convert(i)
        p.terminate()

    file = open("test.txt")  # Student speech file
    data = file.read()
    file.close()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(data)  # tokenizing sentence
    filtered_sentence = [w for w in word_tokens if w not in stop_words]
    filtered_sentence = []

    for w in word_tokens:  # Removing stop words
        if w not in stop_words:
            filtered_sentence.append(w)

    # creating a final file
    f = open('final.txt', 'w')
    for ele in filtered_sentence:
        f.write(ele+' ')
    f.close()

    # checking whether proctor needs to be alerted or not
    file = open("paper.txt")  # Question file
    data = file.read()
    file.close()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(data)  # tokenizing sentence
    filtered_questions = [w for w in word_tokens if not w in stop_words]
    filtered_questions = []

    for w in word_tokens:  # Removing stop words
        if w not in stop_words:
            filtered_questions.append(w)

    def common_member(a, b):
        a_set = set(a)
        b_set = set(b)

        # check length
        if len(a_set.intersection(b_set)) > 0:
            return(a_set.intersection(b_set))
        else:
            return([])

    comm = common_member(filtered_questions, filtered_sentence)
    print('Common words spoken by student:', len(comm))
    print(comm)


def Talking():

    def mouth_aspect_ratio(mouth):
        # compute the euclidean distances between the two sets of
        # vertical mouth landmarks (x, y)-coordinates
        A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
        B = dist.euclidean(mouth[4], mouth[8])  # 53, 57

        # compute the euclidean distance between the horizontal
        # mouth landmark (x, y)-coordinates
        C = dist.euclidean(mouth[0], mouth[6])  # 49, 55

        # compute the mouth aspect ratio
        mar = (A + B) / (2.0 * C)

        # return the mouth aspect ratio
        return mar

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=False, default='shape_predictor_68_face_landmarks.dat',
                    help="path to facial landmark predictor")
    ap.add_argument("-w", "--webcam", type=int, default=0,
                    help="index of webcam on system")
    args = vars(ap.parse_args())

    # define one constants, for mouth aspect ratio to indicate open mouth
    MOUTH_AR_THRESH = 0.692
    YAWNING = 0.85

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # grab the indexes of the facial landmarks for the mouth
    (mStart, mEnd) = (49, 68)

    # start the video stream thread
    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=0).start()
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    start_time = None
    last_face_detected = time.time()
    # time.sleep(1.0)

    frame_width = 840
    frame_height = 600

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
    # time.sleep(1.0)

    # set path in which you want to save images
    path = r'./captures'

    # changing directory to given path
    os.chdir(path)
    # cap = vs
    i = 0
    wait = 0

    def mouth(iframe):
        frame = imutils.resize(iframe, width=840)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the mouth coordinates, then use the
            # coordinates to compute the mouth aspect ratio
            mouth = shape[mStart:mEnd]

            mouthMAR = mouth_aspect_ratio(mouth)
            mar = mouthMAR
            # compute the convex hull for the mouth, then
            # visualize the mouth
            mouthHull = cv2.convexHull(mouth)

            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw text if mouth is open
            if mar > MOUTH_AR_THRESH and mar < YAWNING:
                print("Mouth is Open")
                global mouth_opened
                mouth_opened += 1
                if(mouth_opened > 0):
                    socketio.emit('output', {
                        "message": "Talking Detected", "title": "Please dont Talk !", "detection": "mouth"
                    })
                    # @socketio.on('end exam')
                    # def end_exam(data):
                    #     cam.release()
                    #     cv2.destroyAllWindows()
                    return True
                cv2.putText(frame, "Mouth is Open!", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            elif mar > YAWNING:
                print("Yawning")
                return False

        # Write the frame into the file 'output.avi'
        out.write(frame)
        # show the frame
        # cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

    # loop over frames from the video stream
    while True:
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
        # grab the frame from the threaded video file stream, resize it, and convert it to grayscale # channels)
        frame = vs.read()
        # Display the image

        cv2.imshow("Frame", frame)

        # wait for user to press any key
        key = cv2.waitKey(100)

        # wait variable is to calculate waiting time
        wait = wait+100

    # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            last_face_detected = time.time()

        # Check if there are more than one face in the frame
        if len(faces) > 1:
            if start_time is None:
                start_time = time.time()
            else:
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time >= 3:
                    difference = check_time_difference()
                    print(difference)
                    if (difference==True):
                        cv2.imwrite(f"C:\\Users\\Siddhesh Gadkar\\Desktop\\Sem8 workings\\Flask-Auto-Proctor\\warnings\\warn{dt_string}.jpg", frame)
                    socketio.emit('output', {
                        "message": "Another person detected", "title": "Someone is peeking in your screen"
                    })
                    # Display a warning message in cam
                    # cv2.putText(frame, "Warning: Another person detected", (10, 30),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        elif time.time() - last_face_detected > 2:
            # Display a warning message
            socketio.emit('output', {
                "message": "No Face detected", "title": ""
            })
            difference = check_time_difference()
            print(difference)
            if (difference==True):
                cv2.imwrite(f"C:\\Users\\Siddhesh Gadkar\\Desktop\\Sem8 workings\\Flask-Auto-Proctor\\warnings\\warn{dt_string}.jpg", frame)

            # cv2.putText(frame, "Warning: No face detected", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            start_time = None

        # Display the frame
        cv2.imshow("Live Camera", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if key == ord('q'):
            break
        # when it reaches to 3000 milliseconds # we will save that frame in given folder
        if wait == 3000:
            filename = 'Frame_'+str(i)+'.jpg'
            mouth(frame)

            # Save the images in given path
            cv2.imwrite(filename, frame)
            i = i+1
            wait = 0

        out.write(frame)
        try:
            ret, buffer = cv2.imencode('.jpeg', cv2.flip(frame, 1))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            pass
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

cam = cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from cam
    global out, capture, rec_frame
    file = pd.read_csv('./StudentDetails/StudentDetails.csv')
    df = pd.DataFrame(file.iloc[:, :].values)
    df = pd.DataFrame(file.iloc[-1:, :].values)
    det = df.values.tolist()
    Id = str(det[0][4])
    name = str(det[0][0])
    success, frame = cam.read()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (10, 159, 255), 2)
            # incrementing sample number
            sampleNum = sampleNum+1
            if (sampleNum%15==0):
                socketio.emit('progress', {
                "data": (sampleNum/15)*10})
            # saving the captured face in the dataset folder TrainingImage
            cv2.imwrite("TrainingImage" + os.sep + name + "."+Id +
                        '.'+str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
            # display the frame
            cv2.imshow('frame', frame)
        try:
            ret, buffer = cv2.imencode('.jpeg', cv2.flip(frame, 1))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            pass
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        elif sampleNum > 151:
            cam.release()
            break
    socketio.emit("data", {'data': 'index'})
    train_image()
    return "True"

# Mouth Open Detection


def mouth_open():
    return Talking()

# Capture image to Recognize later


def capture_image():
    Capture_Image.takeImages()

# Recognize Image for Attendance


def recognize_feed():
    a = str(Recognize.recognize_attendence())
    return a


@app.route('/recognize')
def recognize():
    return render_template('recognize.html', a=recognize_feed())

# / Route to home page or login page


@app.route('/')
def index():
    return render_template('index.html')

# / Route to home page or login page


@app.route('/home')
def home():
    return render_template('index.html')

# Route for Exam Camera


@app.route('/exam_cam')
def exam_cam():
    return Response(mouth_open(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Route for Running Exam Cam and Audio Code Simultaneously

@app.route('/exam_feed')
def exam_feed():
    try:
        p1 = Process(target=audio)
        p1.start()
        p2 = Process(target=exam_cam)
        p2.start()
        return Response(mouth_open(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except:
        return None


# Route to send video frames from backend to front end


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to register page


@app.route('/register')
def register():
    return render_template('register.html')

# Route to send data from registration page to Excel sheet


# user validation to send correct data in Database
userValidator = {
    '$jsonSchema': {
        'bsonType': 'object',
        'additionalProperties': True,
        'required': ['Name', 'class', 'rollNo', 'email', 'grNo', 'password', 'role', 'date'],
        'properties': {
            'Name': {
                    'bsonType': 'string',
                    'description': 'Must be a string'
            },
            'class': {
                'bsonType': 'string',
                'description': 'Must be a string'
            },
            'rollNo': {
                'bsonType': 'string',
                'description': 'Must be a string'
            },
            'email': {
                'bsonType': 'string',
                'description': 'Must be a string'
            },
            'grNo': {
                'bsonType': 'string',
                'description': 'Must be a string'
            },
            'password': {
                'bsonType': 'string',
                'description': 'Must be a string'
            },
            'role': {
                'bsonType': 'string',
                'description': 'Must be a string'
            },
            'date': {
                'bsonType': 'string',
                'description': 'Set to default value'
            }
        }
    }
}


@app.route('/gfg', methods=['GET', 'POST'])
def gfg():
    if(request.method == 'POST'):

        Name = str(request.form.get('fname'))
        email = request.form.get('email')
        className = request.form.get('className')
        rollno = request.form.get('rollno')
        regno = str(request.form.get('regno'))
        password = request.form.get('password')
        row = [Name, email, className, rollno, regno, password]
        with open("StudentDetails"+os.sep+"StudentDetails.csv", 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        print(time.time())
        # db.command("collMod", "user")
        now = datetime.now()
        post = {"Name": Name,
                "class": className,
                "rollNo": rollno,
                "email": email,
                "password": password,
                "grNo": regno,
                "role": 'Student',
                "date": now.strftime('%Y-%m-%d')}
        print(post)
        if(collection.find_one({"email": email}) == None and collection.find_one({"grNo": regno}) == None):
            try:

                collection.insert_one(post)
                video_feed()
                return render_template('camera.html', t=True)

            except Exception as e:
                return f'{e} validation failed'
        else:
            return "Email or GrNo Already Exist"


# Route to Verify login details from Excel sheet


@app.route('/login', methods=['POST'])
def login():
    register = int(request.form.get('register'))
    password = request.form.get('password')
    df = pd.read_csv('./StudentDetails/StudentDetails.csv',
                     usecols=["Id", "Password"])
    if(df[df.Id == register].empty):
        return render_template('index.html', e="Login Failed")
    elif(df[df.Password == password].empty):
        return render_template('index.html', e="Login Failed")
    elif(df[df.Id == register].empty and df[df.Password == password].empty):
        return render_template('index.html', e="Login Failed")
    else:
        return render_template('startExam.html', e="Login Success")


@app.route('/camera', methods=['GET', 'POST'])
def camera():
    return render_template('camera.html', text="Hello Sid")


# Route to Open Examination Page
@app.route('/exam')
def exam():
    return render_template('pythonexam.html')


def new_frames():  # generate frame by frame from camera
    while True:
        success, frame = cam.read()
        if success:
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass


@app.route('/new_feed')
def new_feed():
    return Response(new_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Main Method
if __name__ == "__main__":
    socketio.run(app, debug=True)

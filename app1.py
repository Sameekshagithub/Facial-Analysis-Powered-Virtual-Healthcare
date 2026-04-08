import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import speech_recognition as sr
from flask import Flask, render_template, request, session, flash, redirect, url_for, jsonify
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import sqlite3 as sql
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Flask setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Logging setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Q&A CSV for chatbot
qa_df = pd.read_csv("chatbot_qa.csv")

########################################################################################################
# HOME ROUTES

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/gohome')
def homepage():
    return render_template('index1.html')

@app.route('/enternew')
def new_user():
    return render_template('signup.html')

@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            nm = request.form['Name']
            phonno = request.form['MobileNumber']
            email = request.form['email']
            unm = request.form['Username']
            passwd = request.form['password']
            with sql.connect("agricultureuser.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO agriuser(name, phono, email, username, password) VALUES(?, ?, ?, ?, ?)", 
                            (nm, phonno, email, unm, passwd))
                con.commit()
                flash("Record successfully added")
        except Exception as e:
            con.rollback()
            flash(f"Error in insert operation: {e}")
            logging.error(str(e))
        finally:
            return render_template("result1.html")

@app.route('/userlogin')
def user_login():
    return render_template("login.html")

@app.route('/logindetails', methods=['POST', 'GET'])
def logindetails():
    if request.method == 'POST':
        usrname = request.form.get('username')
        passwd = request.form.get('password')
        if not usrname or not passwd:
            flash("Please provide both username and password.")
            return render_template('login.html')

        try:
            with sql.connect("sql.db") as con:
                cur = con.cursor()
                cur.execute("SELECT username, password FROM agriuser WHERE username = ?", (usrname,))
                account = cur.fetchone()
                if account and account[1] == passwd:
                    session['logged_in'] = True
                    flash("Login successful!")
                    return redirect(url_for('facial_expression'))
                else:
                    flash("Invalid credentials.")
        except sql.Error as e:
            flash(f"An error occurred: {str(e)}")
    return render_template('login.html')

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('index1.html')


##############################################################################################################
# EMOTION MODEL SETUP

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

model_path = r"C:\Users\samee\OneDrive\Desktop\suicide_depression\suicide_depression\suicide\facial_expression\your_model.keras"
if not os.path.exists(model_path):
    raise ValueError(f"Model file does not exist: {model_path}")
emotion_model = tf.keras.models.load_model(model_path)

img_width, img_height = 48, 48
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def cleanup_uploads():
    for file in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, file))
        except Exception as e:
            logging.error(f"Error deleting file {file}: {e}")
cleanup_uploads()

def preprocess_image(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (img_width, img_height))
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=-1)

def detect_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return frame[y:y+h, x:x+w]
    return None

def predict_emotion(frame):
    face = detect_face(frame)
    if face is None:
        return "Error: No face detected."
    processed = preprocess_image(face)
    prediction = emotion_model.predict(np.expand_dims(processed, axis=0))
    return emotion_labels[np.argmax(prediction)]

def recognize_face(image_path):
    if not os.path.exists(image_path):
        return "Error: Image not found."
    frame = cv2.imread(image_path)
    if frame is None:
        return "Error: Could not load image."
    emotion = predict_emotion(frame)
    return "Stressed" if emotion in ['angry', 'sad', 'fear', 'neutral', 'disgust'] else "Not Stressed"

def capture_multiple_images(num_images=3):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return [], "Error: Could not open webcam."
    paths = []
    for i in range(num_images):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return [], f"Error capturing image {i+1}."
        path = os.path.join(UPLOAD_FOLDER, f"img_{int(time.time())}_{i}.jpg")
        cv2.imwrite(path, frame)
        paths.append(path)
        time.sleep(1)
    cap.release()
    return paths, None


@app.route('/facial_expression', methods=['GET', 'POST'])
def facial_expression():
    if request.method == 'POST':
        try:
            image_paths, error = capture_multiple_images()
            if error:
                flash(error)
                return render_template('facial_expression.html')
            results = [recognize_face(p) for p in image_paths]
            face_result = "Stressed" if results.count("Stressed") >= 2 else "Not Stressed"
            with open('result.txt', 'w') as f:
                f.write(f"Face: {face_result}\n")
            return redirect(url_for('speech_recognition'))
        except Exception as e:
            flash(f"Unexpected error: {e}")
    return render_template('facial_expression.html')


@app.route('/speech', methods=['GET', 'POST'])
def speech_recognition():
    try:
        stress_data = pd.read_csv(r"C:\Users\samee\OneDrive\Desktop\suicide_depression\suicide_depression\suicide\tweet_emotions.csv")
        model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        model.fit(stress_data['content'], stress_data['sentiment'])

        if request.method == 'POST':
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                prediction = model.predict([text])
                speech_result = "Stressed" if prediction[0] in ['sadness', 'worry', 'anger'] else "Not Stressed"
                with open('result.txt', 'a') as file:
                    file.write(f"Speech: {speech_result}\n")
                return redirect(url_for('result_page', speech_result=speech_result, input_text=text))
            except Exception as e:
                flash(str(e))
    except Exception as e:
        flash(f"Error: {e}")
    return render_template('speech.html')


@app.route('/result_page')
def result_page():
    input_text = request.args.get('input_text', 'No input text')
    try:
        if os.path.exists('result.txt'):
            with open('result.txt', 'r') as f:
                content = f.read()
        else:
            content = "No results."
        results = [line.split(":")[1].strip() for line in content.splitlines() if ":" in line]
        stressed_count = results.count("Stressed")
        final = "Depressed - you need meditation" if stressed_count >= 2 else "Depression Not Detected"
        return render_template('result.html', result_content=content, final_evaluation=final, input_text=input_text)
    except Exception as e:
        return render_template('result.html', result_content="Error", final_evaluation="Error", input_text="N/A")


##############################################################################################################
# EMOTION Q&A CHATBOT ROUTES

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/get_qa_by_emotion', methods=["POST"])
def get_qa_by_emotion():
    emotion = request.json.get("emotion", "").capitalize()
    filtered = qa_df[qa_df["emotion"] == emotion]
    qa_list = filtered[["question", "answer"]].to_dict(orient="records")
    return jsonify({"qa": qa_list})

##############################################################################################################

if __name__ == '__main__':
    app.run(debug=True)
app = Flask(__name__, static_url_path='/static')
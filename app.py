import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import speech_recognition as sr
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, request, session, flash, redirect, url_for,jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import sqlite3 as sql
import logging
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

# Flask app initialization
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

########################################################################################################

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
                msg = "Record successfully added"
                flash(msg)
        except Exception as e:
            con.rollback()
            msg = f"Error in insert operation: {e}"
            logging.error(msg)
            flash(msg)
        finally:
            return render_template("result1.html", msg=msg)

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
            with sql.connect("agricultureuser.db") as con:
                cur = con.cursor()
                cur.execute("SELECT username, password FROM agriuser WHERE username = ?", (usrname,))
                account = cur.fetchone()

                if account:
                    database_user, database_password = account
                    if database_user == usrname and database_password == passwd:
                        # Successful login, set session
                        session['logged_in'] = True
                        flash("Login successful!")
                        # Redirect to index page
                        return redirect(url_for('facial_expression'))  # Redirect to index after login
                    else:
                        flash("Invalid password.")
                else:
                    flash("Username does not exist.")
        except sql.Error as e:
            flash(f"An error occurred: {str(e)}")

    return render_template('login.html')

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('index1.html')


##############################################################################################################

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Load the emotion recognition model
model_path = r"C:\Users\job01\Desktop\facial_expression\your_model.keras"
if not os.path.exists(model_path):
    raise ValueError(f"Model file does not exist: {model_path}")
emotion_model = tf.keras.models.load_model(model_path)

# Image dimensions and labels for the emotion recognition model
img_width, img_height = 48, 48
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Folder for storing captured images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Cleanup old images on server start
def cleanup_uploads():
    """Delete all files in the uploads folder."""
    for file in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Delete file
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {e}")

cleanup_uploads()  # Call cleanup at the start of the program

# Function to preprocess image for emotion model
def preprocess_image(frame):
    """Preprocess image for emotion model."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (img_width, img_height))
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=-1)

def detect_face(frame):
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Use the first detected face
        return frame[y:y+h, x:x+w]  # Crop the face region
    return None  # No face detected


# Function to predict emotion from image
def predict_emotion(frame):
    face = detect_face(frame)
    if face is None:
        return "Error: No face detected."
    processed_image = preprocess_image(face)
    predictions = emotion_model.predict(np.expand_dims(processed_image, axis=0))
    emotion_index = np.argmax(predictions)
    return emotion_labels[emotion_index]


# Function to recognize face and return stress prediction
def recognize_face(image_path):
    """Recognize emotion from an uploaded image."""
    if not os.path.exists(image_path):
        return "Error: Image not found."
    
    frame = cv2.imread(image_path)
    if frame is None:
        return "Error: Could not load image."
    
    face = detect_face(frame)
    if face is None:
        return "Error: No face detected."
    
    predicted_emotion = predict_emotion(face)
    print(f"Detected Emotion: {predicted_emotion}")  # Debugging line
    return "Stressed" if predicted_emotion in ['angry', 'sad', 'fear', 'neutral', 'disgust'] else "Not Stressed"


# Function to capture images from the webcam
# Function to capture multiple images from the webcam
def capture_multiple_images(num_images=6):
    """Capture multiple images from the webcam."""
    cap = cv2.VideoCapture(0)  # Open the webcam
    if not cap.isOpened():
        return [], "Error: Could not open webcam."
    
    image_paths = []
    for i in range(num_images):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return [], f"Error: Failed to capture image {i + 1}."
        
        image_path = os.path.join(UPLOAD_FOLDER, f"captured_image_{int(time.time())}_{i}.jpg")
        cv2.imwrite(image_path, frame)  # Save the image
        image_paths.append(image_path)

        # Add a slight delay between captures (optional)
        time.sleep(1)

    cap.release()
    return image_paths, None


# Function to train the Random Forest model for survey data


@app.route('/facial_expression', methods=['GET', 'POST'])
def facial_expression():
    """Handle facial recognition for stress analysis."""
    if request.method == 'POST':
        try:
            # Capture multiple images for facial recognition
            image_paths, error = capture_multiple_images(num_images=3)
            if error:
                flash(f"Error capturing images: {error}")
                return render_template('facial_expression.html')  # Reload with error message

            # Predict stress for each captured image
            face_results = [recognize_face(image_path) for image_path in image_paths]
            # Determine the final face result by majority vote
            face_result = "Stressed" if face_results.count("Stressed") >= 2 else "Not Stressed"

            # Save results to a text file
            with open('result.txt', 'w') as file:
                file.write(f"Face: {face_result}\n")

            # Log the results for debugging
            logging.info(f"Face results: {face_results}")

            # Redirect to the speech recognition step or results page
            return redirect(url_for('speech_recognition'))

        except Exception as e:
            flash(f"Unexpected error occurred: {e}")
            logging.error(f"Unexpected error occurred: {e}")
            return render_template('facial_expression.html')  # Reload with error message

    # Render facial recognition page for GET requests
    return render_template('facial_expression.html')

@app.route('/speech', methods=['GET', 'POST'])
def speech_recognition():
    """Handle speech recognition and display results, along with live emotion detection."""
    result_file = 'result.txt'
    try:
        # Log the start of speech recognition
        logging.info("Starting speech recognition")

        # Load the stress data for speech recognition
        stress_data = pd.read_csv(r"D:\stress_detection122\stress_detection\tweet_emotions.csv")
        model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        model.fit(stress_data['content'], stress_data['sentiment'])

        if request.method == 'POST':
            # Log before listening to the microphone
            logging.info("Listening for speech input...")

            # Listen and predict
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

            try:
                # Convert speech to text
                text = recognizer.recognize_google(audio)
                logging.info(f"Speech input: {text}")
                prediction = model.predict([text])
                speech_result = "Stressed" if prediction[0] in ['sadness', 'worry', 'anger'] else "Not Stressed"

                # Append the result to a file
                with open(result_file, 'a') as file:
                    file.write(f"Speech: {speech_result} \n")

                # Redirect to the result page
                return redirect(url_for('result_page', speech_result=speech_result, input_text=text))

            except sr.UnknownValueError:
                flash("Google Speech Recognition could not understand the audio")
                logging.error("Google Speech Recognition could not understand the audio")
            except sr.RequestError as e:
                flash(f"Could not request results from Google Speech Recognition service; {e}")
                logging.error(f"Could not request results from Google Speech Recognition service; {e}")

        # Render the speech page (along with the webcam feed and emotion)
        return render_template('speech.html')

    except Exception as e:
        flash(f"Error during speech recognition: {e}")
        logging.error(f"Error during speech recognition: {e}")
        return render_template('result.html', speech_result="Error")







      
@app.route('/result_page')
def result_page():
    """Display the final results after all analysis."""
    input_text = request.args.get('input_text', 'No input text')
    result_file = 'result.txt'

    try:
        # Read the result.txt content
        if os.path.exists(result_file):
            with open(result_file, 'r') as file:
                result_content = file.read()
        else:
            result_content = "No results available. File missing or empty."

        # Display the content of the result file
        print("Result File Content:")
        print(result_content)

        # Split the result content into lines and count "Stressed"
        results = [line.split(":")[1].strip() for line in result_content.splitlines() if ":" in line]
        stressed_count = results.count("Stressed")

        # Print the count for debugging
        print(f"Stressed count: {stressed_count}")

        # Determine final evaluation based on the majority
        if stressed_count >=2:  # If at least 2 out of 3 are 'Stressed'
            final_evaluation = "Suicide Tendancy Detected"
        else:
            final_evaluation = "Suicide Tendancy Not detected"

        # Print error if the final result is "Not Stressed" despite "Stressed" count being high
        if final_evaluation == "Not Stressed" and stressed_count >= 2:
            print("Error: The final result is 'Not Stressed', but there are multiple 'Stressed' indications.")

        # Render result.html with the results and final evaluation
        return render_template(
            'result.html',
            result_content=result_content,
            final_evaluation=final_evaluation,
            input_text=input_text
        )

    except Exception as e:
        print(f"Error during result processing: {e}")
        return render_template(
            'result.html',
            result_content="Error loading results.",
            final_evaluation="Error",
            input_text="No input text"
        )
    

 # Load and preprocess the dataset
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    data['combinedText'] = data['questionTitle'] + " " + data['questionText'] + " " + data['topic']
    return data

# Train the TF-IDF model
def train_chatbot(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['combinedText'])
    return vectorizer, tfidf_matrix

# Generate a response
def get_response(user_query, data, vectorizer, tfidf_matrix):
    user_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
    best_match_idx = similarity_scores.argmax()
    best_match_score = similarity_scores.max()

    if best_match_score > 0.1:  # Threshold for relevance
        question = data.loc[best_match_idx, 'questionTitle']
        answer = data.loc[best_match_idx, 'answerText']
        topic = data.loc[best_match_idx, 'topic']
        return {
            "topic": topic,
            "question": question,
            "answer": answer
        }
    else:
        return {
            "error": "I'm sorry, I couldn't find a relevant response. Could you rephrase your query?"
        }
file_path = '20200325_counsel_chat.csv'
data = load_and_prepare_data(file_path)
vectorizer, tfidf_matrix = train_chatbot(data)

@app.route('/chat')
def home2():
    return render_template('chat.html')

@app.route('/get_response', methods=['POST'])
def chat():
    user_query = request.json.get('query', '')
    if not user_query:
        return jsonify({"error": "Query cannot be empty."})

    response = get_response(user_query, data, vectorizer, tfidf_matrix)
    return jsonify(response)




    



if __name__ == '__main__':
    app.run(debug=True)
import numpy as np
import cv2
import pickle
import os

# Print current working directory for diagnostic purposes
print("Current working directory:", os.getcwd())

# Camera settings
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75  # Confidence threshold for predictions
font = cv2.FONT_HERSHEY_SIMPLEX

# Start the camera
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)   # Frame width
cap.set(4, frameHeight)  # Frame height
cap.set(10, brightness)  # Brightness

# Path to the trained model
model_path = r"C:\Users\Habib\Downloads\b-471\model_trained.p"  
# Alternatively, if the file is in the same folder as this script, use:
# model_path = "model_trained.p"

# Check if the model file exists
if not os.path.exists(model_path):
    print("❌ Model file not found at:", model_path)
    print("Please ensure 'model_trained.p' is in the correct directory.")
    exit()

# Load the trained model
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# Preprocessing functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Function to map class number to traffic sign name
def getClassName(classNo):
    classes = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
        'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
        'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    return classes[classNo] if 0 <= classNo < len(classes) else "Unknown"

# Main loop for prediction
while True:
    success, imgOriginal = cap.read()
    if not success:
        print("❌ Failed to capture image. Check your camera connection.")
        break

    # Preprocess the image for prediction
    img = cv2.resize(imgOriginal, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    # Predict the class of the image
    try:
        predictions = model.predict(img)
        classIndex = np.argmax(predictions)
        probabilityValue = np.max(predictions)
    except Exception as e:
        print("❌ Error during prediction:", e)
        break

    # Display the prediction on the original image if the confidence is high enough
    if probabilityValue > threshold:
        cv2.putText(imgOriginal, f"Class: {getClassName(classIndex)}", (20, 35), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, f"Probability: {round(probabilityValue * 100, 2)}%", (20, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


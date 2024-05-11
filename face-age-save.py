import cv2
from deepface import DeepFace
import os

# Load the image
img_path = "images/abd-al-rhmman.jpg"
img = cv2.imread(img_path)

# Analyze the image
results = DeepFace.analyze(img, actions=("gender", "age", "race", "emotion"))

# Extract directory path and filename without extension
directory = os.path.dirname(img_path)
filename = os.path.splitext(os.path.basename(img_path))[0]

# Create a text file in the same directory as the image
txt_path = os.path.join(directory, f"{filename}.txt")

# Write the analysis results to the text file
with open(txt_path, "w") as file:
    for result in results:
        for key, value in result.items():
            if isinstance(value, tuple):  # If it's a tuple of predictions
                max_index = value.index(max(value))
                max_prediction = DeepFace.actions[key][max_index]
                max_confidence = value[max_index]
                file.write(f"{key}: {max_prediction} = {max_confidence}\n")
            else:
                file.write(f"{key}: {value}\n")

print(f"Analysis results saved to {txt_path}")

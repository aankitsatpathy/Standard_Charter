import numpy as np
import cv2
import pytesseract
import re
import csv
# Load image
image_path = "/content/d8849986f23d93e38061ce3aae9a445b.jpg"  # Update with the correct filename
image = cv2.imread(image_path)

if image is None:
    print("❌ Error: Could not load the image. Check the file path!")
    exit()

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding (improves OCR accuracy)
_, thresh1 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Save processed image
cv2.imwrite("/content/processed_image.jpg", thresh1)

# Tesseract OCR configuration
config = ('-l eng --oem 1 --psm 3')

# Extract text from image
# ✅ Extract text from image
raw_text = pytesseract.image_to_string(thresh1, config=config)

# ✅ Clean and split text into lines
cleaned_lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

# ✅ Select specific lines (2nd, 4th, 5th, 7th, 13th, 14th)
selected_lines = [cleaned_lines[i-1] for i in [2, 4, 5, 7, 13] if i <= len(cleaned_lines)]

# ✅ Print only selected lines
print("\n✅ **Filtered Extracted Text:**")
for line in selected_lines:
    print(line)


# ✅ **Function to determine ID type (Aadhaar or PAN)**
def detect_id_type(text):
    if "Permanent Account Number Card" in text:
        return "PAN"
    elif "Aadhaar" in text or re.search(r"\d{4}\s\d{4}\s\d{4}", text):
        return "Aadhaar"
    else:
        return "Unknown"

# ✅ **Function to extract Aadhaar details**
def extract_aadhaar_details(text):
    lines = text.split("\n")
    cleaned_lines = [line.strip() for line in lines if line.strip()]

    if len(cleaned_lines) < 7:
        print("❌ Error: Aadhaar text doesn't have enough lines!")
        return None, None, None, None

    name = cleaned_lines[3]  # 4th line
    dob_match = re.search(r"\d{2}/\d{2}/\d{4}", cleaned_lines[4])  # 5th line after 'DOB :'
    dob = dob_match.group() if dob_match else None

    gender_match = re.search(r"/\s*(Male|Female|Other)", cleaned_lines[5], re.IGNORECASE)  # 6th line after '/'
    gender = gender_match.group(1) if gender_match else None

    aadhaar_match = re.fullmatch(r"\d{4}\s\d{4}\s\d{4}", cleaned_lines[6])  # 7th line
    aadhaar_number = aadhaar_match.group().replace(" ", "") if aadhaar_match else None

    return name, dob, gender, aadhaar_number

# ✅ **Function to extract PAN details**
def extract_pan_details(text):
    lines = text.split("\n")
    cleaned_lines = [line.strip() for line in lines if line.strip()]

    pan_number, name, dob = None, None, None

    for i, line in enumerate(cleaned_lines):
        if "e - Permanent Account Number Card" in line and i + 4 < len(cleaned_lines):
            pan_number = cleaned_lines[i + 4]  # PAN Number is on the 5th line

        if "Name" in line and i + 1 < len(cleaned_lines):
            # Extract name and remove "Father's Name" if present
            name = cleaned_lines[i + 1]
            if "Father's Name" in name:
                name = name.split("Father's Name")[0].strip()

        if "Date of Birth" in line and i + 1 < len(cleaned_lines):
            dob_match = re.search(r"\d{2}/\d{2}/\d{4}", cleaned_lines[i + 1])
            dob = dob_match.group() if dob_match else None

    return name, dob, pan_number


# ✅ **Face Extraction**
def extract_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        print("❌ No face detected on the ID!")
        return None

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_path = "/content/extracted_face.jpg"
        cv2.imwrite(face_path, face)
        return face_path

    return None

# ✅ **Aadhaar Validation (Verhoeff Algorithm)**
def verhoeff_generate(aadhaar):
    verhoeff_table_d = [
        [0,1,2,3,4,5,6,7,8,9],
        [1,2,3,4,0,6,7,8,9,5],
        [2,3,4,0,1,7,8,9,5,6],
        [3,4,0,1,2,8,9,5,6,7],
        [4,0,1,2,3,9,5,6,7,8],
        [5,9,8,7,6,0,4,3,2,1],
        [6,5,9,8,7,1,0,4,3,2],
        [7,6,5,9,8,2,1,0,4,3],
        [8,7,6,5,9,3,2,1,0,4],
        [9,8,7,6,5,4,3,2,1,0]
    ]

    verhoeff_table_p = [
        [0,1,2,3,4,5,6,7,8,9],
        [1,5,7,6,2,8,3,0,9,4],
        [5,8,0,3,7,9,6,1,4,2],
        [8,9,1,6,0,4,3,5,2,7],
        [9,4,5,3,1,2,6,8,7,0],
        [4,2,8,6,5,7,3,9,0,1],
        [2,7,9,3,8,0,6,4,1,5],
        [7,0,4,6,9,1,3,2,5,8]
    ]

    check_digit = 0
    aadhaar_reversed = aadhaar[::-1]

    for i, num in enumerate(aadhaar_reversed):
        check_digit = verhoeff_table_d[check_digit][verhoeff_table_p[i % 8][int(num)]]

    return check_digit

def verify_aadhaar(aadhaar):
    return verhoeff_generate(aadhaar) == 0

# ✅ **Process Based on ID Type**
id_type = detect_id_type(raw_text)

if id_type == "Aadhaar":
    name, dob, gender, number = extract_aadhaar_details(raw_text)
    valid = verify_aadhaar(number)
elif id_type == "PAN":
    name, dob, number = extract_pan_details(raw_text)
    gender = "N/A"
    valid = True
else:
    print("❌ Unknown ID type!")
    exit()

# ✅ **Extract Face**
face_path = extract_face(image)

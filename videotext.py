import ffmpeg
import whisper
import pandas as pd
import re
import os

def extract_audio(video_path, audio_path="audio.wav"):
    try:
        ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True, quiet=True)
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def transcribe_audio(audio_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def extract_loan_data(text):
    patterns = {
        "person_age": r"(?i)age[\s:]+(\d+\.?\d*)",
        "person_gender": r"(?i)gender[\s:]+(male|female)",
        "person_education": r"(?i)education[\s:]+(Bachelor|Associate|High School|Master|Doctorate)",
        "person_income": r"(?i)income[\s:]+(\d+\.?\d*)",
        "person_emp_exp": r"(?i)employment experience[\s:]+(\d+)",
        "person_home_ownership": r"(?i)home ownership[\s:]+(rent|own|mortgage)",
        "loan_amnt": r"(?i)loan amount[\s:]+(\d+\.?\d*)",
        "loan_intent": r"(?i)loan intent[\s:]+(EDUCATION|MEDICAL|VENTURE|PERSONAL|DEBTCONSOLIDATION|Homeimprovement)",
        "loan_int_rate": r"(?i)interest rate[\s:]+(\d+\.?\d*)",
        "loan_percent_income": r"(?i)percent income[\s:]+(\d+\.?\d*)",
        "cb_person_cred_hist_length": r"(?i)credit history length[\s:]+(\d+\.?\d*)",
        "credit_score": r"(?i)credit score[\s:]+(\d+)",
        "previous_loan_defaults_on_file": r"(?i)previous loan defaults[\s:]+(yes|no)"
    }
    data = {key: None for key in patterns}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            data[key] = match.group(1)

    # Data validation
    try:
        if data['person_age']: data['person_age'] = float(data['person_age'])
        if data['person_income']: data['person_income'] = float(data['person_income'])
        if data['person_emp_exp']: data['person_emp_exp'] = int(data['person_emp_exp'])
        if data['loan_amnt']: data['loan_amnt'] = float(data['loan_amnt'])
        if data['loan_int_rate']: data['loan_int_rate'] = float(data['loan_int_rate'])
        if data['loan_percent_income']: data['loan_percent_income'] = float(data['loan_percent_income'])
        if data['cb_person_cred_hist_length']: data['cb_person_cred_hist_length'] = float(data['cb_person_cred_hist_length'])
        if data['credit_score']: data['credit_score'] = int(data['credit_score'])
    except ValueError as e:
        print(f"Data validation error: {e}")
    
    return data

def save_to_csv(data, csv_path="output.csv"):
    df = pd.DataFrame([data])
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

def video_to_csv(video_path, csv_path="output.csv"):
    if not os.path.exists(video_path):
        print("Video file not found.")
        return

    audio_path = extract_audio(video_path)
    if not audio_path:
        print("Failed to extract audio.")
        return

    text = transcribe_audio(audio_path)
    if not text:
        print("Failed to transcribe audio.")
        return

    loan_data = extract_loan_data(text)
    save_to_csv(loan_data, csv_path)
    os.remove(audio_path)
    print("Processing complete.")

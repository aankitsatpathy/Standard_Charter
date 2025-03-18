import ffmpeg
import whisper
import pandas as pd
import os

def extract_audio(video_path, audio_path="audio.wav"):
    """ Extract audio from video """
    ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True, quiet=True)
    return audio_path

def transcribe_audio(audio_path):
    """ Transcribe audio to text using Whisper """
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

def process_video(video_path):
    """ Extract text from video """
    audio_path = extract_audio(video_path)
    text = transcribe_audio(audio_path)
    os.remove(audio_path)  # Cleanup extracted audio
    return text

def create_separate_csvs(video_paths):
    """ Process each video and save data to separate CSVs """
    # Expected categories
    categories = [
        "person_age", "person_gender", "person_education", "person_income", "person_emp_exp",
        "person_home_ownership", "loan_amnt", "loan_intent", "loan_int_rate", "loan_percent_income",
        "cb_person_cred_hist_length", "credit_score", "previous_loan_defaults_on_file"
    ]
    
    # Ensure correct number of videos provided
    if len(video_paths) != len(categories):
        print(f"Error: Expected {len(categories)} videos, but got {len(video_paths)}.")
        return
    
    # Extract text and save to separate CSVs
    for i, category in enumerate(categories):
        print(f"Processing {category}...")
        text = process_video(video_paths[i]).strip()
        df = pd.DataFrame([{category: text}])
        csv_path = f"{category}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {category} data to {csv_path}")

video_paths = [
        "age.mp4", "gender.mp4", "education.mp4", "income.mp4", "emp_exp.mp4", "home_ownership.mp4",
        "loan_amount.mp4", "loan_intent.mp4", "loan_int_rate.mp4", "loan_percent_income.mp4",
        "cred_hist_length.mp4", "credit_score.mp4", "previous_defaults.mp4"
]
create_separate_csvs(video_paths)
print("hello world")

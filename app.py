from flask import Flask, render_template
from googleapiclient.discovery import build
from quickstart import process_predictions, poll_until_complete, poll_for_completion
import asyncio
import os
import chromadb
from datetime import datetime
from dotenv import load_dotenv
from typing import List
from hume import AsyncHumeClient
from hume.expression_measurement.batch import Face, Models
from hume.expression_measurement.batch.types import UnionPredictResult
from chroma_intro import build_client
app = Flask(__name__)

# YouTube Data API configuration
load_dotenv()
HUME_API_KEY = os.getenv("HUME_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")  # Get your API key from Google Cloud
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def get_video_url(song_title):
    """Fetch the first YouTube video result for a given song title."""
    search_response = youtube.search().list(
        q=song_title,
        part='snippet',
        maxResults=1,
        type='video'
    ).execute()

    if search_response['items']:
        video_id = search_response['items'][0]['id']['videoId']
        return f"https://www.youtube.com/embed/{video_id}"
    return None

@app.route('/')
async def index():
    # Predefined song titles
    
    load_dotenv()
    HUME_API_KEY = os.getenv("HUME_API_KEY")

    # Initialize an authenticated client
    client = AsyncHumeClient(api_key=HUME_API_KEY)

    # Define the URL(s) of the files you would like to analyze
    job_urls = ["https://hume-tutorials.s3.amazonaws.com/faces.zip"]

    # Create configurations for each model you would like to use (blank = default)
    face_config = Face()

    # Create a Models object
    models_chosen = Models(face=face_config)

    # Start an inference job and print the job_id
    job_id = await client.expression_measurement.batch.start_inference_job(
        urls=job_urls, models=models_chosen
    )
    print(f"Job ID: {job_id}")

    # Await the completion of the inference job with timeout and exponential backoff
    await poll_for_completion(client, job_id, timeout=120)

    # After the job is over, access its predictions
    job_predictions = await client.expression_measurement.batch.get_job_predictions(
        id=job_id
    )
    
    # Print the raw prediction output
    # print(job_predictions)

    # Define parameters for processing predictions
    start_time = 0          # Start time in seconds, relative to when the inference was made
    end_time = 12           # End time in seconds, relative to when the inference was made
    n_top_values = 1        # Number of top emotions to display
    peak_threshold = 0.7    # Threshold for peaked emotions
    
    # Process and display the predictions
    song_recs = process_predictions(
        job_predictions, start_time, end_time, n_top_values, peak_threshold
    )
    videos = []

    # Fetch YouTube video URLs for each song title
    for song in song_recs:
        video_url = get_video_url(song)
        if video_url:
            videos.append({'title': song, 'video_url': video_url})

    return render_template('index.html', videos=videos)

if __name__ == '__main__':
    app.run(debug=True)
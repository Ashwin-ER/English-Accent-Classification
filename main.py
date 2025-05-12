import streamlit as st
import os
import tempfile
import uuid
import re
import numpy as np
import requests
import time
import subprocess
import pandas as pd
from pydub import AudioSegment
import speech_recognition as sr
from urllib.parse import urlparse
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
import nltk
import os

# Automatically find or create a directory for NLTK data
def get_nltk_data_dir():
    # Get the list of NLTK data directories
    nltk_data_dirs = nltk.data.path

    # Try to find a directory with write access
    for directory in nltk_data_dirs:
        if os.access(directory, os.W_OK):  # Check if we have write access to the directory
            return directory

    # If no directory with write access is found, create one in the user's home directory
    home_dir = os.path.expanduser("~")
    custom_dir = os.path.join(home_dir, "nltk_data")
    os.makedirs(custom_dir, exist_ok=True)
    return custom_dir

# Set the NLTK data path
nltk_data_dir = get_nltk_data_dir()
os.environ['NLTK_DATA'] = nltk_data_dir

# Download the 'punkt' tokenizer data into the valid directory
nltk.download('punkt', download_dir=nltk_data_dir)

print(f"NLTK data will be stored in: {nltk_data_dir}")




# Set page config
st.set_page_config(
    page_title="English Accent Analyzer",
    page_icon="🎤",
    layout="wide"
)

# Title and description
st.title("🎤 English Accent Analyzer")
st.markdown("""
This tool analyzes the speaker's accent from a video. Simply provide a public video URL
(e.g., a direct MP4 file or Loom video), and the system will:
1. Extract the audio
2. Transcribe the speech
3. Analyze the accent
4. Provide a classification with confidence score
""")

# Accent feature patterns dictionary
ACCENT_PATTERNS = {
    "American": {
        "phonetic_patterns": ["r after vowels", "t flapping", "o as 'ah'"],
        "word_markers": ["gonna", "wanna", "y'all", "awesome", "totally", "like"],
        "spelling_markers": ["color", "center", "defense"],
        "description": "Characterized by rhotic pronunciation (pronouncing 'r' after vowels), flapping of 't' sounds, and specific vocabulary."
    },
    "British": {
        "phonetic_patterns": ["non-rhotic", "t glottal stop", "broader 'a'"],
        "word_markers": ["proper", "brilliant", "cheers", "mate", "bloody", "quite"],
        "spelling_markers": ["colour", "centre", "defence"],
        "description": "Typically non-rhotic (dropping 'r' after vowels), with clear 't' pronunciation and distinctive vocabulary."
    },
    "Australian": {
        "phonetic_patterns": ["rising intonation", "i sound change", "non-rhotic"],
        "word_markers": ["mate", "no worries", "arvo", "reckon", "heaps"],
        "spelling_markers": ["colour", "centre", "defence"],
        "description": "Features rising intonation at sentence ends, extended vowels, and unique slang terms."
    },
    "Indian": {
        "phonetic_patterns": ["retroflex consonants", "v/w confusion", "stress timing"],
        "word_markers": ["actually", "itself", "only", "kindly", "prepone"],
        "spelling_markers": ["colour", "centre", "defence"],
        "description": "Characterized by retroflex consonants, syllable-timing, and specific Indian English vocabulary."
    },
    "Canadian": {
        "phonetic_patterns": ["canadian raising", "rhotic", "about pronunciation"],
        "word_markers": ["eh", "sorry", "toque", "washroom", "loonie", "double-double"],
        "spelling_markers": ["colour", "centre", "defence"],
        "description": "Blends American and British features, with distinctive 'about' pronunciation and unique vocabulary."
    }
}

# Function to download a video from a URL
def download_video(url):
    """Download video from URL to a temporary file"""
    try:
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.mp4")
        
        # Handle Loom URLs
        if "loom.com" in url:
            # Extract video ID from Loom URL
            if "/share/" in url:
                video_id = url.split("/share/")[1].split("?")[0]
            else:
                # Try to extract ID from other formats
                video_id = url.split("/")[-1]
            
            # Construct direct download URL (this is approximate and may need adjustment)
            direct_url = f"https://cdn.loom.com/sessions/thumbnails/{video_id}.mp4"
        else:
            direct_url = url
        
        # Download the video
        response = requests.get(direct_url, stream=True)
        if response.status_code != 200:
            return None, "Failed to download video: HTTP Error " + str(response.status_code)
        
        # Save to temporary file
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        
        return temp_path, None
    
    except Exception as e:
        return None, f"Error downloading video: {str(e)}"

# Function to extract audio using FFmpeg directly
def extract_audio(video_path):
    """Extract audio from video file using FFmpeg"""
    try:
        # Create a temporary file for audio
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
        
        # Extract audio using FFmpeg
        cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", audio_path]
        
        # Run FFmpeg command
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Check if FFmpeg was successful
        if process.returncode != 0:
            return None, f"Error extracting audio with FFmpeg: {stderr.decode()}"
        
        # Check if file exists and has size > 0
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            return None, "Failed to extract audio: Output file is empty or doesn't exist"
        
        return audio_path, None
    
    except Exception as e:
        return None, f"Error extracting audio: {str(e)}"

# Function to check if FFmpeg is installed
def check_ffmpeg():
    try:
        process = subprocess.Popen(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return process.returncode == 0
    except:
        return False

# Function to transcribe audio
def transcribe_audio(audio_path):
    """Transcribe audio file using speech recognition"""
    try:
        # Load audio file using pydub for processing
        audio = AudioSegment.from_wav(audio_path)
        
        # Split audio into 30-second chunks (to handle large files)
        chunk_length = 30 * 1000  # 30 seconds
        chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        full_transcript = ""
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Export chunk to temporary file
            chunk_path = os.path.join(tempfile.gettempdir(), f"chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            
            # Transcribe chunk
            with sr.AudioFile(chunk_path) as source:
                audio_data = recognizer.record(source)
                try:
                    # Using Google's speech recognition
                    transcript = recognizer.recognize_google(audio_data)
                    full_transcript += " " + transcript
                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    st.warning(f"Error transcribing chunk {i+1}: {str(e)}")
            
            # Remove temporary chunk file
            try:
                os.remove(chunk_path)
            except:
                pass
        
        return full_transcript.strip(), None
    
    except Exception as e:
        return None, f"Error transcribing audio: {str(e)}"

# Function to analyze accent
def analyze_accent(transcript):
    """Analyze the accent based on transcript"""
    if not transcript:
        return None, "No speech detected in the audio"
    
    # Lowercased transcript for analysis
    text = transcript.lower()
    
    # Tokenize
    words = word_tokenize(text)
    
    # Track scores for each accent
    scores = {accent: 0.0 for accent in ACCENT_PATTERNS}
    
    # Analyze word markers
    word_count = Counter(words)
    total_words = len(words)
    
    for accent, patterns in ACCENT_PATTERNS.items():
        # Check for word markers
        marker_count = sum(word_count.get(marker.lower(), 0) for marker in patterns["word_markers"])
        marker_score = marker_count / max(1, min(20, total_words)) * 25  # Scale factor
        scores[accent] += marker_score
        
        # Check for spelling markers (in full text)
        spelling_matches = sum(1 for marker in patterns["spelling_markers"] if marker.lower() in text)
        scores[accent] += spelling_matches * 5
        
        # Check for phonetic patterns (approximated via text analysis)
        if accent == "American":
            if bool(re.search(r'\b(gonna|wanna|gotta)\b', text)):
                scores[accent] += 15
            if bool(re.search(r'\b(movie|duty|beauty)\b', text)):  # Words with potential t-flapping
                scores[accent] += 10
                
        elif accent == "British":
            if bool(re.search(r'\b(whilst|amongst|learnt|spelt)\b', text)):
                scores[accent] += 15
            if bool(re.search(r'\b(schedule|lieutenant|garage)\b', text)):  # Words pronounced differently
                scores[accent] += 10
                
        elif accent == "Australian":
            if bool(re.search(r'\b(arvo|brekkie|footy|ute)\b', text)):
                scores[accent] += 20
            if bool(re.search(r'\b(reckon|heaps|mate)\b', text)):
                scores[accent] += 10
                
        elif accent == "Indian":
            if bool(re.search(r'\b(itself|only|kindly|actually)\b', text)):
                scores[accent] += 15
            if bool(re.search(r'\b(prepone|timepass|batch-mate)\b', text)):
                scores[accent] += 15
                
        elif accent == "Canadian":
            if bool(re.search(r'\b(eh|toque|washroom|loonie)\b', text)):
                scores[accent] += 20
            if bool(re.search(r'\b(about|house|out)\b', text)):  # Words with potential Canadian raising
                scores[accent] += 10
    
    # Find best match
    best_accent = max(scores, key=scores.get)
    
    # Calculate confidence
    max_score = scores[best_accent]
    total_score = sum(scores.values())
    
    # Normalize confidence to 0-100%
    if total_score > 0:
        confidence = (max_score / total_score) * 100
        # Apply sigmoid transformation to spread confidence values
        confidence = 100 / (1 + np.exp(-0.05 * (confidence - 50)))
    else:
        confidence = 30  # Default when not enough indicators found
    
    # Get next best accent
    scores_copy = scores.copy()
    scores_copy.pop(best_accent)
    second_best = max(scores_copy, key=scores_copy.get) if scores_copy else None
    
    result = {
        "accent": best_accent,
        "confidence": min(round(confidence, 1), 95.0),  # Cap confidence at 95%
        "description": ACCENT_PATTERNS[best_accent]["description"],
        "transcript": transcript,
        "detailed_scores": {k: round(v, 2) for k, v in scores.items()},
        "second_best": second_best
    }
    
    return result, None

# Function to validate URL
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# Main application
def main():
    # Check if FFmpeg is installed
    ffmpeg_installed = check_ffmpeg()
    if not ffmpeg_installed:
        st.error("""
        ⚠️ FFmpeg is not installed or not found in PATH. This tool requires FFmpeg for audio extraction.
        
        Please install FFmpeg:
        - Windows: Download from ffmpeg.org and add to PATH
        - macOS: `brew install ffmpeg`
        - Linux: `sudo apt install ffmpeg`
        
        See the installation guide for detailed instructions.
        """)
        st.stop()
    
    # Input section
    st.header("1. Input Video")
    
    url = st.text_input("Enter public video URL (MP4 or Loom link):", 
                         help="For example: https://loom.com/share/your-video-id or https://example.com/video.mp4")
    
    # Alternatively allow file upload
    uploaded_file = st.file_uploader("Or upload a video file:", type=["mp4", "mov", "avi", "webm"])
    
    # Add demo videos

    
    # Process section
    if st.button("Analyze Accent", type="primary"):
        if not url and not uploaded_file:
            st.error("Please provide either a video URL or upload a video file.")
            return
        
        with st.spinner("Processing video..."):
            # Case 1: URL provided
            if url:
                if not is_valid_url(url):
                    st.error("Please enter a valid URL.")
                    return
                
                # Step 1: Download video
                progress_bar = st.progress(0)
                st.info("Downloading video...")
                video_path, error = download_video(url)
                progress_bar.progress(25)
                
                if error:
                    st.error(error)
                    return
                
            # Case 2: File uploaded
            else:
                # Save uploaded file to temp location
                progress_bar = st.progress(20)
                video_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp4")
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                progress_bar.progress(25)
            
            # Step 2: Extract audio
            st.info("Extracting audio...")
            audio_path, error = extract_audio(video_path)
            progress_bar.progress(50)
            
            if error:
                st.error(error)
                return
            
            # Step 3: Transcribe speech
            st.info("Transcribing speech...")
            transcript, error = transcribe_audio(audio_path)
            progress_bar.progress(75)
            
            if error:
                st.error(error)
                return
            
            if not transcript or len(transcript) < 10:
                st.error("Could not detect enough speech in the video. Please make sure the video contains clear English speech.")
                return
            
            # Step 4: Analyze accent
            st.info("Analyzing accent...")
            result, error = analyze_accent(transcript)
            progress_bar.progress(100)
            
            if error:
                st.error(error)
                return
            
            # Clean up
            try:
                os.remove(video_path)
                os.remove(audio_path)
            except:
                pass
            
            # Display results
            st.header("2. Analysis Results")
            
            # Summary box
            st.success(f"✅ **Accent Analysis Complete!**")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Primary Accent")
                
                # Create a more visually appealing primary result
                primary_accent = result['accent']
                confidence = result['confidence']
                
                # Display accent with colored background based on confidence
                color = "green" if confidence > 75 else "orange" if confidence > 50 else "red"
                
                # Create metrics for primary result
                st.metric(
                    label=f"{primary_accent} English", 
                    value=f"{confidence}% Confidence"
                )
                
                # Create description card
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 5px; background-color: #0e1117;">
                    <p><strong>Accent Characteristics:</strong></p>
                    <p>{result['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display secondary influence if available
                if result['second_best']:
                    st.markdown("""
                    <div style="margin-top: 20px;">
                        <p><strong>Secondary accent influence:</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info(f"**{result['second_best']} English**")
            
            with col2:
                st.subheader("Accent Distribution")
                
                # Format detailed scores
                scores = result['detailed_scores']
                accents = list(scores.keys())
                values = list(scores.values())
                
                # Normalize values for better visualization
                max_val = max(values)
                if max_val > 0:
                    normalized = [v/max_val*100 for v in values]
                else:
                    normalized = values
                    
                # Create a pie chart for accent distribution
                fig, ax = plt.subplots(figsize=(8, 8))
                wedges, texts, autotexts = ax.pie(
                    values, 
                    labels=accents,
                    autopct='%1.1f%%', 
                    startangle=90,
                    shadow=True,
                    explode=[0.1 if accent == primary_accent else 0 for accent in accents],
                    colors=sns.color_palette('viridis', len(accents))
                )
                
                # Style the pie chart
                plt.setp(autotexts, size=10, weight="bold")
                ax.set_title('Accent Distribution Analysis', fontsize=16)
                
                # Display the pie chart
                st.pyplot(fig)
                
                # Create a proper dataframe for charting
                import pandas as pd
                chart_df = pd.DataFrame({
                    'Accent': accents,
                    'Score': normalized
                })
                
                # Display as bar chart with proper formatting
                st.bar_chart(
                    chart_df.set_index('Accent'),
                    use_container_width=True
                )
                
                # Display detailed scores in tabular format
                st.subheader("Raw Accent Scores")
                score_df = pd.DataFrame({
                    'Accent': list(scores.keys()),
                    'Raw Score': list(scores.values()),
                    'Confidence (%)': [round((v/max_val)*100, 1) if max_val > 0 else 0 for v in scores.values()]
                })
                score_df = score_df.sort_values('Raw Score', ascending=False)
                st.dataframe(score_df, use_container_width=True, hide_index=True)
            
            # Display transcript
            st.header("3. Transcript")
            st.text_area("Speech Transcript", result['transcript'], height=200)
            
            st.info("""
            **Note:** This analysis is based on textual patterns and common linguistic features. 
            For professional accent evaluation, consult with a trained linguist.
            """)

if __name__ == "__main__":
    main()

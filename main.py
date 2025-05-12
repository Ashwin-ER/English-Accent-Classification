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

# Ensure necessary NLTK data is available
# Streamlit Cloud allows downloads to the user's home directory
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    st.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

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
(e.g., a direct MP4 file or Loom video), or upload a video file. The system will:
1. Extract the audio
2. Transcribe the speech
3. Analyze the accent
4. Provide a classification with confidence score
""")

# Accent feature patterns dictionary
ACCENT_PATTERNS = {
    "American": {
        "phonetic_patterns": ["r after vowels", "t flapping", "o as 'ah'"], # Simplified representation
        "word_markers": ["gonna", "wanna", "y'all", "awesome", "totally", "like", "literally"],
        "spelling_markers": ["color", "center", "defense", "traveling"],
        "description": "Characterized by rhotic pronunciation (pronouncing 'r' after vowels), flapping of 't' sounds, and specific vocabulary."
    },
    "British": {
        "phonetic_patterns": ["non-rhotic", "t glottal stop", "broader 'a'"], # Simplified representation
        "word_markers": ["proper", "brilliant", "cheers", "mate", "bloody", "quite", "crikey", "rubbish"],
        "spelling_markers": ["colour", "centre", "defence", "travelling"],
        "description": "Typically non-rhotic (dropping 'r' after vowels), with clear 't' pronunciation and distinctive vocabulary."
    },
    "Australian": {
        "phonetic_patterns": ["rising intonation", "i sound change", "non-rhotic"], # Simplified representation
        "word_markers": ["mate", "no worries", "arvo", "reckon", "heaps", "barbie", "esky"],
        "spelling_markers": ["colour", "centre", "defence", "travelling"],
        "description": "Features rising intonation at sentence ends, extended vowels, and unique slang terms."
    },
    "Indian": {
        "phonetic_patterns": ["retroflex consonants", "v/w confusion", "stress timing"], # Simplified representation
        "word_markers": ["actually", "itself", "only", "kindly", "prepone", "co-brother", "doubt"],
        "spelling_markers": ["colour", "centre", "defence", "travelling"], # Indian English often follows British spelling
        "description": "Characterized by retroflex consonants, syllable-timing, and specific Indian English vocabulary."
    },
    "Canadian": {
        "phonetic_patterns": ["canadian raising", "rhotic", "about pronunciation"], # Simplified representation
        "word_markers": ["eh", "sorry", "toque", "washroom", "loonie", "double-double", "chesterfield"],
        "spelling_markers": ["colour", "centre", "defence", "travelling"], # Canadian English often follows British spelling for some words
        "description": "Blends American and British features, with distinctive 'about' pronunciation and unique vocabulary."
    }
}

# Function to download a video from a URL
def download_video(url):
    """Download video from URL to a temporary file"""
    try:
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        # Use a more specific suffix to help systems recognize file type
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.mp4") 
        
        # Handle Loom URLs - This is a heuristic and may break if Loom changes its structure
        if "loom.com" in url:
            # Try to extract video ID from Loom URL
            # This pattern is common but not guaranteed for all Loom share links
            match = re.search(r'/share/([a-f0-9]+)', url)
            if match:
                video_id = match.group(1)
                # Construct a potential direct download URL
                # This is based on observed patterns, not official API
                direct_url = f"https://cdn.loom.com/sessions/thumbnails/{video_id}.mp4"
                st.info(f"Attempting direct download from inferred Loom URL: {direct_url}")
            else:
                 # If pattern doesn't match, try the original URL, might fail
                 direct_url = url
                 st.warning("Loom URL pattern not recognized. Trying original URL directly (may fail).")
        else:
            direct_url = url
        
        # Set a timeout and headers to mimic a browser request
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(direct_url, stream=True, timeout=60, headers=headers) # Added timeout
        
        # Check for successful response
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        # Get content length for progress bar (optional but nice)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024*1024 # 1MB chunks
        progress_placeholder = st.empty()
        bytes_downloaded = 0

        # Save to temporary file
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
                    if total_size_in_bytes > 0:
                         progress = min(1.0, bytes_downloaded / total_size_in_bytes) # Cap progress at 1.0
                         # Update progress bar within the main progress context (handled in main)
        
        # Check if file was actually downloaded
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
             return None, "Download failed: The file was not downloaded correctly or is empty."
             
        return temp_path, None
    
    except requests.exceptions.RequestException as req_err:
        return None, f"Network or Request Error during download: {str(req_err)}"
    except Exception as e:
        return None, f"Error downloading video: {str(e)}"

# Function to extract audio using FFmpeg directly
def extract_audio(video_path):
    """Extract audio from video file using FFmpeg"""
    try:
        # Create a temporary file for audio
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
        
        # FFmpeg command:
        # -i <video_path>: Input file
        # -vn: No video output
        # -acodec pcm_s16le: Audio codec (signed 16-bit little-endian PCM - widely compatible WAV format)
        # -ar 44100: Audio sample rate (44.1 kHz - CD quality)
        # -ac 1: Audio channels (1 for mono, simplifies transcription)
        cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", audio_path, "-y"] # Added -y to overwrite if needed
        
        # Run FFmpeg command
        # Use capture_output=True and text=True for easier error reading
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        # Check if FFmpeg was successful
        if process.returncode != 0:
            # Log the error output from FFmpeg
            st.error(f"FFmpeg command failed: {' '.join(cmd)}")
            st.error(f"FFmpeg STDOUT:\n{process.stdout}")
            st.error(f"FFmpeg STDERR:\n{process.stderr}")
            return None, f"Error extracting audio with FFmpeg: {process.stderr.strip()}"
        
        # Check if file exists and has size > 0
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            return None, "Failed to extract audio: Output audio file is empty or doesn't exist."
        
        return audio_path, None
    
    except FileNotFoundError:
        return None, "FFmpeg command not found. Please ensure FFmpeg is installed and in your system's PATH. On Streamlit Cloud, ensure 'ffmpeg' is listed in packages.txt."
    except Exception as e:
        return None, f"An unexpected error occurred during audio extraction: {str(e)}"

# Function to check if FFmpeg is installed
# Adjusted message to be helpful for cloud deployment
def check_ffmpeg():
    try:
        process = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)
        return process.returncode == 0
    except FileNotFoundError:
        return False
    except Exception:
        return False


# Function to transcribe audio
def transcribe_audio(audio_path):
    """Transcribe audio file using speech recognition"""
    try:
        # Load audio file using pydub for processing
        # Added error handling for loading audio
        try:
            audio = AudioSegment.from_wav(audio_path)
        except Exception as e:
             return None, f"Error loading audio file with pydub: {str(e)}. Ensure it's a valid WAV file."
             
        # Split audio into 30-second chunks (to handle large files and API limits)
        chunk_length_ms = 30 * 1000  # 30 seconds
        chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        full_transcript = ""
        
        # Process each chunk with a progress indicator
        st.info(f"Splitting audio into {len(chunks)} chunks for transcription...")
        
        for i, chunk in enumerate(chunks):
            # Export chunk to temporary file
            chunk_path = os.path.join(tempfile.gettempdir(), f"chunk_{uuid.uuid4()}.wav")
            try:
                chunk.export(chunk_path, format="wav")
            except Exception as e:
                 st.warning(f"Error exporting audio chunk {i+1}: {str(e)}")
                 continue # Skip this chunk
                 
            # Transcribe chunk
            try:
                with sr.AudioFile(chunk_path) as source:
                    # Adjusting energy threshold can help with background noise, though default is often good
                    # recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = recognizer.record(source)
                    
                # Using Google's speech recognition (free tier, might have limits)
                try:
                    # Added a timeout for the API request
                    transcript = recognizer.recognize_google(audio_data, language="en-US", key=None, pfilter=0, show_all=False, timeout=10)
                    full_transcript += " " + transcript
                except sr.UnknownValueError:
                    # API was unable to understand the speech in this chunk
                    st.info(f"Chunk {i+1}: Speech Recognition could not understand audio.")
                    pass # Skip chunk if no speech is detected
                except sr.RequestError as e:
                    # API was unreachable or unresponsive
                    st.error(f"Chunk {i+1}: Could not request results from Google Speech Recognition service; {e}")
                    # Could potentially stop or continue depending on how critical getting all transcript is
                    pass # Continue to next chunk
                except Exception as e:
                    st.warning(f"An unexpected error occurred during transcription of chunk {i+1}: {str(e)}")

            finally:
                 # Ensure temporary chunk file is removed
                 try:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                 except Exception as e:
                     st.warning(f"Could not remove temporary chunk file {chunk_path}: {str(e)}")
            
            # Update progress bar (handled in main)
            
        return full_transcript.strip(), None
    
    except Exception as e:
        return None, f"An unexpected error occurred during audio transcription process: {str(e)}"

# Function to analyze accent
def analyze_accent(transcript):
    """Analyze the accent based on transcript"""
    if not transcript or len(transcript.strip()) < 10: # Check for minimal text
        return None, "Transcript is too short or empty. Not enough speech detected for analysis."
    
    # Lowercased transcript for analysis
    text = transcript.lower()
    
    # Tokenize
    words = word_tokenize(text)
    
    # Track scores for each accent
    # Initialize scores with a small base value to avoid division by zero later if all scores are 0
    scores = {accent: 0.1 for accent in ACCENT_PATTERNS} 
    
    # Analyze word markers
    word_count = Counter(words)
    total_words = len(words)
    unique_words = len(word_count)
    
    # Define weighting factors (can be adjusted)
    WORD_MARKER_WEIGHT = 30
    SPELLING_MARKER_WEIGHT = 10
    PHONETIC_PATTERN_WEIGHT = 20 # Weight for heuristic phonetic checks

    for accent, patterns in ACCENT_PATTERNS.items():
        # Score 1: Word Markers (proportion of total relevant words or unique words)
        marker_count = sum(word_count.get(marker.lower(), 0) for marker in patterns["word_markers"])
        # Use presence and frequency, normalized
        word_marker_score = (marker_count / max(1, unique_words)) * WORD_MARKER_WEIGHT
        scores[accent] += word_marker_score
        
        # Score 2: Spelling Markers (presence in text)
        spelling_matches = sum(1 for marker in patterns["spelling_markers"] if marker.lower() in text)
        scores[accent] += spelling_matches * SPELLING_MARKER_WEIGHT
        
        # Score 3: Heuristic phonetic pattern checks (based on specific word usage)
        # These are very rough approximations based on common associated words
        if accent == "American":
            if bool(re.search(r'\b(gonna|wanna|gotta)\b', text)):
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.8 # High indicator
            if bool(re.search(r'\b(duty|beauty|city)\b', text)): # Words with potential t-flapping
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.5 
            if bool(re.search(r'\b(car|start|park)\b', text)): # Rhotic 'r' words
                scores[accent] += PHONETIC_PATTERN_PATTERN_WEIGHT * 0.6

        elif accent == "British":
            if bool(re.search(r'\b(whilst|amongst|learnt|spelt)\b', text)):
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.8
            if bool(re.search(r'\b(schedule|lieutenant|garage)\b', text)):
                 scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.5
            # Non-rhotic is harder to detect from text alone, but presence of specific linking R examples might hint
            if bool(re.search(r'\b(idea of|far East)\b', text)): # Very weak indicator
                 scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.2

        elif accent == "Australian":
            if bool(re.search(r'\b(arvo|brekkie|footy|ute)\b', text)):
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 1.0 # Strong slang indicators
            if bool(re.search(r'\b(reckon|heaps|mate)\b', text)):
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.6
                
        elif accent == "Indian":
            if bool(re.search(r'\b(itself|only|kindly|actually)\b', text)):
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.7
            if bool(re.search(r'\b(prepone|timepass|batch-mate|fresher|canteen)\b', text)):
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.9 # Specific Indian English words
                
        elif accent == "Canadian":
            if bool(re.search(r'\b(eh|toque|washroom|loonie|double-double)\b', text)):
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 1.0 # Strong slang indicators
            if bool(re.search(r'\b(about|house|out)\b', text)):  # Words with potential Canadian raising - very hard to detect from text, this is weak
                 scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.4

    # Find best match
    best_accent = max(scores, key=scores.get)
    
    # Calculate confidence
    max_score = scores[best_accent]
    total_score = sum(scores.values())
    
    # Normalize scores to percentages relative to the maximum score
    # This gives a sense of how much each accent contributed to the *total* calculated score
    # A more sophisticated approach would use probability models or distances
    if total_score > 0.1: # Check against the small base value
        confidence = (max_score / total_score) * 100
        
        # Apply a gentle sigmoid-like transformation to spread values,
        # making low scores lower and high scores higher relative to the mean.
        # This is still heuristic, not true statistical confidence.
        # The 0.05 and 50 are tuning parameters.
        confidence = 100 / (1 + np.exp(-0.05 * (confidence - 50)))
        
        # Further adjust confidence based on the absolute score magnitude
        # If the total score is very low, the confidence should be low regardless of relative proportion
        # Example: Scale confidence by a factor based on total score vs max possible score (roughly)
        max_possible_score = (WORD_MARKER_WEIGHT * len(ACCENT_PATTERNS["American"]["word_markers"])) + \
                             (SPELLING_MARKER_WEIGHT * len(ACCENT_PATTERNS["American"]["spelling_markers"])) + \
                             PHONETIC_PATTERN_WEIGHT * 3 # Rough max phonetic hits
        
        # Avoid division by zero
        total_score_factor = min(1.0, total_score / max(1, max_possible_score * 0.5)) # Cap factor at 1, scale by half the max possible
        confidence = confidence * total_score_factor

    else:
        confidence = 10 # Very low confidence if no patterns matched significantly

    # Cap confidence at a reasonable maximum for a heuristic method
    confidence = min(round(confidence, 1), 95.0) 
    
    # Get next best accent and its score (for comparison)
    scores_copy = scores.copy()
    scores_copy.pop(best_accent, None) # Use pop with default to avoid error if scores is somehow empty
    second_best = None
    if scores_copy:
        second_best = max(scores_copy, key=scores_copy.get)
        
    # Calculate the ratio between best and second best score as another confidence indicator
    best_score_val = scores[best_accent]
    second_best_score_val = scores_copy.get(second_best, 0.1) # Use default 0.1 if no second best
    score_ratio = best_score_val / second_best_score_val if second_best_score_val > 0 else float('inf')

    # Adjust confidence based on score ratio - A much higher score for the best accent increases confidence
    if score_ratio < 1.5: # If best score is not much higher than second best
         confidence = confidence * 0.7 # Reduce confidence
    elif score_ratio > 3.0: # If best score is significantly higher
         confidence = min(95.0, confidence * 1.1) # Slightly increase confidence (capped)


    result = {
        "accent": best_accent,
        "confidence": confidence,
        "description": ACCENT_PATTERNS[best_accent]["description"],
        "transcript": transcript,
        "detailed_scores": {k: round(v, 2) for k, v in scores.items()},
        "second_best": second_best,
        "score_ratio": round(score_ratio, 2) if score_ratio != float('inf') else 'N/A'
    }
    
    return result, None

# Function to validate URL
def is_valid_url(url):
    """Checks if a string is a potentially valid URL"""
    try:
        result = urlparse(url)
        # Check if scheme and netloc exist. Schemes like http/https are expected.
        # Adding specific scheme check for robustness
        return all([result.scheme in ['http', 'https'], result.netloc])
    except:
        return False

# Main application
def main():
    # Check if FFmpeg is installed
    # This check is helpful locally and can indicate issues on cloud if packages.txt fails
    ffmpeg_installed = check_ffmpeg()
    if not ffmpeg_installed:
        st.error("""
        ⚠️ FFmpeg is not installed or not found in PATH. This tool requires FFmpeg for audio extraction.
        
        **For Streamlit Cloud deployment:** Ensure you have a `packages.txt` file in your repository 
        containing the line `ffmpeg`.
        
        **For local development:** Please install FFmpeg:
        - Windows: Download from ffmpeg.org and add to PATH
        - macOS: `brew install ffmpeg`
        - Linux: `sudo apt install ffmpeg`
        
        See installation guides online for detailed instructions.
        """)
        st.stop()
    
    # Input section
    st.header("1. Input Video")
    
    url = st.text_input("Enter public video URL (MP4, MOV, WEBM, AVI or Loom link):", 
                         help="For example: https://loom.com/share/your-video-id or https://example.com/video.mp4")
    
    # Alternatively allow file upload
    uploaded_file = st.file_uploader("Or upload a video file (.mp4, .mov, .avi, .webm):", type=["mp4", "mov", "avi", "webm"])
    
    # Add demo videos
    st.markdown("""
    **Try these sample links:**
    - American accent: `https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4`
    - British accent: `https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4`
    """)
    
    # Process section
    if st.button("Analyze Accent", type="primary"):
        if not url and not uploaded_file:
            st.error("Please provide either a video URL or upload a video file.")
            return
        
        # Use a main progress bar for overall steps
        main_progress = st.progress(0, text="Starting analysis...")
        
        video_path = None
        audio_path = None

        try:
            # Case 1: URL provided
            if url:
                if not is_valid_url(url):
                    st.error("Please enter a valid URL (must start with http or https).")
                    return
                
                # Step 1: Download video
                main_progress.progress(10, text="Downloading video...")
                video_path, error = download_video(url)
                
                if error:
                    st.error(error)
                    return # Stop execution on error
                main_progress.progress(25, text="Video downloaded. Proceeding to audio extraction...")
                
            # Case 2: File uploaded
            else:
                # Save uploaded file to temp location
                main_progress.progress(10, text="Saving uploaded file...")
                # Ensure file extension is preserved for ffmpeg/pydub
                ext = os.path.splitext(uploaded_file.name)[1]
                video_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext}")
                try:
                    with open(video_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                except Exception as e:
                     st.error(f"Error saving uploaded file: {str(e)}")
                     return
                main_progress.progress(25, text="File saved. Proceeding to audio extraction...")

            # Step 2: Extract audio
            main_progress.progress(35, text="Extracting audio...")
            audio_path, error = extract_audio(video_path)
            
            if error:
                st.error(error)
                return # Stop execution on error
            main_progress.progress(60, text="Audio extracted. Proceeding to transcription...")
            
            # Step 3: Transcribe speech
            main_progress.progress(70, text="Transcribing speech...")
            # The transcribe_audio function now has its own progress info
            transcript, error = transcribe_audio(audio_path)
            
            if error:
                st.error(error)
                return # Stop execution on error
            
            if not transcript or len(transcript.strip()) < 10:
                st.error("Could not detect enough speech in the audio. Please make sure the video contains clear English speech and is longer than a few seconds.")
                return # Stop execution if transcription failed or is too short
                
            main_progress.progress(90, text="Transcription complete. Analyzing accent...")
            
            # Step 4: Analyze accent
            result, error = analyze_accent(transcript)
            
            if error:
                st.error(error)
                return # Stop execution on error
            
            main_progress.progress(100, text="Analysis complete!")
            st.success("✅ Accent Analysis Complete!")
            
            # Display results
            st.header("2. Analysis Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Primary Accent Classification")
                
                primary_accent = result['accent']
                confidence = result['confidence']
                
                st.metric(
                    label=f"{primary_accent} English", 
                    value=f"{confidence}% Confidence"
                )
                
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 5px; background-color: rgba(14, 17, 23, 0.5); margin-top: 10px;">
                    <p style="font-size: 1.1em;"><strong>Characteristics of {primary_accent} English:</strong></p>
                    <p>{result['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if result['second_best'] and result['second_best'] != primary_accent:
                    st.markdown(f"""
                    <div style="margin-top: 20px;">
                        <p><strong>Secondary accent influence detected:</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info(f"Potentially some influence from **{result['second_best']} English** (Ratio of Best to Second Best Score: {result['score_ratio']})")
                else:
                     st.markdown("""
                    <div style="margin-top: 20px;">
                        <p>No significant secondary accent influence detected.</p>
                    </div>
                    """)

            with col2:
                st.subheader("Accent Scores Distribution")
                
                scores = result['detailed_scores']
                
                # Prepare data for chart - filter out very low scores for clarity in visualization? Or show all?
                # Let's show all but highlight the main ones
                scores_df = pd.DataFrame(list(scores.items()), columns=['Accent', 'Score'])
                scores_df = scores_df.sort_values('Score', ascending=False)

                # Create a bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Score', y='Accent', data=scores_df, palette='viridis', ax=ax)
                ax.set_title('Heuristic Score for Each Accent')
                ax.set_xlabel('Score (Arbitrary Units)')
                ax.set_ylabel('Accent Type')
                plt.tight_layout()
                st.pyplot(fig)

                # Display raw scores in tabular format
                st.subheader("Detailed Scores")
                st.dataframe(scores_df, use_container_width=True, hide_index=True)

            # Display transcript
            st.header("3. Transcript")
            st.text_area("Speech Transcript", result['transcript'], height=200)
            
            st.info("""
            **Important Note:** This tool uses rule-based heuristics derived from common vocabulary, spelling, 
            and simplified phonetic patterns identifiable in text transcripts. It does **not** perform 
            acoustic analysis of pronunciation.
            
            The classification and confidence score are **approximations** based on these textual cues
            and should be interpreted with caution. For professional, linguistically accurate accent 
            evaluation, consult with a trained phonetician or linguist.
            """)

        except Exception as e:
            # Catch any unhandled exceptions during the process
            st.error(f"An unexpected error occurred during processing: {str(e)}")
            st.error("Please try again or try a different video.")

        finally:
            # Clean up temporary files regardless of success or failure
            main_progress.empty() # Hide progress bar finally
            st.text("Cleaning up temporary files...")
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    # st.write(f"Removed temp video: {video_path}") # For debugging cleanup
                except Exception as e:
                    st.warning(f"Could not remove temporary video file {video_path}: {str(e)}")
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    # st.write(f"Removed temp audio: {audio_path}") # For debugging cleanup
                except Exception as e:
                    st.warning(f"Could not remove temporary audio file {audio_path}: {str(e)}")
            # Temporary chunk files are attempted to be removed within transcribe_audio

if __name__ == "__main__":
    main()

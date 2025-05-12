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

# --- NLTK Setup - Ensure 'punkt' is downloaded and available ---
# This block runs when the script starts, before the UI is built.
# Streamlit Cloud will cache successful downloads.

NLTK_DATA_RESOURCE = 'tokenizers/punkt'
nltk_data_path = os.path.join(tempfile.gettempdir(), "nltk_data") # Suggest a writable temp dir location

# Add the suggested path to NLTK's search path
# This might help NLTK find data downloaded to the temp directory
nltk.data.path.append(nltk_data_path)

# Check if the resource is already available
try:
    nltk.data.find(NLTK_DATA_RESOURCE)
    st.info(f"NLTK resource '{NLTK_DATA_RESOURCE}' found successfully.")
except LookupError:
    # If not found, attempt to download it
    st.warning(f"NLTK resource '{NLTK_DATA_RESOURCE}' not found. Attempting download of 'punkt' to {nltk_data_path}...")
    try:
        # Download the 'punkt' package to the specified path
        # Remove quiet=True to see download progress/errors in logs
        nltk.download('punkt', download_dir=nltk_data_path) 
        st.success(f"NLTK 'punkt' package download attempted.")

        # --- Crucial Check After Download Attempt ---
        # Verify if the resource is now available
        try:
            nltk.data.find(NLTK_DATA_RESOURCE)
            st.success(f"NLTK resource '{NLTK_DATA_RESOURCE}' verified after download.")
        except LookupError:
            # If it's *still* not found after download, something is wrong.
            st.error(f"""
            Fatal Error: NLTK resource '{NLTK_DATA_RESOURCE}' is still not found after attempting download.
            This indicates a potential issue with NLTK data paths or permissions in the environment.
            NLTK searched paths: {nltk.data.path}
            Please check the Streamlit Cloud logs for download errors or path issues.
            The application cannot proceed without this resource for text tokenization.
            """)
            st.stop() # Stop the application gracefully if the resource is truly missing

    except Exception as e:
        # Catch potential errors during the download process itself
        st.error(f"Error during NLTK 'punkt' download: {e}")
        st.error("NLTK download failed. This is required for text processing.")
        st.stop() # Stop the app if download fails
# --- End NLTK Setup ---


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

**Note:** This analysis is based on **textual patterns** detected in the transcript,
not acoustic analysis of pronunciation. The confidence score is a heuristic approximation.
""")

# Accent feature patterns dictionary
ACCENT_PATTERNS = {
    "American": {
        "phonetic_patterns": ["r after vowels", "t flapping", "o as 'ah'"], # Simplified representation
        "word_markers": ["gonna", "wanna", "y'all", "awesome", "totally", "like", "literally", "dude"],
        "spelling_markers": ["color", "center", "defense", "traveling"],
        "description": "Characterized by rhotic pronunciation (pronouncing 'r' after vowels), flapping of 't' sounds, and specific vocabulary."
    },
    "British": {
        "phonetic_patterns": ["non-rhotic", "t glottal stop", "broader 'a'"], # Simplified representation
        "word_markers": ["proper", "brilliant", "cheers", "mate", "bloody", "quite", "crikey", "rubbish", "innit"],
        "spelling_markers": ["colour", "centre", "defence", "travelling"],
        "description": "Typically non-rhotic (dropping 'r' after vowels), with clear 't' pronunciation and distinctive vocabulary."
    },
    "Australian": {
        "phonetic_patterns": ["rising intonation", "i sound change", "non-rhotic"], # Simplified representation
        "word_markers": ["mate", "no worries", "arvo", "reckon", "heaps", "barbie", "esky", "footy"],
        "spelling_markers": ["colour", "centre", "defence", "travelling"],
        "description": "Features rising intonation at sentence ends, extended vowels, and unique slang terms."
    },
    "Indian": {
        "phonetic_patterns": ["retroflex consonants", "v/w confusion", "stress timing"], # Simplified representation
        "word_markers": ["actually", "itself", "only", "kindly", "prepone", "co-brother", "doubt", "canteen"],
        "spelling_markers": ["colour", "centre", "defence", "travelling"], # Indian English often follows British spelling
        "description": "Characterized by retroflex consonants, syllable-timing, and specific Indian English vocabulary."
    },
    "Canadian": {
        "phonetic_patterns": ["canadian raising", "rhotic", "about pronunciation"], # Simplified representation
        "word_markers": ["eh", "sorry", "toque", "washroom", "loonie", "double-double", "chesterfield", "hoser"],
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
        direct_url = url
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
                 st.warning("Loom URL pattern not recognized. Trying original URL directly (may fail).")

        # Set a timeout and headers to mimic a browser request
        headers = {'User-Agent': 'Mozilla/5.0'}
        # Added timeout for the request itself
        response = requests.get(direct_url, stream=True, timeout=30, headers=headers)

        # Check for successful response
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        # Save to temporary file
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                if chunk:
                    f.write(chunk)

        # Check if file was actually downloaded
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
             return None, "Download failed: The file was not downloaded correctly or is empty."

        return temp_path, None

    except requests.exceptions.Timeout:
        return None, "Download timed out. The URL was too slow to respond."
    except requests.exceptions.RequestException as req_err:
        return None, f"Network or Request Error during download: {str(req_err)}. Please check the URL is publicly accessible."
    except Exception as e:
        return None, f"An unexpected error occurred during video download: {str(e)}"

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
        # -ar 16000: Audio sample rate (16 kHz is sufficient for speech recognition and smaller)
        # -ac 1: Audio channels (1 for mono, simplifies transcription)
        # -y: Overwrite output file without asking
        # -loglevel error: Suppress verbose FFmpeg output, show only errors
        cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y", "-loglevel", "error"]

        # Run FFmpeg command
        # Use capture_output=True and text=True for easier error reading
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # Check if FFmpeg was successful
        if process.returncode != 0:
            # Log the error output from FFmpeg
            st.error(f"FFmpeg command failed with error code {process.returncode}.")
            st.error(f"FFmpeg STDERR:\n{process.stderr}")
            return None, f"Error extracting audio with FFmpeg. FFmpeg Output:\n{process.stderr.strip()}"

        # Check if file exists and has size > 0
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
             # Try to provide more context if stderr was empty but file wasn't created
             error_msg = f"Failed to extract audio: Output audio file ({audio_path}) is empty or doesn't exist."
             if process.stderr:
                 error_msg += f"\nFFmpeg stderr: {process.stderr.strip()}"
             return None, error_msg

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
             return None, f"Error loading audio file with pydub: {str(e)}. Ensure the extracted file is a valid WAV."

        # Split audio into 30-second chunks (to handle large files and API limits)
        chunk_length_ms = 30 * 1000  # 30 seconds
        chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        # Initialize recognizer
        recognizer = sr.Recognizer()
        full_transcript = ""

        st.info(f"Splitting audio into {len(chunks)} chunks for transcription...")

        chunk_progress_bar = st.progress(0) # Progress bar for chunks

        for i, chunk in enumerate(chunks):
            # Update progress bar for current chunk
            chunk_progress = (i + 1) / len(chunks)
            chunk_progress_bar.progress(chunk_progress)

            # Export chunk to temporary file
            # Use a unique filename for each chunk attempt
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
                    # recognizer.adjust_for_ambient_noise(source, duration=0.5) # You could uncomment this line
                    audio_data = recognizer.record(source)

                # Using Google's speech recognition (free tier, might have limits)
                try:
                    # Removed timeout argument as it caused an error in some versions
                    transcript = recognizer.recognize_google(audio_data, language="en-US", key=None, pfilter=0, show_all=False)
                    full_transcript += " " + transcript
                except sr.UnknownValueError:
                    # API was unable to understand the speech in this chunk
                    # st.info(f"Chunk {i+1}: Speech Recognition could not understand audio.") # Optional: keep this for debug
                    pass # Skip chunk if no speech is detected
                except sr.RequestError as e:
                    # API was unreachable or unresponsive
                    st.error(f"Chunk {i+1}: Could not request results from Google Speech Recognition service; {e}")
                    # Could potentially stop or continue depending on how critical getting all transcript is
                    pass # Continue to next chunk
                except Exception as e:
                    # Catch any other unexpected errors during recognition
                    st.warning(f"An unexpected error occurred during transcription of chunk {i+1}: {str(e)}")

            finally:
                 # Ensure temporary chunk file is removed
                 try:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                 except Exception as e:
                     st.warning(f"Could not remove temporary chunk file {chunk_path}: {str(e)}")

        chunk_progress_bar.empty() # Hide chunk progress bar

        return full_transcript.strip(), None

    except Exception as e:
        # Catch any errors that happen outside the chunk loop (e.g., loading audio)
        return None, f"An unexpected error occurred during the audio transcription process: {str(e)}"

# Function to analyze accent
def analyze_accent(transcript):
    """Analyze the accent based on transcript"""
    if not transcript or len(transcript.strip()) < 20: # Check for minimal text length
        return None, "Transcript is too short or empty. Not enough clear speech detected for analysis."

    # Lowercased transcript for analysis
    text = transcript.lower()

    # Tokenize using NLTK's word_tokenize
    # This function requires the 'punkt' tokenizer data.
    # The NLTK setup block at the top should ensure this data is available.
    try:
        words = word_tokenize(text)
    except LookupError as e:
         # This block serves as a final fail-safe. The initial setup should prevent reaching here.
         return None, f"NLTK 'punkt' tokenizer data is not available during tokenization: {e}. Please check NLTK setup at app start."
    except Exception as e:
         return None, f"An unexpected error occurred during text tokenization: {str(e)}"

    # Track scores for each accent
    # Initialize scores with a small base value to avoid division by zero later if all scores are 0
    scores = {accent: 0.1 for accent in ACCENT_PATTERNS}

    # Analyze word markers
    word_count = Counter(words)
    # total_words = len(words) # Not directly used in scoring logic currently
    unique_words = len(word_count)

    # Define weighting factors (can be adjusted)
    WORD_MARKER_WEIGHT = 30
    SPELLING_MARKER_WEIGHT = 10
    PHONETIC_PATTERN_WEIGHT = 20 # Weight for heuristic phonetic checks

    for accent, patterns in ACCENT_PATTERNS.items():
        # Score 1: Word Markers (proportion of total relevant words or unique words)
        marker_count = sum(word_count.get(marker.lower(), 0) for marker in patterns["word_markers"])
        # Use presence and frequency, normalized by unique words
        word_marker_score = (marker_count / max(1, unique_words)) * WORD_MARKER_WEIGHT
        scores[accent] += word_marker_score

        # Score 2: Spelling Markers (presence in text)
        spelling_matches = sum(1 for marker in patterns["spelling_markers"] if marker.lower() in text)
        scores[accent] += spelling_matches * SPELLING_MARKER_WEIGHT

        # Score 3: Heuristic phonetic pattern checks (based on specific word usage)
        # These are very rough approximations based on common associated words or spelling variations
        if accent == "American":
            if bool(re.search(r'\b(gonna|wanna|gotta)\b', text)):
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.8 # High indicator
            if bool(re.search(r'\b(duty|beauty|city)\b', text)): # Words with potential t-flapping
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.5
            if bool(re.search(r'\b(car|start|park|water)\b', text)): # Rhotic 'r' words or related patterns
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.6

        elif accent == "British":
            if bool(re.search(r'\b(whilst|amongst|learnt|spelt)\b', text)):
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.8
            if bool(re.search(r'\b(schedule|lieutenant|garage)\b', text)): # Words pronounced differently
                 scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.5
            # Non-rhotic is harder to detect from text alone, but presence of specific linking R examples might hint
            # Example: "idea of" might be transcribed if linking R is used, but this is highly unreliable
            # if bool(re.search(r'\b(idea of|far East)\b', text)): # Very weak indicator, perhaps remove or reduce weight
            #      scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.1

        elif accent == "Australian":
            if bool(re.search(r'\b(arvo|brekkie|footy|ute|mozzie)\b', text)):
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 1.0 # Strong slang indicators
            if bool(re.search(r'\b(reckon|heaps|mate|yeah nah)\b', text)):
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.6

        elif accent == "Indian":
            if bool(re.search(r'\b(itself|only|kindly|actually)\b', text)):
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.7
            if bool(re.search(r'\b(prepone|timepass|batch-mate|fresher|canteen|do the needful)\b', text)):
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.9 # Specific Indian English words

        elif accent == "Canadian":
            if bool(re.search(r'\b(eh|toque|washroom|loonie|double-double|chesterfield)\b', text)):
                scores[accent] += PHONETIC_PATTERN_WEIGHT * 1.0 # Strong slang indicators
            if bool(re.search(r'\b(about|house|out)\b', text)):  # Words with potential Canadian raising - very hard to detect from text, this is weak
                 scores[accent] += PHONETIC_PATTERN_WEIGHT * 0.4

    # Find best match
    best_accent = max(scores, key=scores.get)

    # Calculate confidence
    max_score = scores[best_accent]
    total_score = sum(scores.values())

    # Normalize scores to percentages relative to the sum of scores
    confidence = 0.0
    if total_score > 0.1: # Check against the small base value
        raw_percentage = (max_score / total_score) * 100

        # Apply a gentle sigmoid-like transformation
        transformed_confidence = 100 / (1 + np.exp(-0.05 * (raw_percentage - 50)))

        # Scale confidence based on whether the total score is significant
        total_score_threshold = 10 # Adjust threshold based on typical scores
        total_score_scaling = min(1.0, total_score / total_score_threshold) # Scales from 0 to 1 up to the threshold

        confidence = transformed_confidence * total_score_scaling


    # Cap confidence at a reasonable maximum for a heuristic method
    confidence = min(round(confidence, 1), 95.0)

    # Get next best accent and its score (for comparison)
    scores_copy = scores.copy()
    scores_copy.pop(best_accent, None) # Use pop with default to avoid error if scores is somehow empty
    second_best = None
    score_ratio = 'N/A'
    if scores_copy:
        # Ensure there's a second best with a score above the base value
        filtered_scores_copy = {k: v for k, v in scores_copy.items() if v > 0.1}
        if filtered_scores_copy:
            second_best = max(filtered_scores_copy, key=filtered_scores_copy.get)
            best_score_val = scores[best_accent]
            second_best_score_val = filtered_scores_copy[second_best] # Get score from filtered dict

            if second_best_score_val > 0:
                 score_ratio = round(best_score_val / second_best_score_val, 2)
            else:
                 score_ratio = 'Inf' # Should not happen with base score > 0, but defensive

            # Adjust confidence based on score ratio - A much higher score for the best accent increases confidence
            if isinstance(score_ratio, (int, float)): # Check if score_ratio is a number
                if score_ratio < 1.5: # If best score is not much higher than second best
                     confidence = confidence * 0.85 # Reduce confidence slightly
                elif score_ratio > 2.5: # If best score is significantly higher
                     confidence = min(95.0, confidence * 1.1) # Slightly increase confidence (capped)
                confidence = min(round(confidence, 1), 95.0) # Re-cap

    result = {
        "accent": best_accent,
        "confidence": confidence,
        "description": ACCENT_PATTERNS[best_accent]["description"],
        "transcript": transcript,
        "detailed_scores": {k: round(v, 2) for k, v in scores.items()},
        "second_best": second_best,
        "score_ratio": score_ratio
    }

    return result, None

# Function to validate URL
def is_valid_url(url):
    """Checks if a string is a potentially valid URL"""
    try:
        result = urlparse(url)
        # Check if scheme (http/https) and network location exist.
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
        ⚠️ FFmpeg command not found. This tool requires FFmpeg for audio extraction.

        **For Streamlit Cloud deployment:** Ensure you have a `packages.txt` file in your repository
        containing the line `ffmpeg`.

        **For local development:** Please install FFmpeg and ensure it's in your system's PATH.
        (e.g., `brew install ffmpeg` on macOS, `sudo apt update && sudo apt install ffmpeg` on Debian/Ubuntu Linux).
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
                ext = os.path.splitext(uploaded_file.name)[1].lower() # Get lower case extension
                if ext not in [".mp4", ".mov", ".avi", ".webm"]:
                     st.error(f"Unsupported file type: {ext}. Please upload .mp4, .mov, .avi, or .webm.")
                     return
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
            # Passing video_path acquired from either download or upload
            audio_path, error = extract_audio(video_path)

            if error:
                st.error(error)
                return # Stop execution on error
            main_progress.progress(60, text="Audio extracted. Proceeding to transcription...")

            # Step 3: Transcribe speech
            main_progress.progress(70, text="Transcribing speech...")
            # The transcribe_audio function now has its own progress info within it
            transcript, error = transcribe_audio(audio_path)

            if error:
                st.error(error)
                return # Stop execution on error

            if not transcript or len(transcript.strip()) < 20: # Check minimum length again after transcription
                st.error("Could not detect enough clear English speech in the video. Please make sure the video contains clear speech longer than ~10 seconds.")
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

                # Use a colored box or marker based on confidence
                if confidence > 75:
                    confidence_color = "green"
                elif confidence > 50:
                    confidence_color = "orange"
                else:
                    confidence_color = "red"

                st.markdown(f"""
                <div style="padding: 15px; border-radius: 5px; border: 2px solid {confidence_color}; background-color: rgba(14, 17, 23, 0.5);">
                    <h3 style="color: {confidence_color}; margin-top: 0;">{primary_accent} English</h3>
                    <p style="font-size: 1.2em;"><strong>Confidence:</strong> {confidence}%</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="padding: 15px; border-radius: 5px; background-color: rgba(14, 17, 23, 0.5); margin-top: 20px;">
                    <p style="font-size: 1.1em; margin-top: 0;"><strong>Characteristics of {primary_accent} English:</strong></p>
                    <p>{result['description']}</p>
                </div>
                """, unsafe_allow_html=True)

                if result['second_best'] and result['second_best'] != primary_accent:
                    st.markdown(f"""
                    <div style="margin-top: 20px;">
                        <p><strong>Potential Secondary Accent Influence:</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info(f"Some characteristics suggest **{result['second_best']} English** (Ratio of Best to Second Best Score: {result['score_ratio']})")
                else:
                     st.markdown("""
                    <div style="margin-top: 20px;">
                        <p>No significant secondary accent influence detected.</p>
                    </div>
                    """)

            with col2:
                st.subheader("Accent Scores Distribution")

                scores = result['detailed_scores']

                # Prepare data for chart
                scores_df = pd.DataFrame(list(scores.items()), columns=['Accent', 'Score'])
                # Sort by score for the bar chart
                scores_df = scores_df.sort_values('Score', ascending=False)

                # Create a bar chart using Streamlit's built-in chart for interactivity
                st.bar_chart(
                    scores_df.set_index('Accent'),
                    use_container_width=True,
                    color="#1f77b4" # A standard blue color
                )

                # Display raw scores in tabular format
                st.subheader("Detailed Heuristic Scores")
                # Add a note about what the scores mean
                st.markdown("<small>Scores are arbitrary values based on detected patterns, not a true statistical measure.</small>", unsafe_allow_html=True)
                st.dataframe(scores_df, use_container_width=True, hide_index=True)


            # Display transcript
            st.header("3. Transcript")
            if result['transcript']:
                 st.text_area("Speech Transcript", result['transcript'], height=200)
            else:
                 st.warning("No transcript could be generated.")

            st.info("""
            **Important Note:** This tool uses rule-based heuristics derived from common vocabulary, spelling,
            and simplified phonetic patterns identifiable in text transcripts. It does **not** perform
            acoustic analysis of pronunciation (e.g., analyzing sounds like vowels, consonants, or intonation patterns directly from the audio waveform).

            The classification and confidence score are **approximations** based on these textual cues
            and should be interpreted with caution. The output reflects which accent's textual indicators
            were most present in the *transcription*. For professional, linguistically accurate accent
            evaluation, consult with a trained phonetician or linguist.
            """)

        except Exception as e:
            # Catch any unhandled exceptions during the process
            st.error(f"An unexpected error occurred during processing: {str(e)}")
            st.error("Please try again or try a different video.")

        finally:
            # Clean up temporary files regardless of success or failure
            main_progress.empty() # Hide main progress bar finally
            st.text("Cleaning up temporary files...")
            cleanup_success = True

            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception as e:
                    st.warning(f"Could not remove temporary video file {video_path}: {str(e)}")
                    cleanup_success = False
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    st.warning(f"Could not remove temporary audio file {audio_path}: {str(e)}")
                    cleanup_success = False

            # Temporary chunk files are attempted to be removed within transcribe_audio
            # A final check/cleanup for any lingering chunk files could be added here if needed,
            # but it might be overkill and can sometimes cause issues if files are still being accessed.

            if cleanup_success:
                 st.text("Temporary files cleaned up.")
            else:
                 st.text("Cleanup finished with some warnings.")


if __name__ == "__main__":
    main()

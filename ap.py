import streamlit as st
import os
import tempfile
import uuid
import re
import numpy as np
import requests
import time
import subprocess
from pydub import AudioSegment
import speech_recognition as sr
from urllib.parse import urlparse
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import altair as alt

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
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
(e.g., a direct MP4 file or Loom video), and the system will:
1. Extract the audio
2. Transcribe the speech
3. Analyze the accent based on the transcript
4. Provide a classification with a confidence score
""")

# Accent feature patterns dictionary
ACCENT_PATTERNS = {
    "American": {
        "phonetic_patterns_keywords": ["gonna", "wanna", "gotta", "movie", "duty", "beauty"], # Approximated keywords
        "word_markers": ["gonna", "wanna", "y'all", "awesome", "totally", "like", "period", "vacation", "apartment", "elevator"],
        "spelling_markers": ["color", "center", "defense", "analyze"], # Less reliable from transcript, but kept for context
        "description": "Characterized by rhotic pronunciation (pronouncing 'r' after vowels), flapping of 't' sounds, and specific vocabulary.",
        "weight": 1.5 # Give slightly higher weight to prominent features
    },
    "British": {
        "phonetic_patterns_keywords": ["whilst", "amongst", "learnt", "spelt", "schedule", "lieutenant", "garage"], # Approximated keywords
        "word_markers": ["proper", "brilliant", "cheers", "mate", "bloody", "quite", "whilst", "amongst", "learnt", "spelt"],
        "spelling_markers": ["colour", "centre", "defence", "analyse"],
        "description": "Typically non-rhotic (dropping 'r' after vowels), with clear 't' pronunciation and distinctive vocabulary.",
         "weight": 1.3
    },
    "Australian": {
        "phonetic_patterns_keywords": ["arvo", "brekkie", "footy", "ute", "reckon", "heaps", "mate"], # Approximated keywords
        "word_markers": ["mate", "no worries", "arvo", "reckon", "heaps", "footy", "brekkie", "ute"],
        "spelling_markers": ["colour", "centre", "defence"], # Less relevant for Aus, often uses British spelling
        "description": "Features rising intonation at sentence ends (not detectable here), extended vowels, and unique slang terms.",
        "weight": 1.2
    },
    "Indian": {
        "phonetic_patterns_keywords": ["actually", "itself", "only", "kindly", "prepone", "timepass", "batch-mate"], # Approximated keywords
        "word_markers": ["actually", "itself", "only", "kindly", "prepone", "timepass", "batch-mate", "doubt", "frock"], # Specific Indian English words/usages
        "spelling_markers": ["colour", "centre", "defence"], # Often uses British spelling
        "description": "Characterized by retroflex consonants, syllable-timing, and specific Indian English vocabulary (features not fully detectable from text alone).",
        "weight": 1.4 # Indian English often has distinct vocabulary patterns
    },
    "Canadian": {
        "phonetic_patterns_keywords": ["eh", "about", "house", "out"], # Approximated keywords (for Canadian Raising)
        "word_markers": ["eh", "sorry", "toque", "washroom", "loonie", "double-double", "klick"],
        "spelling_markers": ["colour", "centre", "defence"], # Often uses British spelling
        "description": "Blends American and British features, with distinctive 'about' pronunciation and unique vocabulary.",
        "weight": 1.1
    }
}

# Function to download a video from a URL
def download_video(url):
    """Download video from URL to a temporary file"""
    try:
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        # Use a more robust extension check or default to mp4 if unsure
        parsed_url = urlparse(url)
        path = parsed_url.path
        ext = os.path.splitext(path)[1].lower()
        if not ext or ext not in ['.mp4', '.mov', '.avi', '.webm']:
             ext = '.mp4' # Default if extension is missing or unknown
             
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
        
        # Handle Loom URLs - This is approximate and may need adjustment
        if "loom.com" in url:
            st.warning("Loom URL detected. Attempting to find a direct download link. This might not always work.")
            # Basic attempt: assume common pattern, might fail for private videos or future Loom changes
            match = re.search(r'/share/([a-f0-9]+)', url)
            if match:
                 video_id = match.group(1)
                 # This URL pattern is known to change or be restricted
                 direct_url = f"https://cdn.loom.com/sessions/thumbnails/{video_id}.mp4"
                 st.info(f"Attempting download from: {direct_url}")
            else:
                 return None, "Could not extract Loom video ID from URL format."
        else:
            direct_url = url
        
        # Download the video
        response = requests.get(direct_url, stream=True)
        if response.status_code == 403:
             return None, "Access forbidden (403). The video might be private or require login."
        if response.status_code != 200:
            return None, f"Failed to download video: HTTP Error {response.status_code}. Check if the URL is a direct link to a public video file."
        
        # Save to temporary file
        file_size = int(response.headers.get('Content-Length', 0))
        downloaded_size = 0
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): # Use a reasonable chunk size
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # Optional: Update progress bar during download if needed, but might make UI jumpy
                    # download_progress = downloaded_size / max(1, file_size)
                    # st.progress(download_progress)
        
        if os.path.getsize(temp_path) == 0:
             return None, "Downloaded file is empty. The URL might not point to a valid video file."
             
        return temp_path, None
    
    except requests.exceptions.RequestException as e:
         return None, f"Network or Request Error downloading video: {str(e)}. Check your internet connection or if the URL is accessible."
    except Exception as e:
        return None, f"Unexpected Error downloading video: {str(e)}"

# Function to extract audio using FFmpeg directly
def extract_audio(video_path):
    """Extract audio from video file using FFmpeg"""
    try:
        # Create a temporary file for audio
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
        
        # FFmpeg command to extract audio to WAV (16-bit PCM, 44.1kHz, stereo)
        # Using -y to overwrite if file exists (unlikely with uuid but safe)
        # Adding -nostdin to prevent interactive prompts
        # Adding -loglevel error to reduce noise, only show errors
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", "-nostdin", "-loglevel", "error", audio_path]
        
        # Run FFmpeg command
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Check if FFmpeg was successful
        if process.returncode != 0:
            return None, f"Error extracting audio with FFmpeg (code {process.returncode}): {stderr.decode()}"
        
        # Check if file exists and has size > 0
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
             # If stderr is empty but file is missing/empty, maybe input was bad
            if not stderr: stderr = b"FFmpeg ran, but output file is missing or empty."
            return None, f"Failed to extract audio: Output file is empty or doesn't exist. FFmpeg output: {stderr.decode()}"
        
        return audio_path, None
    
    except FileNotFoundError:
         return None, "FFmpeg command not found. Please ensure FFmpeg is installed and in your system's PATH."
    except Exception as e:
        return None, f"Error extracting audio: {str(e)}"

# Function to check if FFmpeg is installed
def check_ffmpeg():
    """Checks if ffmpeg command is available in the system's PATH."""
    try:
        # Use 'which ffmpeg' or 'where ffmpeg' to check if it's in PATH
        # This is more reliable than 'ffmpeg -version' which might return non-zero for other reasons
        if os.name == 'nt': # Windows
            subprocess.check_call(["where", "ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else: # Linux/macOS
            subprocess.check_call(["which", "ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    except Exception as e:
         st.error(f"An unexpected error occurred while checking FFmpeg: {e}")
         return False


# Function to transcribe audio
def transcribe_audio(audio_path):
    """Transcribe audio file using speech recognition"""
    try:
        # Load audio file using pydub for processing
        audio = AudioSegment.from_wav(audio_path)

        # Increase silence threshold and decrease pause threshold slightly
        # This might help with noisy audio or varying speech patterns
        # Parameters below are examples and might need tuning
        min_silence_len = 500 # milliseconds
        silence_thresh = audio.dBFS - 16 # dB below full scale
        
        # Use silence to split audio into non-silent chunks
        # This helps recognizer focus on speech segments
        speech_chunks = split_on_silence(audio, 
                                         min_silence_len=min_silence_len,
                                         silence_thresh=silence_thresh,
                                         keep_silence=200 # Keep a little silence around speech
                                        )

        # Initialize recognizer
        recognizer = sr.Recognizer()
        full_transcript = ""
        
        if not speech_chunks:
             return "", "No significant speech detected in the audio."

        # Process each speech chunk
        st.info(f"Found {len(speech_chunks)} speech segments. Transcribing...")
        
        # Use a progress bar for transcription chunks
        transcript_progress = st.progress(0)

        for i, chunk in enumerate(speech_chunks):
            # Export chunk to temporary file
            chunk_path = os.path.join(tempfile.gettempdir(), f"chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            
            # Transcribe chunk
            try:
                with sr.AudioFile(chunk_path) as source:
                    # Adjust for ambient noise before processing each chunk
                    recognizer.adjust_for_ambient_noise(source, duration=0.5) # Listen for 0.5 sec to calibrate energy threshold
                    audio_data = recognizer.record(source)
                    
                    # Using Google's speech recognition
                    # Add language hint for English
                    transcript = recognizer.recognize_google(audio_data, language="en-US") # Can specify "en-GB", "en-AU" etc. but en-US is often broadest
                    full_transcript += " " + transcript
                    
                # Update progress
                transcript_progress.progress((i + 1) / len(speech_chunks))

            except sr.UnknownValueError:
                # API was unable to understand speech
                # print(f"Google Speech Recognition could not understand audio in chunk {i+1}")
                pass # Skip this chunk if unclear
            except sr.RequestError as e:
                # API was unreachable or unresponsive
                st.warning(f"Could not request results from Google Speech Recognition service for chunk {i+1}; {e}")
                pass # Try next chunk
            except Exception as e:
                st.warning(f"Error processing audio chunk {i+1}: {str(e)}")
                pass # Try next chunk
            finally:
                # Remove temporary chunk file
                try:
                    os.remove(chunk_path)
                except OSError:
                    pass # File might not have been created if previous steps failed
        
        transcript_progress.empty() # Remove progress bar after completion

        return full_transcript.strip(), None
    
    except FileNotFoundError:
        return None, "Could not find the audio file for transcription."
    except Exception as e:
        return None, f"Error transcribing audio: {str(e)}"

# Import split_on_silence - it's part of pydub.silence
from pydub.silence import split_on_silence


# Function to analyze accent
def analyze_accent(transcript):
    """Analyze the accent based on transcript"""
    if not transcript or len(transcript.split()) < 20: # Require a minimum number of words
        return None, "Transcript is too short or empty for analysis. Need at least 20 words."
    
    # Lowercased transcript for analysis
    text = transcript.lower()
    
    # Tokenize
    words = word_tokenize(text)
    word_count = Counter(words)
    total_words = len(words)
    
    # Track scores for each accent
    # Initialize with a base score to avoid zero scores if no patterns match
    base_score = 1.0 # A small base score
    scores = {accent: base_score for accent in ACCENT_PATTERNS}
    
    # Analyze patterns and contribute to scores
    for accent, patterns in ACCENT_PATTERNS.items():
        accent_score = 0.0
        weight = patterns.get("weight", 1.0) # Get accent-specific weight

        # 1. Word Markers
        marker_count = sum(word_count.get(marker.lower(), 0) for marker in patterns["word_markers"])
        # Score contribution based on frequency, capped to avoid single word dominating
        # Scale by weight
        accent_score += min(marker_count, 10) * 5.0 * weight # Max 10 occurrences contribute significantly

        # 2. Phonetic Patterns (approximated via keywords)
        # Check for keyword presence
        phonetic_keyword_count = sum(1 for keyword in patterns["phonetic_patterns_keywords"] if keyword.lower() in text)
        # Score contribution based on number of *different* keywords found
        accent_score += phonetic_keyword_count * 8.0 * weight # Each keyword gives a boost

        # 3. Spelling Markers (less reliable from transcript)
        # Check for spelling marker presence
        spelling_match_count = sum(1 for marker in patterns["spelling_markers"] if marker.lower() in text)
        accent_score += spelling_match_count * 3.0 # Lower weight for spelling from transcript

        # Add the calculated score to the base score
        scores[accent] += accent_score
        
    # Find best match
    best_accent = max(scores, key=scores.get)
    
    # Calculate confidence
    max_score = scores[best_accent]
    total_score = sum(scores.values())
    
    # Normalize confidence using softmax-like approach or simpler ratio
    # Using a ratio relative to the sum of scores provides a comparative measure
    if total_score > len(ACCENT_PATTERNS) * base_score: # Ensure total score is more than just base scores
         # Calculate raw ratio
         raw_confidence = (max_score / total_score) * 100
         
         # Simple mapping to spread confidence - adjust parameters as needed
         # For example, map 20% ratio to 30% confidence, 40% to 50%, 60% to 70%, 80% to 90%
         # This is heuristic and would ideally be based on model training
         # A linear interpolation based on range might work:
         # conf = lower_bound + (raw_ratio - min_ratio) * (upper_bound - lower_bound) / (max_ratio - min_ratio)
         # Let's use a sigmoid-like function scaled to the 30-95 range for typical outputs
         # A simple scaling: map raw_confidence (e.g., 20-80%) to target range (e.g., 30-95%)
         
         # Let's try a simpler approach: scale score relative to the max score among others
         # If the best score is much higher than the second best, confidence is high
         scores_copy = sorted(scores.values(), reverse=True)
         if len(scores_copy) > 1:
              second_best_score = scores_copy[1]
              score_difference = max_score - second_best_score
              # Normalize difference relative to total score
              normalized_difference = score_difference / max_score if max_score > 0 else 0
              
              # Heuristic mapping of normalized difference to confidence
              # Small difference -> lower confidence (e.g., 30-50)
              # Large difference -> higher confidence (e.g., 70-95)
              confidence = 30 + normalized_difference * 65 # Maps 0 diff to 30, 1 diff to 95
              confidence = max(30, min(95, confidence)) # Cap between 30 and 95
         else:
              # Only one accent possible (unlikely with current setup), high confidence? Or low? Low seems safer.
              confidence = 30 

         # Ensure confidence doesn't go below a base level if some features were found
         confidence = max(30, confidence)

    else:
        confidence = 30  # Default when not enough indicators found (only base scores)
    
    # Get next best accent
    scores_copy = scores.copy()
    scores_copy.pop(best_accent)
    second_best = max(scores_copy, key=scores_copy.get) if scores_copy else None
    
    result = {
        "accent": best_accent,
        "confidence": round(confidence, 1),
        "description": ACCENT_PATTERNS[best_accent]["description"],
        "transcript": transcript,
        "detailed_scores": {k: round(v, 2) for k, v in scores.items()}, # Use rounded scores
        "second_best": second_best
    }
    
    return result, None

# Function to validate URL
def is_valid_url(url):
    try:
        result = urlparse(url)
        # Check for scheme and netloc (hostname)
        # Also check if it looks like a file extension commonly used for videos
        valid_extensions = ['.mp4', '.mov', '.avi', '.webm', '.mkv']
        is_video_url = any(url.lower().endswith(ext) for ext in valid_extensions) or "loom.com" in url.lower()
        
        return all([result.scheme in ['http', 'https'], result.netloc]) and (is_video_url or not os.path.splitext(result.path)[1]) # Allow URLs without obvious extensions too

    except:
        return False

# Main application
def main():
    # Check if FFmpeg is installed
    ffmpeg_installed = check_ffmpeg()
    if not ffmpeg_installed:
        st.error("""
        ⚠️ **FFmpeg is not installed or not found in your system's PATH.**
        
        This tool requires FFmpeg for audio extraction. Please install it:
        - **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add its `bin` directory to your system's PATH.
        - **macOS (using Homebrew):** Open Terminal and run `brew install ffmpeg`.
        - **Linux (Debian/Ubuntu):** Open Terminal and run `sudo apt update && sudo apt install ffmpeg`.
        
        After installation, you might need to restart your terminal or computer for the changes to take effect.
        """)
        st.stop()
    
    # Input section
    st.header("1. Input Video")
    
    url = st.text_input("Enter public video URL (MP4, MOV, etc., or Loom link):", 
                         help="For example: `https://loom.com/share/your-video-id` or `https://example.com/video.mp4`")
    
    # Add demo videos
    st.markdown("""
    **Try these sample links:**
    - **American accent:** `https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4`
    - **British accent:** `https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4`
    
    *Note: The accuracy depends heavily on the quality and length of speech.*
    """)
    
    st.markdown("---") # Separator
    
    # Allow file upload (less reliable due to server limitations, but included)
    # st.header("Or Upload Video File (experimental)") # Renamed to make URL primary
    # st.warning("File upload is experimental and may fail for large files due to server limitations.")
    # uploaded_file = st.file_uploader("Upload a video file:", type=["mp4", "mov", "avi", "webm"])
    # Disabling upload for simplicity and focus on URL method's requirements
    uploaded_file = None # Ensure it's None if not used

    # Process section
    if st.button("Analyze Accent", type="primary"):
        if not url and not uploaded_file:
            st.error("Please provide either a video URL.")
            return
        
        # Clear previous results if any
        st.empty() # This won't actually clear prior elements, but subsequent writes will overwrite
        
        # Use placeholders for dynamic updates
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        video_path = None
        audio_path = None

        try:
            # Case 1: URL provided
            if url:
                if not is_valid_url(url):
                    st.error("Please enter a valid URL that points to a public video file or a recognized platform link (like Loom).")
                    return
                
                # Step 1: Download video
                status_text.info("1/4 - Downloading video...")
                video_path, error = download_video(url)
                progress_bar.progress(25)
                
                if error:
                    st.error(error)
                    return
                
            # Case 2: File uploaded (currently disabled)
            # else:
            #     status_text.info("1/4 - Saving uploaded file...")
            #     video_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp4") # Assuming mp4 for upload
            #     with open(video_path, "wb") as f:
            #         f.write(uploaded_file.getbuffer())
            #     progress_bar.progress(25)
            
            # Step 2: Extract audio
            status_text.info("2/4 - Extracting audio...")
            audio_path, error = extract_audio(video_path)
            progress_bar.progress(50)
            
            if error:
                st.error(error)
                return
            
            # Step 3: Transcribe speech
            status_text.info("3/4 - Transcribing speech...")
            # Transcription progress is handled internally by transcribe_audio
            transcript, error = transcribe_audio(audio_path)
            progress_bar.progress(75) # Main progress updates after transcription is done
            
            if error:
                st.error(error)
                return
            
            if not transcript or len(transcript.split()) < 20:
                st.error("Could not detect enough clear English speech in the video (need at least 20 words). Please try a video with longer, clearer speech.")
                return
            
            # Step 4: Analyze accent
            status_text.info("4/4 - Analyzing accent...")
            result, error = analyze_accent(transcript)
            progress_bar.progress(100)
            
            if error:
                st.error(error)
                return
            
            status_text.empty() # Clear final status
            progress_bar.empty() # Clear progress bar
            
            # Display results
            st.header("2. Analysis Results")
            
            # Summary box
            st.success(f"✅ **Analysis Complete!**")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Primary Accent Classification")
                # Use markdown for emphasis
                st.markdown(f"""
                Based on the analysis, the primary accent detected is:
                ## <span style="color:green;">{result['accent']} English</span>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                **Confidence Score:** **{result['confidence']}%**
                
                **Accent Description:** {result['description']}
                """)
                
                if result['second_best']:
                    st.markdown(f"""
                    *Note: Some characteristics of **{result['second_best']} English** were also observed.*
                    """)
            
            with col2:
                st.subheader("Detailed Accent Scores")
                st.markdown("""
                These scores represent the relative strength of evidence found in the transcript for patterns associated with each accent.
                Higher scores indicate more matching features.
                """)
                
                # Create DataFrame for Altair chart
                scores_df = pd.DataFrame(list(result['detailed_scores'].items()), columns=['Accent', 'Score'])
                
                # Sort DataFrame by score for better visualization
                scores_df = scores_df.sort_values('Score', ascending=False)

                # Create Altair bar chart
                chart = alt.Chart(scores_df).mark_bar().encode(
                    x=alt.X('Accent:N', sort='y', title='Accent'), # Sort X-axis by Y-value
                    y=alt.Y('Score:Q', title='Calculated Feature Score'),
                    color=alt.Color('Accent:N', legend=None), # Color bars by accent type, hide legend
                    tooltip=['Accent', 'Score'] # Add tooltips for interactivity
                ).properties(
                    title='Calculated Accent Feature Scores'
                ).interactive() # Enable basic interactivity like zoom/pan

                st.altair_chart(chart, use_container_width=True)

                # Optionally display raw scores as text below the chart
                st.markdown("Raw Scores:")
                # Format detailed scores as a list for clarity
                detailed_scores_list = "\n".join([f"- **{accent}**: {score:.2f}" for accent, score in result['detailed_scores'].items()])
                st.markdown(detailed_scores_list)


            # Display transcript
            st.header("3. Speech Transcript")
            if result['transcript']:
                 st.text_area("Detected Speech", result['transcript'], height=200)
            else:
                 st.info("No significant speech was transcribed.")


            st.markdown("---") # Separator

            st.info("""
            **Important Notes:**
            *   This tool uses a heuristic approach based on common word choices and patterns detectable from a transcript.
            *   Actual accent classification requires analyzing phonetic nuances (pronunciation, intonation, rhythm), which are not fully captured by transcribing text alone.
            *   The accuracy depends heavily on the clarity, length, and content of the speech in the video.
            *   For professional and accurate accent evaluation, consult with a trained linguist or speech pathologist.
            """)

        finally:
            # Clean up temporary files
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    # print(f"Removed temporary video: {video_path}") # Debugging
                except OSError as e:
                    st.warning(f"Could not remove temporary video file {video_path}: {e}")

            if audio_path and os.path.exists(audio_path):
                 try:
                     os.remove(audio_path)
                     # print(f"Removed temporary audio: {audio_path}") # Debugging
                 except OSError as e:
                    st.warning(f"Could not remove temporary audio file {audio_path}: {e}")


if __name__ == "__main__":
    main()
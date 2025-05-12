import streamlit as st
import os
import tempfile
import uuid
import re  # Use the re module for tokenization
import numpy as np
import requests
import time
import subprocess
import pandas as pd
from pydub import AudioSegment
import speech_recognition as sr
from urllib.parse import urlparse
# Removed nltk imports
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Removed NLTK download block

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
        # Ensure directory exists, although tempfile usually handles this
        os.makedirs(temp_dir, exist_ok=True) 
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.mp4")
        
        # Handle Loom URLs
        # This is a basic attempt and might fail if Loom changes its structure
        if "loom.com" in url:
            # Attempt to extract video ID - this is not officially supported
            # and might break. A robust solution might require a specific Loom API.
            # This remains a potential point of failure for Loom links.
            match = re.search(r"/share/([\w-]+)", url)
            if match:
                 video_id = match.group(1)
                 # Construct potential direct URL - again, not guaranteed by Loom
                 direct_url = f"https://cdn.loom.com/sessions/thumbnails/{video_id}-with-intro-outro.mp4" # Common pattern
                 # Fallback to original if the common pattern doesn't work
                 try:
                     response = requests.head(direct_url, allow_redirects=True)
                     if response.status_code != 200:
                          direct_url = f"https://cdn.loom.com/sessions/thumbnails/{video_id}.mp4" # Another pattern
                 except:
                     direct_url = url # Use original if head request fails
            else:
                direct_url = url # Fallback if ID extraction fails
            
        else:
            direct_url = url
            
        st.write(f"Attempting to download from: {direct_url}") # Debugging line
        
        # Download the video
        response = requests.get(direct_url, stream=True, timeout=30) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        # Save to temporary file
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): # Increased chunk size
                if chunk:
                    f.write(chunk)
        
        # Basic check if file was actually written
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
             return None, "Downloaded file is empty or doesn't exist after saving."

        return temp_path, None
    
    except requests.exceptions.RequestException as req_e:
         return None, f"Error downloading video (Request failed): {str(req_e)}. Please ensure the URL is directly accessible."
    except Exception as e:
        return None, f"Error downloading video: {str(e)}"

# Function to extract audio using FFmpeg directly
def extract_audio(video_path):
    """Extract audio from video file using FFmpeg"""
    try:
        # Create a temporary file for audio
        temp_dir = tempfile.gettempdir()
        # Ensure directory exists
        os.makedirs(temp_dir, exist_ok=True) 
        audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
        
        # FFmpeg command to extract audio as WAV
        # -y: Overwrite output files without asking
        # -i: Input file
        # -vn: No video
        # -acodec pcm_s16le: Audio codec (signed 16-bit little-endian PCM)
        # -ar 44100: Audio sample rate (44.1 kHz)
        # -ac 1: Audio channels (mono) - mono often works better for transcription
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", audio_path]
        
        # Run FFmpeg command
        # Capture stdout and stderr
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Check if FFmpeg was successful
        if process.returncode != 0:
            return None, f"Error extracting audio with FFmpeg:\n{stderr.decode()}"
        
        # Check if file exists and has size > 0
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            # Try to provide more context if FFmpeg reported success but no file
            error_output = stderr.decode()
            if "Input file does not contain any stream" in error_output:
                 return None, "Error extracting audio: FFmpeg reported no audio stream found in the video."
            return None, f"Failed to extract audio: Output file '{audio_path}' is empty or doesn't exist. FFmpeg output:\n{error_output}"
        
        return audio_path, None
    
    except FileNotFoundError:
         return None, "FFmpeg command not found. Please ensure FFmpeg is installed and in your system's PATH."
    except Exception as e:
        return None, f"Error extracting audio: {str(e)}"

# Function to check if FFmpeg is installed
def check_ffmpeg():
    try:
        process = subprocess.Popen(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return process.returncode == 0
    except (FileNotFoundError, Exception):
        return False

# Function to transcribe audio
def transcribe_audio(audio_path):
    """Transcribe audio file using speech recognition"""
    try:
        # Load audio file using pydub for processing
        # Ensure audio is in the correct format (mono, 16-bit PCM, 44100Hz)
        # Pydub might re-encode if needed, but FFmpeg already did this.
        audio = AudioSegment.from_wav(audio_path)
        
        # Split audio into chunks (e.g., 30-60 seconds) to manage memory and API limits
        chunk_length_ms = 45 * 1000  # 45 seconds
        chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        full_transcript = ""
        
        st.info(f"Processing {len(chunks)} audio chunk(s) for transcription...")
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Use a temporary file for each chunk to pass to speech_recognition
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                chunk_path = fp.name
            
            try:
                # Export chunk to temporary file
                chunk.export(chunk_path, format="wav")
                
                # Transcribe chunk
                with sr.AudioFile(chunk_path) as source:
                    # Adjust for ambient noise before recording
                    recognizer.adjust_for_ambient_noise(source, duration=0.5) 
                    audio_data = recognizer.record(source)
                    
                    st.write(f"Transcribing chunk {i+1}...")
                    try:
                        # Using Google's speech recognition (requires internet)
                        # Can add language parameter if needed, e.g., language="en-US"
                        transcript = recognizer.recognize_google(audio_data)
                        full_transcript += " " + transcript
                        st.write(f"Chunk {i+1} transcribed.")
                    except sr.UnknownValueError:
                        st.write(f"Speech Recognition could not understand audio in chunk {i+1}")
                        pass # Ignore chunks with no recognizable speech
                    except sr.RequestError as e:
                        st.error(f"Could not request results from Google Speech Recognition service for chunk {i+1}; {e}")
                        # Optionally break or continue depending on desired behavior
                    except Exception as e:
                         st.warning(f"An unexpected error occurred during transcription of chunk {i+1}: {str(e)}")
                         
            finally:
                # Ensure the temporary file is removed
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
        
        return full_transcript.strip(), None
    
    except Exception as e:
        return None, f"Error transcribing audio: {str(e)}"

# Function to analyze accent
def analyze_accent(transcript):
    """Analyze the accent based on transcript using regex tokenization"""
    if not transcript:
        return None, "No speech detected in the audio"
    
    # Lowercased transcript for analysis
    text = transcript.lower()
    
    # Tokenize using regex: Find sequences of letters, possibly with an apostrophe inside
    # This is a simpler replacement for nltk.word_tokenize for this purpose
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    
    if not words:
         return None, "No recognizable words found in the transcript for analysis."
    
    # Track scores for each accent
    scores = {accent: 0.0 for accent in ACCENT_PATTERNS}
    
    # Analyze word markers
    word_count = Counter(words)
    total_words = len(words)
    
    # Base score to ensure all accents get a baseline consideration
    base_score_per_accent = 10.0 # Give each accent a starting score
    for accent in scores:
        scores[accent] += base_score_per_accent

    # Factors for scoring
    word_marker_factor = 5.0 # Score per word marker instance (capped)
    spelling_marker_factor = 15.0 # Score per spelling marker instance
    phonetic_pattern_factor = 15.0 # Score per matched phonetic pattern indicator
    
    for accent, patterns in ACCENT_PATTERMS.items():
        # Check for word markers
        # Cap the influence of frequently repeated markers
        marker_count = sum(min(word_count.get(marker.lower(), 0), 5) for marker in patterns["word_markers"]) # Cap count per marker
        marker_score = marker_count * word_marker_factor
        scores[accent] += marker_score
        
        # Check for spelling markers (in full text)
        spelling_matches = sum(1 for marker in patterns["spelling_markers"] if marker.lower() in text)
        scores[accent] += spelling_matches * spelling_marker_factor
        
        # Check for phonetic patterns (approximated via text analysis)
        # Use regex to find word patterns indicative of phonetic features
        
        # Example: American T-flapping / R-coloring indicators
        if accent == "American":
            # 'gonna', 'wanna', 'gotta' - common contractions
            if re.search(r'\b(gonna|wanna|gotta)\b', text):
                scores[accent] += phonetic_pattern_factor
            # Words where T might be flapped in American English ('city', 'better', 'water')
            if re.search(r'\b(city|better|water)\b', text):
                 scores[accent] += phonetic_pattern_factor * 0.75 # Slightly less weight
            # Words with R after vowel ('car', 'bird', 'far') - indicative of rhoticity
            if re.search(r'\b([a-z]+er\b|[a-z]+ar\b|[a-z]+or\b)', text):
                 scores[accent] += phonetic_pattern_factor * 0.5 # Lower weight, as this is common but pronunciation varies

        # Example: British non-rhoticity / glottal stop indicators
        elif accent == "British":
             # Words ending in r/re where R is often dropped ('car', 'there', 'sister')
             if re.search(r'\b([a-z]+er\b|[a-z]+ar\b|[a-z]+or\b|[a-z]+re\b)', text) and not re.search(r'\b([a-z]+[aeiouy]r)\b\s+([aeiouy])', text):
                  # Simple check for R followed by vowel sound (linking R), suggesting non-rhoticity might be present
                  scores[accent] += phonetic_pattern_factor * 0.5 # Moderate weight
             # Words with potential glottal stop ('bottle', 'better', 'city')
             if re.search(r'\b(bottle|better|city)\b', text):
                  scores[accent] += phonetic_pattern_factor * 0.75

        # Example: Australian features (can overlap with British)
        elif accent == "Australian":
             # Specific slang checked in word markers, add a bonus if multiple slang terms appear
             if sum(word_count.get(marker.lower(), 0) for marker in ["arvo", "brekkie", "footy", "ute"]) > 1:
                  scores[accent] += phonetic_pattern_factor

        # Example: Indian English features
        elif accent == "Indian":
            # Vocabulary specific to Indian English
            if re.search(r'\b(prepone|timepass|batch-mate|cousin-brother|cousin-sister)\b', text):
                 scores[accent] += phonetic_pattern_factor

        # Example: Canadian features (canadian raising indicators)
        elif accent == "Canadian":
             # Words potentially affected by Canadian raising ('about', 'house', 'out', 'loud')
             if re.search(r'\b(about|house|out|loud)\b', text):
                  scores[accent] += phonetic_pattern_factor

    # Find best match
    best_accent = max(scores, key=scores.get)
    
    # Calculate confidence
    max_score = scores[best_accent]
    total_score = sum(scores.values())
    
    # Normalize confidence relative to the maximum possible score for the transcript length
    # A rough estimate of potential maximum score based on transcript length
    # This is a heuristic and can be adjusted
    max_possible_score = base_score_per_accent * len(ACCENT_PATTERNS) + \
                         min(total_words, len(ACCENT_PATTERNS["American"]["word_markers"])*5 + len(ACCENT_PATTERNS["British"]["word_markers"])*5 + len(ACCENT_PATTERNS["Australian"]["word_markers"])*5 + len(ACCENT_PATTERNS["Indian"]["word_markers"])*5 + len(ACCENT_PATTERNS["Canadian"]["word_markers"])*5) * word_marker_factor + \
                         len(ACCENT_PATTERNS["American"]["spelling_markers"]) * spelling_marker_factor + len(ACCENT_PATTERNS["British"]["spelling_markers"]) * spelling_marker_factor + len(ACCENT_PATTERNS["Australian"]["spelling_markers"]) * spelling_marker_factor + len(ACCENT_PATTERNS["Indian"]["spelling_markers"]) * spelling_marker_factor + len(ACCENT_PATTERNS["Canadian"]["spelling_markers"]) * spelling_marker_factor + \
                         len(ACCENT_PATTERNS) * phonetic_pattern_factor # Assuming roughly one phonetic match per accent type is possible

    # More robust confidence calculation: Compare the top score to the second best,
    # and also consider the absolute score value relative to total words.
    
    scores_sorted = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    best_score = scores_sorted[0][1] if scores_sorted else 0
    second_score = scores_sorted[1][1] if len(scores_sorted) > 1 else 0
    
    # Confidence based on margin over the second best
    margin = best_score - second_score
    
    # Confidence based on absolute score (relative to transcript length or word count)
    # Using total_words as a simple proxy for content length
    score_per_word = best_score / max(1, total_words)
    
    # Combine margin and score_per_word into a confidence metric
    # This is a heuristic; values can be tuned.
    # Example: Add a base confidence, then bonus for margin and score density.
    confidence = 30 # Base confidence
    confidence += min(margin * 2, 40) # Bonus for margin (capped)
    confidence += min(score_per_word * 200, 30) # Bonus for score density (capped)
    
    # Ensure confidence is within a reasonable range (e.g., 10% to 95%)
    confidence = max(10, min(round(confidence, 1), 95.0))
    
    result = {
        "accent": best_accent,
        "confidence": confidence,
        "description": ACCENT_PATTERNS[best_accent]["description"],
        "transcript": transcript,
        "detailed_scores": {k: round(v, 2) for k, v in scores.items()},
        "second_best": scores_sorted[1][0] if len(scores_sorted) > 1 else None
    }
    
    return result, None

# Function to validate URL
def is_valid_url(url):
    try:
        result = urlparse(url)
        # Check if scheme (e.g., http, https) and network location (domain) are present
        return all([result.scheme, result.netloc])
    except ValueError: # urlparse can raise ValueError for invalid URLs
        return False
    except Exception: # Catch any other potential errors
        return False


# Main application
def main():
    # Check if FFmpeg is installed
    ffmpeg_installed = check_ffmpeg()
    if not ffmpeg_installed:
        st.error("""
        ⚠️ **FFmpeg is not installed or not found in PATH.** This tool requires FFmpeg for audio extraction.
        
        Please install FFmpeg:
        - **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` directory to your system's PATH.
        - **macOS:** Open Terminal and run `brew install ffmpeg` (requires Homebrew: [brew.sh](https://brew.sh)).
        - **Linux:** Open Terminal and run `sudo apt update && sudo apt install ffmpeg`.
        
        After installation, restart Streamlit or your terminal session.
        """)
        st.stop()
    
    # Input section
    st.header("1. Input Video")
    
    url = st.text_input("Enter public video URL (MP4 or Loom link):", 
                         placeholder="e.g., https://loom.com/share/your-video-id or https://example.com/video.mp4",
                         help="Ensure the video is publicly accessible.")
    
    # Alternatively allow file upload
    st.markdown("---") # Separator
    uploaded_file = st.file_uploader("Or upload a video file:", type=["mp4", "mov", "avi", "webm", "mkv"])
    
    # Add demo videos (optional but helpful)
    st.markdown("---")
    st.subheader("Try a Demo Video")
    demo_options = {
        "--- Select a demo ---": None,
        "American English Demo": "https://file-examples-com.github.io/uploads/2017/04/file_example_MP4_480_1_5MG.mp4",
        # Add other demo URLs if you have them
        # "British English Demo": "..."
    }
    demo_choice = st.selectbox("Choose a demo video:", list(demo_options.keys()))
    
    if demo_options[demo_choice]:
        url = demo_options[demo_choice]
        st.text_input("Selected Demo URL:", value=url, disabled=True) # Show selected URL

    st.markdown("---")
    
    # Process section
    if st.button("Analyze Accent", type="primary"):
        if not url and not uploaded_file:
            st.error("Please provide either a video URL or upload a video file.")
            return
        
        input_source = "URL" if url else "Upload"
        
        st.header("Processing Steps")
        progress_text = "Operation in progress. Please wait."
        progress_bar = st.progress(0, text=progress_text)
        
        temp_video_path = None
        temp_audio_path = None

        try:
            # Case 1: URL provided
            if url:
                if not is_valid_url(url):
                    st.error("Please enter a valid URL.")
                    return
                
                # Step 1: Download video
                st.info("Step 1: Downloading video...")
                progress_bar.progress(10, text="Step 1/4: Downloading video...")
                temp_video_path, error = download_video(url)
                
                if error:
                    st.error(error)
                    return # Stop execution on error
                
                progress_bar.progress(25, text="Step 1/4: Download complete.")
                st.success("Video downloaded successfully.")

            # Case 2: File uploaded
            else:
                # Save uploaded file to temp location
                st.info("Step 1: Saving uploaded video...")
                progress_bar.progress(10, text="Step 1/4: Saving uploaded file...")
                
                # Use the actual file extension from the upload
                file_extension = os.path.splitext(uploaded_file.name)[1]
                temp_video_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{file_extension}")
                
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                progress_bar.progress(25, text="Step 1/4: File saved.")
                st.success("Uploaded video saved temporarily.")
            
            # Step 2: Extract audio
            st.info("Step 2: Extracting audio...")
            progress_bar.progress(40, text="Step 2/4: Extracting audio...")
            temp_audio_path, error = extract_audio(temp_video_path)
            
            if error:
                st.error(error)
                return # Stop execution on error

            progress_bar.progress(60, text="Step 2/4: Audio extracted.")
            st.success("Audio extracted successfully.")

            # Step 3: Transcribe speech
            st.info("Step 3: Transcribing speech...")
            progress_bar.progress(70, text="Step 3/4: Transcribing speech...")
            transcript, error = transcribe_audio(temp_audio_path)
            
            if error:
                st.error(error)
                return # Stop execution on error
            
            if not transcript or len(transcript.strip()) < 20: # Require a bit more text
                st.warning("Could not detect sufficient clear English speech in the audio. Analysis may be inaccurate or impossible.")
                # Depending on how strict you want to be, you could return here
                # For now, let's proceed with whatever transcript we got, but warn the user.
                # If transcript is truly empty after strip, the analyze_accent function handles it.
                if not transcript:
                    st.error("No speech detected in the audio.")
                    return

            progress_bar.progress(90, text="Step 3/4: Transcription complete.")
            st.success("Speech transcribed successfully.")

            # Step 4: Analyze accent
            st.info("Step 4: Analyzing accent...")
            progress_bar.progress(95, text="Step 4/4: Analyzing accent...")
            result, error = analyze_accent(transcript)
            
            if error:
                st.error(error)
                return # Stop execution on error

            progress_bar.progress(100, text="Step 4/4: Analysis complete.")
            st.success("Accent analysis complete!")
            
            # Display results
            st.header("2. Analysis Results")
            
            # Summary box
            st.success(f"✅ **Analysis Complete!**")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Primary Accent Classification")
                
                primary_accent = result['accent']
                confidence = result['confidence']
                
                # Display accent and confidence
                st.metric(
                    label=f"Classified Accent:", 
                    value=f"{primary_accent} English"
                )
                st.metric(
                     label="Confidence Score:",
                     value=f"{confidence}%"
                )

                # Create description card
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 5px; background-color: #0e1117; margin-top: 15px;">
                    <p><strong>Characteristics of {primary_accent} English:</strong></p>
                    <p>{result['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display secondary influence if available and score is meaningful
                if result['second_best'] and result['detailed_scores'][result['second_best']] > result['detailed_scores'][primary_accent] * 0.3: # Only show if secondary score is significant
                    st.markdown("""
                    <div style="margin-top: 20px;">
                        <p><strong>Potential secondary influences detected:</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info(f"**{result['second_best']} English**")
                else:
                    st.markdown("""
                    <div style="margin-top: 20px;">
                        <p><strong>No significant secondary accent influence detected.</strong></p>
                    </div>
                    """, unsafe_allow_html=True)


            with col2:
                st.subheader("Relative Accent Scores")
                
                # Format detailed scores
                scores = result['detailed_scores']
                accents = list(scores.keys())
                values = list(scores.values())
                
                # Create a proper dataframe for charting
                score_df = pd.DataFrame({
                    'Accent': accents,
                    'Score': values
                })
                # Sort for better visualization
                score_df = score_df.sort_values('Score', ascending=False)
                
                # Display as bar chart
                st.bar_chart(
                    score_df.set_index('Accent'),
                    use_container_width=True
                )
                
                # Display detailed scores in tabular format
                st.subheader("Detailed Scores Table")
                # Normalize scores for display purposes if max_val > 0
                max_val = score_df['Score'].max()
                if max_val > 0:
                     score_df['Relative Score (%)'] = round((score_df['Score'] / max_val) * 100, 1)
                else:
                     score_df['Relative Score (%)'] = 0
                     
                st.dataframe(score_df[['Accent', 'Score', 'Relative Score (%)']], use_container_width=True, hide_index=True)


            # Display transcript
            st.header("3. Transcript")
            st.text_area("Speech Transcript", result['transcript'], height=200, disabled=True)
            
            st.info("""
            **Disclaimer:** This analysis is based on identifying textual patterns and common linguistic features often associated with different accents in transcribed speech. 
            It does **not** analyze phonetic pronunciation directly from the audio. 
            The results are probabilistic and should be interpreted as indicative trends rather than definitive classifications. 
            For professional accent evaluation, consult with a trained linguist.
            """)

        except Exception as e:
             st.error(f"An unexpected error occurred during processing: {str(e)}")
             st.exception(e) # Display traceback for debugging

        finally:
            # Clean up temporary files
            st.info("Cleaning up temporary files...")
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                    st.write(f"Removed temporary video: {temp_video_path}")
                except Exception as e:
                    st.warning(f"Could not remove temporary video file {temp_video_path}: {e}")
            if temp_audio_path and os.path.exists(temp_audio_path):
                 try:
                     os.remove(temp_audio_path)
                     st.write(f"Removed temporary audio: {temp_audio_path}")
                 except Exception as e:
                     st.warning(f"Could not remove temporary audio file {temp_audio_path}: {e}")
            
            progress_bar.empty() # Hide progress bar


if __name__ == "__main__":
    main()

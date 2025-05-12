import streamlit as st
import os
import tempfile
import uuid
import re  # Used for simple tokenization
import numpy as np
import requests
import time
import subprocess
import pandas as pd
from pydub import AudioSegment
import speech_recognition as sr
from urllib.parse import urlparse
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="English Accent Analyzer",
    page_icon="🎤",
    layout="wide"
)

# Title and description
st.title("🎤 English Accent Analyzer")
st.markdown("""
This tool performs a **text-based** analysis of the speaker's accent from a video. 
Simply provide a public video URL (e.g., a direct MP4 file or Loom video) or upload a file.
The system will:
1. Extract the audio using FFmpeg.
2. Transcribe the speech using Google's Speech Recognition (requires internet access).
3. Analyze the transcript for linguistic patterns (vocabulary, spelling, simplified phonetic cues).
4. Provide a classification with a confidence score based on detected patterns.

**Important:** This analysis relies solely on the text transcript and not the audio's phonetic details. Results are indicative, not definitive.
""")

# Accent feature patterns dictionary - DEFINED GLOBALLY
ACCENT_PATTERNS = {
    "American": {
        "phonetic_patterns": ["r after vowels", "t flapping", "o as 'ah'"],
        "word_markers": ["gonna", "wanna", "y'all", "awesome", "totally", "like"],
        "spelling_markers": ["color", "center", "defense"],
        "description": "Commonly features rhotic pronunciation (pronouncing 'r' after vowels), flapping of 't' sounds (like 'tt' in 'better'), and specific vocabulary/contractions."
    },
    "British": {
        "phonetic_patterns": ["non-rhotic", "t glottal stop", "broader 'a'"],
        "word_markers": ["proper", "brilliant", "cheers", "mate", "bloody", "quite"],
        "spelling_markers": ["colour", "centre", "defence"],
        "description": "Often non-rhotic (dropping 'r' after vowels unless followed by a vowel sound), uses clear 't' pronunciation or glottal stops, and distinctive vocabulary."
    },
    "Australian": {
        "phonetic_patterns": ["rising intonation", "i sound change", "non-rhotic"],
        "word_markers": ["mate", "no worries", "arvo", "reckon", "heaps"],
        "spelling_markers": ["colour", "centre", "defence"], # Shares with British commonly
        "description": "Known for potential rising intonation at sentence ends (high rising terminal), extended vowels, and unique slang terms (e.g., arvo, reckon)."
    },
    "Indian": {
        "phonetic_patterns": ["retroflex consonants", "v/w confusion", "stress timing"], # These are hard to detect from text
        "word_markers": ["actually", "itself", "only", "kindly", "prepone"], # Common Indian English vocabulary
        "spelling_markers": ["colour", "centre", "defence"], # Can use British spelling
        "description": "May exhibit retroflex consonant sounds, syllable-timing rather than stress-timing, and specific vocabulary borrowed from Indian languages or unique phrasing."
    },
    "Canadian": {
        "phonetic_patterns": ["canadian raising", "rhotic", "about pronunciation"],
        "word_markers": ["eh", "sorry", "toque", "washroom", "loonie", "double-double"],
        "spelling_markers": ["colour", "centre", "defence"], # Can use British spelling
        "description": "A mix of American and British influences, often rhotic like American, but with unique vowel shifts (Canadian raising in words like 'about') and distinct vocabulary."
    }
}


# Function to download a video from a URL
def download_video(url):
    """Download video from URL to a temporary file"""
    try:
        # Create a temporary directory if it doesn't exist (tempfile.gettempdir() is usually sufficient)
        temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True) 
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.mp4")
        
        direct_url = url # Start with the provided URL

        # Basic attempt to handle Loom URLs - This is heuristic and not guaranteed to work
        # A robust solution might require Loom's API or a more sophisticated method.
        if "loom.com" in url:
             match = re.search(r"/share/([\w-]+)", url)
             if match:
                 video_id = match.group(1)
                 # Attempt common Loom download patterns
                 potential_urls = [
                     f"https://cdn.loom.com/sessions/thumbnails/{video_id}-with-intro-outro.mp4",
                     f"https://cdn.loom.com/sessions/thumbnails/{video_id}.mp4"
                 ]
                 
                 found_direct = False
                 for p_url in potential_urls:
                      try:
                           # Check if the URL is valid and accessible without actually downloading
                           response = requests.head(p_url, allow_redirects=True, timeout=5)
                           if response.status_code == 200 and 'video' in response.headers.get('Content-Type', ''):
                                direct_url = p_url
                                found_direct = True
                                break
                      except requests.exceptions.RequestException:
                           # Ignore if head request fails for this potential URL
                           pass
                 
                 if not found_direct:
                      st.warning(f"Could not find a direct MP4 link for the Loom video ({video_id}). Attempting original URL, which may not work for download.")
                      direct_url = url # Fallback to original if direct patterns fail

        st.write(f"Attempting to download from: {direct_url}") 
        
        # Download the video with a timeout and error checking
        response = requests.get(direct_url, stream=True, timeout=60) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        # Save to temporary file
        download_size = 0
        chunk_size = 8192 # 8KB chunks
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    download_size += len(chunk)
        
        # Basic check if file was actually written and has content
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
             return None, f"Downloaded file is empty or doesn't exist after saving. Downloaded {download_size} bytes."

        st.write(f"Downloaded {download_size} bytes to {temp_path}")
        return temp_path, None
    
    except requests.exceptions.Timeout:
         return None, "Error downloading video: Request timed out."
    except requests.exceptions.ConnectionError:
         return None, "Error downloading video: Failed to connect to the URL. Please check the URL or your internet connection."
    except requests.exceptions.HTTPError as http_e:
         return None, f"Error downloading video: HTTP Error {http_e.response.status_code} - {http_e.response.reason}. Please ensure the URL is publicly accessible."
    except Exception as e:
        return None, f"An unexpected error occurred during video download: {str(e)}"

# Function to extract audio using FFmpeg directly
def extract_audio(video_path):
    """Extract audio from video file using FFmpeg"""
    try:
        temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True) 
        audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
        
        # FFmpeg command to extract audio as WAV
        # -y: Overwrite output files without asking
        # -i: Input file
        # -vn: No video stream
        # -acodec pcm_s16le: Audio codec (signed 16-bit little-endian PCM) - widely compatible
        # -ar 16000: Audio sample rate (16 kHz) - sufficient for speech, smaller file
        # -ac 1: Audio channels (mono) - ideal for most speech recognition
        # -loglevel error: Suppress verbose FFmpeg output, only show errors
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-loglevel", "error", audio_path]
        
        st.write(f"Running FFmpeg command: {' '.join(cmd)}") # Debugging FFmpeg command
        
        # Run FFmpeg command
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Check if FFmpeg was successful
        if process.returncode != 0:
            # Include stderr output in the error message
            return None, f"Error extracting audio with FFmpeg:\n{stderr.decode('utf-8', errors='ignore')}"
        
        # Check if file exists and has size > 0
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            # Try to provide more context if FFmpeg reported success but no file
            error_output = stderr.decode('utf-8', errors='ignore')
            if "Input file does not contain any stream" in error_output:
                 return None, "Error extracting audio: FFmpeg reported no audio stream found in the video."
            if "codec not found" in error_output or "Invalid data found when processing input" in error_output:
                 return None, f"Error extracting audio: FFmpeg could not process the video file format. FFmpeg output:\n{error_output}"
            return None, f"Failed to extract audio: Output file '{audio_path}' is empty or doesn't exist. FFmpeg output:\n{error_output}"
        
        st.write(f"Audio extracted successfully to {audio_path}")
        return audio_path, None
    
    except FileNotFoundError:
         return None, "FFmpeg command not found. Please ensure FFmpeg is installed and in your system's PATH."
    except Exception as e:
        return None, f"An unexpected error occurred during audio extraction: {str(e)}"

# Function to check if FFmpeg is installed
def check_ffmpeg():
    try:
        # Use shell=True on Windows might help find it if it's in PATH but not in the current process's environment
        # On Linux/macOS, shell=False is generally safer unless you need shell features
        is_windows = os.name == 'nt'
        process = subprocess.Popen(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=is_windows)
        stdout, stderr = process.communicate()
        return process.returncode == 0
    except (FileNotFoundError, Exception):
        return False

# Function to transcribe audio
def transcribe_audio(audio_path):
    """Transcribe audio file using speech recognition"""
    try:
        # Load audio file using pydub for processing
        # Ensure audio is in the correct format (mono, 16-bit PCM, 16000Hz as extracted)
        audio = AudioSegment.from_wav(audio_path)
        
        # Split audio into chunks (e.g., 30 seconds) to manage memory and API limits
        chunk_length_ms = 30 * 1000  # 30 seconds
        chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        full_transcript = ""
        
        st.info(f"Processing {len(chunks)} audio chunk(s) for transcription...")
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Use a temporary file for each chunk to pass to speech_recognition
            # Use suffix '.wav' to help speech_recognition identify the format
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                chunk_path = fp.name
            
            try:
                # Export chunk to temporary file
                chunk.export(chunk_path, format="wav")
                
                # Transcribe chunk
                with sr.AudioFile(chunk_path) as source:
                    # Adjust for ambient noise before recording for potentially better results
                    # Can set a timeout for recording if needed, but record() processes the whole file/chunk
                    # recognizer.adjust_for_ambient_noise(source, duration=0.5) # Adjust duration if needed
                    
                    st.write(f"Transcribing chunk {i+1}/{len(chunks)}...")
                    audio_data = recognizer.record(source)
                    
                    try:
                        # Using Google's speech recognition (requires internet)
                        # Set language to English for potentially better accuracy
                        transcript = recognizer.recognize_google(audio_data, language="en-US")
                        full_transcript += " " + transcript
                        st.write(f"Chunk {i+1} transcribed.")
                    except sr.UnknownValueError:
                        st.write(f"Speech Recognition could not understand audio in chunk {i+1}. Skipping.")
                        pass # Ignore chunks with no recognizable speech
                    except sr.RequestError as e:
                        # API was unreachable or unresponsive
                        st.error(f"Could not request results from Google Speech Recognition service for chunk {i+1}; {e}. Check your internet connection.")
                        # Decide if you want to stop or try the next chunk
                        # For now, we'll continue to try other chunks
                    except Exception as e:
                         st.warning(f"An unexpected error occurred during transcription of chunk {i+1}: {str(e)}")
                         
            finally:
                # Ensure the temporary file is removed, even if transcription failed
                if os.path.exists(chunk_path):
                    try:
                        os.remove(chunk_path)
                    except Exception as e:
                        st.warning(f"Could not remove temporary chunk file {chunk_path}: {e}")

        return full_transcript.strip(), None
    
    except Exception as e:
        return None, f"An unexpected error occurred during audio transcription process: {str(e)}"

# Function to analyze accent
def analyze_accent(transcript):
    """Analyze the accent based on transcript using regex tokenization"""
    if not transcript or len(transcript.strip()) < 10: # Require at least 10 characters of transcript
        return None, "Transcript too short for meaningful analysis."
    
    # Lowercased transcript for analysis
    text = transcript.lower()
    
    # Tokenize using regex: Find sequences of letters, possibly with an apostrophe inside
    # This simple regex handles words and contractions like don't, isn't.
    # It will skip punctuation.
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    
    if not words:
         return None, "No recognizable words found in the transcript for analysis."
         
    st.write(f"Analyzed {len(words)} words.")

    # Track scores for each accent
    scores = {accent: 0.0 for accent in ACCENT_PATTERNS}
    
    # Scoring factors (tuned weights - these are heuristic)
    base_score_per_accent = 5.0 # Minimum score for each accent
    word_marker_factor = 3.0 # Score per word marker instance (capped per marker)
    spelling_marker_factor = 10.0 # Score per spelling marker instance (present in text)
    phonetic_pattern_factor = 10.0 # Score per matched phonetic pattern indicator (regex match)
    
    # Add base score
    for accent in scores:
        scores[accent] += base_score_per_accent

    # Analyze word markers
    word_count = Counter(words)
    
    for accent, patterns in ACCENT_PATTERNS.items():
        # Check for word markers
        # Cap the influence of frequently repeated markers (e.g., 'like', 'actually')
        marker_score = 0
        for marker in patterns["word_markers"]:
             marker_score += min(word_count.get(marker.lower(), 0), 3) * word_marker_factor # Cap each marker at 3 instances
        scores[accent] += marker_score
        
        # Check for spelling markers (look for exact string in the full text)
        spelling_matches = sum(1 for marker in patterns["spelling_markers"] if marker.lower() in text)
        scores[accent] += spelling_matches * spelling_marker_factor
        
        # Check for phonetic pattern *indicators* (via word patterns)
        # These are approximations based on words that *might* be pronounced a certain way
        
        if accent == "American":
            # Indicators for t-flapping or linking R
            if re.search(r'\b(water|better|city|data)\b', text): scores[accent] += phonetic_pattern_factor * 0.7
            if re.search(r'\b(car is|far away|there is)\b', text): scores[accent] += phonetic_pattern_factor * 0.5 # Linking R indicator
            # Indicators for 'o' sounds (e.g., hot, stop)
            if re.search(r'\b(hot|stop|got)\b', text): scores[accent] += phonetic_pattern_factor * 0.3 # Weak indicator

        elif accent == "British":
             # Indicators for non-rhoticity (R not followed by vowel)
             if re.search(r'\b([a-z]+er|ar|or)\b(?!\s+[aeiou])', text): scores[accent] += phonetic_pattern_factor * 0.7 # Word ends with R-vowel combination, not followed by vowel
             # Indicators for glottal stops
             if re.search(r'\b(bottle|butter|little)\b', text): scores[accent] += phonetic_pattern_factor * 0.6

        elif accent == "Australian":
             # Indicators for specific vowel sounds (hard to capture reliably from text)
             # Rely more on slang/word markers for this accent in a text-based analysis
             pass # No strong text-based phonetic indicators added here

        elif accent == "Indian":
            # Indicators for specific consonant clusters or pronunciations (very hard from text)
            # Rely more on vocabulary
            pass # No strong text-based phonetic indicators added here

        elif accent == "Canadian":
             # Indicators for Canadian Raising (e.g., 'out', 'house')
             if re.search(r'\b(about|house|out|loud)\b', text): scores[accent] += phonetic_pattern_factor

    # Find best match
    scores_sorted = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    if not scores_sorted: # Should not happen if ACCENT_PATTERNS is not empty
        return None, "Internal error: Could not calculate scores."

    best_accent = scores_sorted[0][0]
    best_score = scores_sorted[0][1]
    
    # Calculate confidence
    # Confidence is a heuristic based on the difference between the top two scores
    # and the overall score relative to the total score or number of indicators found.
    
    second_best = scores_sorted[1][0] if len(scores_sorted) > 1 else None
    second_score = scores_sorted[1][1] if len(scores_sorted) > 1 else 0
    
    # Margin between best and second best
    margin = best_score - second_score
    
    # Total score as a measure of how many patterns were matched overall
    total_score = sum(scores.values()) - base_score_per_accent * len(ACCENT_PATTERNS) # Exclude base score from total for this calculation

    # Heuristic confidence calculation:
    # Start with a base level.
    # Add points based on the margin over the second best.
    # Add points based on the absolute best score relative to the total possible indicators matched (simplified).
    
    confidence = 20 # Starting confidence
    
    # Add confidence based on margin (linear or non-linear scale)
    confidence += min(margin * 3, 50) # Cap the margin's influence

    # Add confidence based on the overall strength of the best score (relative to something)
    # Using the best score relative to the sum of all scores (excluding base) is one way
    if total_score > 0:
        relative_best_score = best_score / total_score
        confidence += min(relative_best_score * 40, 30) # Cap this influence

    # Ensure confidence is within bounds
    confidence = max(10.0, min(round(confidence, 1), 95.0)) # Cap between 10% and 95%

    result = {
        "accent": best_accent,
        "confidence": confidence,
        "description": ACCENT_PATTERNS[best_accent]["description"],
        "transcript": transcript,
        "detailed_scores": {k: round(v, 2) for k, v in scores.items()},
        "second_best": second_best
    }
    
    return result, None

# Function to validate URL
def is_valid_url(url):
    """Checks if a string is a valid URL."""
    try:
        result = urlparse(url)
        # Check if scheme (e.g., http, https) and network location (domain) are present
        # And that it's not just a scheme or just a network location
        return all([result.scheme, result.netloc, result.scheme in ['http', 'https']])
    except ValueError: 
        return False
    except Exception: 
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
    
    url_input = st.text_input("Enter public video URL (MP4 or Loom link):", 
                         placeholder="e.g., https://loom.com/share/your-video-id or https://example.com/video.mp4",
                         help="Ensure the video is publicly accessible. Direct video file URLs work best. Loom support is experimental.")
    
    # Alternatively allow file upload
    st.markdown("---") # Separator
    uploaded_file = st.file_uploader("Or upload a video file:", type=["mp4", "mov", "avi", "webm", "mkv"])
    
    # Add demo videos (optional but helpful)
    st.markdown("---")
    st.subheader("Try a Demo Video")
    demo_options = {
        "--- Select a demo ---": "", # Use empty string instead of None
        "American English Demo (Example.com)": "https://file-examples-com.github.io/uploads/2017/04/file_example_MP4_480_1_5MG.mp4",
        # Add other demo URLs if you have them
        # "British English Demo": "...",
        # "Australian English Demo": "..."
    }
    demo_choice = st.selectbox("Choose a demo video:", list(demo_options.keys()))
    
    # Use the selected demo URL if one is chosen, otherwise use the manual input URL
    url_to_process = demo_options[demo_choice] if demo_options[demo_choice] else url_input

    st.markdown("---")
    
    # Process section
    if st.button("Analyze Accent", type="primary"):
        if not url_to_process and not uploaded_file:
            st.error("Please provide either a video URL or upload a video file.")
            return
        
        st.header("Processing Steps")
        progress_text = "Operation in progress. Please wait."
        progress_bar = st.progress(0, text=progress_text)
        
        temp_video_path = None
        temp_audio_path = None

        try:
            # Case 1: URL provided (manual input or demo selection)
            if url_to_process:
                if not is_valid_url(url_to_process):
                    st.error("Please enter a valid URL starting with http or https.")
                    return
                
                # Step 1: Download video
                st.info("Step 1: Downloading video...")
                progress_bar.progress(10, text="Step 1/4: Downloading video...")
                temp_video_path, error = download_video(url_to_process)
                
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
                # Add a check for allowed extensions? The file_uploader type already does this.
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
            
            if not transcript or len(transcript.strip()) < 20: 
                st.warning("Could not detect sufficient clear English speech in the audio. Analysis may be inaccurate or impossible.")
                if not transcript or len(transcript.strip()) < 5: # Set a lower threshold to completely stop if transcript is almost empty
                     st.error("Transcript is too short or empty. Analysis cannot proceed.")
                     return
                # Otherwise, proceed with the limited transcript but keep the warning

            progress_bar.progress(90, text="Step 3/4: Transcription complete.")
            st.success("Speech transcribed successfully.")
            st.text_area("Preview Transcript (for analysis)", transcript.strip(), height=100, disabled=True)


            # Step 4: Analyze accent
            st.info("Step 4: Analyzing accent...")
            progress_bar.progress(95, text="Step 4/4: Analyzing accent...")
            result, error = analyze_accent(transcript)
            
            if error:
                st.error(error)
                # If analysis failed but transcript exists, show the transcript anyway
                if transcript:
                    st.header("3. Transcript")
                    st.text_area("Speech Transcript", transcript.strip(), height=200, disabled=True)
                    st.info("Analysis failed, but transcription was successful.")
                return # Stop execution on error

            progress_bar.progress(100, text="Step 4/4: Analysis complete.")
            st.success("Accent analysis complete!")
            
            # Display results
            st.header("2. Analysis Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Primary Accent Classification")
                
                primary_accent = result['accent']
                confidence = result['confidence']
                
                # Display accent and confidence using metrics for prominence
                st.metric(
                    label=f"Classified Primary Accent:", 
                    value=f"{primary_accent} English"
                )
                st.metric(
                     label="Confidence Score:",
                     value=f"{confidence}%"
                )

                # Create description card
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 5px; background-color: rgba(14, 17, 23, 0.8); margin-top: 15px;">
                    <p style="font-weight: bold;">Characteristics of {primary_accent} English (based on text analysis):</p>
                    <p>{result['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display secondary influence if available and score is meaningful (e.g., > 20% of best score)
                if result['second_best']:
                     second_best_score = result['detailed_scores'].get(result['second_best'], 0)
                     if second_best_score > result['detailed_scores'].get(primary_accent, 0) * 0.25: # Threshold for showing secondary influence
                         st.markdown("""
                         <div style="margin-top: 20px;">
                             <p style="font-weight: bold;">Potential secondary influences detected:</p>
                         </div>
                         """, unsafe_allow_html=True)
                         
                         st.info(f"**{result['second_best']} English**")
                     else:
                         st.markdown("""
                         <div style="margin-top: 20px;">
                             <p style="font-weight: bold;">No strong secondary accent influence detected.</p>
                         </div>
                         """, unsafe_allow_html=True)
                else: # Should not happen if there's more than one accent defined
                     st.markdown("""
                     <div style="margin-top: 20px;">
                         <p style="font-weight: bold;">No strong secondary accent influence detected.</p>
                     </div>
                     """, unsafe_allow_html=True)


            with col2:
                st.subheader("Relative Accent Scores")
                
                scores = result['detailed_scores']
                
                # Create a proper dataframe for charting
                score_df = pd.DataFrame(list(scores.items()), columns=['Accent', 'Score'])
                # Sort for better visualization
                score_df = score_df.sort_values('Score', ascending=False)
                
                # Display as bar chart
                st.bar_chart(
                    score_df.set_index('Accent'),
                    use_container_width=True,
                    # Add color based on accent? Hardcoding colors can be tricky with seaborn palettes.
                    # Could map accent to color index if needed.
                )
                
                # Display detailed scores in tabular format
                st.subheader("Detailed Scores Table")
                # Calculate Relative Score (%) if max_val > 0
                max_val = score_df['Score'].max()
                if max_val > 0:
                     score_df['Relative Score (%)'] = round((score_df['Score'] / max_val) * 100, 1)
                else:
                     score_df['Relative Score (%)'] = 0
                     
                st.dataframe(score_df[['Accent', 'Score', 'Relative Score (%)']], use_container_width=True, hide_index=True)


            # Display transcript
            st.header("3. Transcript")
            st.text_area("Speech Transcript", result['transcript'].strip(), height=200, disabled=True)
            
            st.info("""
            **Disclaimer:** This analysis is based on identifying textual patterns and common linguistic features often associated with different accents in transcribed speech. 
            It does **not** analyze phonetic pronunciation directly from the audio. 
            The results are probabilistic and should be interpreted as indicative trends rather than definitive classifications. 
            Accuracy depends heavily on the quality of the audio, the clarity of the speech, and the effectiveness of the speech-to-text transcription. 
            For professional accent evaluation, consult with a trained linguist.
            """)

        except Exception as e:
             # Catch any unhandled exceptions during the process
             st.error(f"An unexpected error occurred during processing: {type(e).__name__} - {str(e)}")
             st.exception(e) # Display traceback for debugging

        finally:
            # Clean up temporary files regardless of success or failure
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
            
            # Hide the progress bar once processing is done or an error occurred
            progress_bar.empty()


if __name__ == "__main__":
    main()

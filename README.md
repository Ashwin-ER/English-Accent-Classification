# English Accent Analyzer 

This is a Streamlit web application that analyzes the likely English accent of a speaker from a video by extracting the audio, transcribing the speech, and analyzing the transcript for common accent-specific patterns.

## Features

*   **Video Input:** Accept public video URLs (MP4, Loom) or direct file uploads.
*   **Audio Extraction:** Uses FFmpeg to extract audio efficiently from various video formats.
*   **Speech Transcription:** Transcribes the audio into text using the `speech_recognition` library (leveraging Google's Web Speech API by default).
*   **Accent Analysis:** Analyzes the transcribed text for specific word choices, spellings, and approximated phonetic indicators based on predefined patterns for different English accents.
*   **Accent Classification:** Identifies the most likely accent and provides a confidence score.
*   **Supported Accents:** American, British, Australian, Indian, Canadian.
*   **Interactive UI:** Built with Streamlit for an easy-to-use web interface.
*   **Results Visualization:** Displays detailed scores and transcript.

## Prerequisites

1.  **Python:** You need Python 3.6 or higher installed.
2.  **FFmpeg:** This is a crucial system dependency required for audio extraction.
    *   **macOS:** `brew install ffmpeg` (using Homebrew)
    *   **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install ffmpeg`
    *   **Windows:** Download from the official FFmpeg website (https://ffmpeg.org/download.html) and ensure the `ffmpeg` executable is in your system's PATH.
3.  **Internet Connection:** Required for downloading videos (if using URLs) and for the speech transcription service (Google Web Speech API).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ashwin-ER/English-Accent-Classification.git
    cd English-Accent-Classification
    ```
  

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    *On Windows:*
    ```bash
    venv\Scripts\activate
    ```
    *On macOS/Linux:*
    ```bash
    source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    Create a `requirements.txt` file in the project root with the following content:
    ```txt
    streamlit
    numpy
    requests
    pandas
    pydub
    SpeechRecognition
    nltk
    matplotlib
    seaborn
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install FFmpeg:**
    Follow the instructions in the [Prerequisites](#prerequisites) section for your operating system.

5.  **NLTK Data:**
    The script will attempt to download the necessary `punkt` tokenizer data from NLTK on the first run if it's not already present. This requires an internet connection.

## How to Run

1.  Navigate to the project directory where the script (`your_script_name.py` - likely `app.py` or similar) is located.
2.  Activate your virtual environment (if you created one).
3.  Run the Streamlit application:
    ```bash
    streamlit run your_script_name.py
    ```
    *(Replace `your_script_name.py` with the actual name of your Python script, likely the one provided in the prompt).*
4.  A new tab will open in your web browser with the application.

## How to Use

1.  Open the application URL in your web browser (usually `http://localhost:8501`).
2.  Enter the public URL of an English video (MP4 file, Loom link etc.) or upload a video file directly.
3.  Click the "Analyze Accent" button.
4.  Wait for the process to complete (downloading, extracting audio, transcribing, analyzing).
5.  View the analysis results, including the classified accent, confidence score, and the full transcript.


This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if you include one, otherwise state it here).

## Acknowledgements

*   Built with [Streamlit](https://streamlit.io/).
*   Uses [FFmpeg](https://ffmpeg.org/) for audio extraction.
*   Leverages [pydub](https://github.com/jiaaro/pydub) and [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) for audio processing and transcription.
*   Uses [NLTK](https://www.nltk.org/) for text processing.
*   Visualization powered by [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/).

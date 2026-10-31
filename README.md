# AI-Powered Meeting Transcript Summariser – Mehfil Event Planners

**Task 3: Integrate open-source summarisation model**

Web-based tool that turns raw meeting transcripts into concise, actionable summaries using a local HuggingFace model. No API keys required.

## Features (Task 3)

- Upload plain-text meeting transcripts
- Automatic chunking of long transcripts
- Local summarisation with `sshleifer/distilbart-cnn-12-6`
- Combined summary displayed in a scrollable text area
- Model cached with `@st.cache_resource` for fast subsequent runs
- Sample transcript included for quick testing

## Tech Stack

- **Streamlit** – web UI
- **HuggingFace Transformers** – summarisation pipeline
- **PyTorch** – model inference (CPU)
- Model: `sshleifer/distilbart-cnn-12-6` (DistilBART, CNN/DailyMail fine-tuned)

## Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd mehfil-transcript-summariser

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> First run downloads the model (~300 MB). Subsequent runs use the cache.

## Usage

```bash
streamlit run app.py
```

1. Open the local URL shown in the terminal (usually http://localhost:8501)
2. Upload a `.txt` transcript **or** click “Load sample transcript”
3. Click **Generate Summary**
4. View the combined summary in the text area below the preview

## How the summarisation works

1. Transcript is split into ~400-word chunks (sentence-aware)
2. Each chunk is passed through the DistilBART summarisation pipeline
3. Individual summaries are concatenated with blank lines
4. Result is shown in an `st.text_area` (height 350 px, scrollable)

The model runs entirely on CPU; a typical 5-chunk transcript finishes in well under 30 seconds on a modern laptop after the first model load.

## Project structure

```
mehfil-transcript-summariser/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── sample_transcript.txt   # Example meeting transcript
└── README.md               # This file
```

## What was done (Task 3 note)

- Added `transformers` and `torch` to `requirements.txt`
- Created `load_summarizer()` decorated with `@st.cache_resource` so the pipeline loads only once
- Implemented `summarise_chunks()` that iterates over every chunk, calls the pipeline, and joins the results
- Displayed the final summary in a disabled `st.text_area` with fixed height
- Kept the model fully local – no HuggingFace token or external API needed
- Verified that a multi-chunk transcript produces a coherent combined summary

## Next tasks (for reference)

- Task 4: Quality checklist for the summary
- Task 5: Markdown → PDF downloadable report
- Task 6: Deploy to Streamlit Community Cloud + handover document

## License

MIT – built as part of an AI internship project for Mehfil Event Planners.

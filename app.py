"""
Mehfil Event Planners - AI-Powered Meeting Transcript Summariser
Task 3: Integrate open-source HuggingFace summarisation model
"""

import streamlit as st
from transformers import pipeline
import re
from typing import List


# ---------------------------------------------------------------------------
# Model loading (cached so it only happens once per session / process)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_summarizer():
    """
    Load the HuggingFace summarisation pipeline once.
    Model: sshleifer/distilbart-cnn-12-6 (lightweight, no API key required)
    """
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1,  # CPU – works on typical laptops without GPU
    )
    return summarizer


# ---------------------------------------------------------------------------
# Chunking helpers (from Task 2)
# ---------------------------------------------------------------------------
def chunk_text(text: str, max_words: int = 400) -> List[str]:
    """
    Split a long transcript into roughly equal chunks of ~max_words words.
    Tries to break on sentence boundaries when possible.
    """
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    if len(words) <= max_words:
        return [text]

    chunks = []
    current = []

    for word in words:
        current.append(word)
        if len(current) >= max_words:
            # Prefer breaking at the end of a sentence
            chunk_str = " ".join(current)
            # Look for the last sentence end in the last 50 words
            last_period = max(
                chunk_str.rfind(". "),
                chunk_str.rfind("? "),
                chunk_str.rfind("! "),
            )
            if last_period > len(chunk_str) // 2:
                chunks.append(chunk_str[: last_period + 1].strip())
                leftover = chunk_str[last_period + 1 :].strip()
                current = leftover.split() if leftover else []
            else:
                chunks.append(chunk_str.strip())
                current = []

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------
def summarise_chunks(chunks: List[str], summarizer) -> str:
    """
    Run the summarisation pipeline on every chunk and concatenate the results.
    """
    partial_summaries = []

    progress = st.progress(0, text="Summarising chunks…")
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        # distilbart-cnn has a practical max of ~1024 tokens; we already chunked
        # by words so this stays safe. min/max length keep outputs concise.
        result = summarizer(
            chunk,
            max_length=150,
            min_length=30,
            do_sample=False,
            truncation=True,
        )
        summary_text = result[0]["summary_text"].strip()
        partial_summaries.append(summary_text)

        progress.progress((i + 1) / total, text=f"Summarised chunk {i + 1}/{total}")

    progress.empty()
    return "\n\n".join(partial_summaries)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Mehfil Meeting Summariser",
        page_icon="📋",
        layout="wide",
    )

    st.title("📋 Mehfil Event Planners – Meeting Transcript Summariser")
    st.markdown(
        "Upload a raw meeting transcript and get a concise, actionable summary "
        "with key points automatically extracted by an open-source model."
    )

    # Sidebar info
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            **Client:** Mehfil Event Planners  
            **Task 3:** Integrate open-source summarisation model  

            Model used:  
            `sshleifer/distilbart-cnn-12-6`  
            (runs fully locally – no API key required)
            """
        )
        st.markdown("---")
        st.caption("Built with Streamlit + HuggingFace Transformers")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload meeting transcript (.txt)",
        type=["txt"],
        help="Plain-text transcript of a vendor or client meeting",
    )

    # Determine source of transcript
    raw_text = None
    if uploaded_file is not None:
        raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.session_state["transcript"] = raw_text
    elif "transcript" in st.session_state:
        raw_text = st.session_state["transcript"]

    if raw_text is None:
        st.info("👆 Upload a transcript to get started. You can also try the sample below.")
        if st.button("Load sample transcript"):
            sample = """Meeting Transcript – Mehfil Event Planners
Date: 28 October 2026
Attendees: Ayesha (Client Coordinator), Imran (Vendor Manager), Sara (Decor Lead), Bilal (Catering)

Ayesha: Good morning everyone. Let’s finalise the timeline for the Khan wedding on 15 November.
Imran: The venue has confirmed availability from 2 pm. We need the décor team on site by 10 am for setup.
Sara: Agreed. Flower arrangements will arrive at 9:30. We still need a decision on the centrepiece style – classic white roses or mixed pastel?
Ayesha: Client prefers classic white roses with gold accents. Budget for décor is 180,000 PKR.
Sara: That works. I’ll lock the order today.
Bilal: Catering menu – we proposed a four-course Pakistani + continental fusion. Client asked for two vegetarian options and a live station for desserts.
Ayesha: Approved. Also add a mocktail bar. Guest count is 350.
Bilal: Understood. I’ll revise the quote by tomorrow morning.
Imran: Photography package – we shortlisted three vendors. Client liked the second one (Studio Lumina) but wants a second shooter included.
Ayesha: Let’s confirm Studio Lumina with the extra shooter. Payment schedule: 40 % advance, 30 % one week before, 30 % on the day.
Imran: I’ll send the contract this afternoon.
Sara: One more item – lighting. Client wants warm fairy lights and a spotlight on the stage. Extra cost is 25,000 PKR.
Ayesha: Approved within the overall budget. Please update the run-of-show document and share it by Friday.
All: Agreed. Meeting adjourned.
"""
            st.session_state["transcript"] = sample
            st.rerun()
        return

    # Preview
    st.subheader("Transcript Preview")
    st.text_area(
        "Raw transcript",
        value=raw_text,
        height=250,
        disabled=True,
        label_visibility="collapsed",
    )

    # Chunking
    chunks = chunk_text(raw_text, max_words=400)
    st.caption(f"Transcript split into **{len(chunks)}** chunk(s) for processing.")

    # Summarise button
    if st.button("Generate Summary", type="primary"):
        with st.spinner("Loading model (first run may take a minute)…"):
            summarizer = load_summarizer()

        combined = summarise_chunks(chunks, summarizer)

        st.subheader("📝 Combined Meeting Summary")
        st.text_area(
            "Summary",
            value=combined,
            height=350,
            disabled=True,
            label_visibility="collapsed",
            key="final_summary",
        )

        st.success(
            f"Summary generated from {len(chunks)} chunk(s). "
            "You can copy the text above or proceed to later tasks for PDF export."
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Process ML Prague conference video: transcribe the talk and extract slide text,
align them by timestamp, save aligned.json, then ingest into Pinecone.

Phases are resumable — each one is skipped if its output file already exists.

Usage:
    python scripts/process_mlprague.py [options]

Options:
    --skip-transcription   Load transcript.json instead of running WhisperX
    --skip-slides          Load slide_texts.json instead of running OCR
    --skip-ingest          Skip Pinecone upload
    --whisper-model        WhisperX model size (default: large-v3)
"""

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "mlprague"
TALK_VIDEO = DATA_DIR / "conference_video" / "39063864_13197_nZvIrjW7_720p_dvr.mp4"
SLIDE_VIDEO = DATA_DIR / "slide_video" / "39063864--slides_13197_720p.mp4"
FRAMES_DIR = DATA_DIR / "slide_frames"
TRANSCRIPT_PATH = DATA_DIR / "transcript.json"
SLIDE_TEXTS_PATH = DATA_DIR / "slide_texts.json"
ALIGNED_PATH = DATA_DIR / "aligned.json"
TALKS_PATH = DATA_DIR / "talks.json"


# ---------------------------------------------------------------------------
# Phase 1: Transcription with WhisperX
# ---------------------------------------------------------------------------


def transcribe(
    video_path: Path, output_path: Path, model_name: str = "large-v3"
) -> dict:
    if output_path.exists():
        logger.info("Transcript found at %s — skipping WhisperX", output_path)
        return json.loads(output_path.read_text())

    logger.info("Loading WhisperX model '%s'…", model_name)
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    logger.info("Device: %s, compute_type: %s", device, compute_type)

    model = whisperx.load_model(model_name, device, compute_type=compute_type)

    logger.info("Loading audio from %s…", video_path)
    audio = whisperx.load_audio(str(video_path))

    logger.info("Transcribing (this will take a while for long videos)…")
    result = model.transcribe(audio, batch_size=8)

    logger.info("Aligning transcript to audio…")
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("Transcript saved → %s", output_path)
    return result


# ---------------------------------------------------------------------------
# Phase 2: Slide frame extraction via sampling + perceptual hash deduplication
#
# Scene-change detection (ffmpeg select filter) doesn't work reliably for
# slide recordings because slides share the same background template, keeping
# the per-frame difference below any useful threshold.
#
# Instead we:
#   1. Sample 1 frame per SAMPLE_INTERVAL seconds with ffmpeg's fps filter.
#   2. Compute a perceptual hash (pHash) for each raw frame with imagehash.
#   3. Keep only frames whose pHash differs from the previous kept frame by
#      more than HASH_DISTANCE bits — these are genuine slide transitions.
# ---------------------------------------------------------------------------

SAMPLE_INTERVAL = 5.0  # seconds between sampled frames
HASH_DISTANCE_THRESHOLD = 8  # hamming bits; slides differ by 20-40 bits


def extract_slide_frames(video_path: Path, frames_dir: Path) -> list[dict]:
    """
    Sample frames from the slide video and deduplicate with pHash.
    Returns [{frame_path, timestamp}, ...] for every unique slide.
    Fully resumable: raw frames and the unique-slide index are cached.
    """
    import shutil

    import imagehash
    from PIL import Image

    raw_dir = frames_dir / "raw"
    unique_dir = frames_dir / "unique"
    index_file = frames_dir / "index.json"

    if index_file.exists():
        logger.info("Slide index already exists — loading from %s", index_file)
        return json.loads(index_file.read_text())

    # ---- Step 1: extract raw frames ----
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing_raw = sorted(raw_dir.glob("frame_*.jpg"))

    if not existing_raw:
        fps = 1.0 / SAMPLE_INTERVAL
        logger.info(
            "Extracting 1 frame per %.0fs from %s (this may take several minutes)…",
            SAMPLE_INTERVAL,
            video_path,
        )
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
            "-q:v",
            "5",
            str(raw_dir / "frame_%06d.jpg"),
            "-y",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("ffmpeg stderr:\n%s", result.stderr[-3000:])
            raise RuntimeError("ffmpeg frame extraction failed")
        existing_raw = sorted(raw_dir.glob("frame_*.jpg"))

    logger.info("Raw frames available: %d", len(existing_raw))

    # ---- Step 2: pHash deduplication ----
    unique_dir.mkdir(parents=True, exist_ok=True)
    unique_frames: list[dict] = []
    last_hash = None

    for i, frame_path in enumerate(existing_raw):
        timestamp = i * SAMPLE_INTERVAL
        img = Image.open(frame_path)
        phash = imagehash.phash(img)

        if last_hash is None or (phash - last_hash) > HASH_DISTANCE_THRESHOLD:
            dest = (
                unique_dir / f"slide_{len(unique_frames):05d}_t{int(timestamp):07d}.jpg"
            )
            shutil.copy2(frame_path, dest)
            unique_frames.append({"frame_path": str(dest), "timestamp": timestamp})
            last_hash = phash

        if i % 200 == 0:
            logger.info(
                "  dedup %d/%d frames → %d unique slides so far",
                i,
                len(existing_raw),
                len(unique_frames),
            )

    logger.info(
        "Deduplication done: %d unique slides from %d raw frames",
        len(unique_frames),
        len(existing_raw),
    )
    index_file.write_text(json.dumps(unique_frames, indent=2))
    return unique_frames


# ---------------------------------------------------------------------------
# Phase 3: Slide text extraction via Mistral Pixtral vision
# ---------------------------------------------------------------------------


def extract_slide_texts(frames: list[dict], output_path: Path) -> list[dict]:
    """
    Use Mistral Pixtral vision to extract text from each slide frame.
    Saves results incrementally so a restart resumes where it left off.
    """
    # Load any partial progress
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        if len(existing) == len(frames):
            logger.info(
                "All %d slide texts already extracted — skipping Pixtral", len(existing)
            )
            return existing
        logger.info("Resuming slide OCR from frame %d/%d…", len(existing), len(frames))
        done_indices = {e["slide_index"] for e in existing}
    else:
        existing = []
        done_indices = set()

    from mistralai.client.sdk import Mistral

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set in environment / .env")

    client = Mistral(api_key=api_key)

    for i, frame_info in enumerate(frames):
        if i in done_indices:
            continue

        frame_path = frame_info["frame_path"]
        timestamp = frame_info["timestamp"]
        logger.info(
            "Slide %d/%d  (t=%.1fs)  %s",
            i + 1,
            len(frames),
            timestamp,
            Path(frame_path).name,
        )

        with open(frame_path, "rb") as fh:
            img_b64 = base64.standard_b64encode(fh.read()).decode()

        for attempt in range(5):
            try:
                response = client.chat.complete(
                    model="pixtral-12b-2409",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/jpeg;base64,{img_b64}",
                                },
                                {
                                    "type": "text",
                                    "text": (
                                        "Extract all text from this presentation slide exactly as it appears. "
                                        "Preserve the hierarchy (title, bullet points, sub-bullets) using plain text. "
                                        "If the slide contains no meaningful text, return an empty string. "
                                        "Return only the extracted text — no commentary."
                                    ),
                                },
                            ],
                        }
                    ],
                )
                break
            except Exception as exc:
                if attempt == 4:
                    raise
                wait = 2**attempt * 5
                logger.warning("Request failed (%s), retrying in %ds…", exc, wait)
                time.sleep(wait)

        time.sleep(1.5)
        text = response.choices[0].message.content.strip()
        existing.append(
            {
                "slide_index": i,
                "timestamp": timestamp,
                "frame_path": frame_path,
                "text": text,
            }
        )

        # Save after every slide so we can resume on interruption
        output_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))

    logger.info("Slide texts saved → %s", output_path)
    return existing


# ---------------------------------------------------------------------------
# Phase 4: Align transcript segments with slides
# ---------------------------------------------------------------------------


def align(transcript: dict, slide_texts: list[dict]) -> list[dict]:
    """
    For each slide, collect all transcript segments whose time window overlaps
    with the slide's display window (slide_start … next_slide_start).
    """
    segments = transcript.get("segments", [])
    slides = sorted(slide_texts, key=lambda s: s["timestamp"])

    aligned = []
    for i, slide in enumerate(slides):
        slide_start = slide["timestamp"]
        slide_end = slides[i + 1]["timestamp"] if i + 1 < len(slides) else float("inf")

        matching = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in segments
            if seg["start"] < slide_end and seg["end"] > slide_start
        ]

        aligned.append(
            {
                "slide_index": i,
                "slide_timestamp": slide_start,
                "slide_text": slide["text"],
                "transcript_segments": matching,
                "transcript_text": " ".join(s["text"].strip() for s in matching),
            }
        )

    return aligned


# ---------------------------------------------------------------------------
# Phase 5: Ingest into Pinecone
# ---------------------------------------------------------------------------


def detect_conference_location(slide_texts: list[dict]) -> tuple[str, str]:
    """Extract city and country from conference name slides (e.g. 'Machine Learning Prague')."""
    import re
    from geopy.geocoders import Nominatim

    for slide in slide_texts[:10]:
        text = slide["text"].strip()
        m = re.search(r"Machine Learning\s+([A-Z][a-z]+)", text)
        if m:
            city = m.group(1)
            try:
                geolocator = Nominatim(user_agent="mlprague-ingest")
                location = geolocator.geocode(city, exactly_one=True, language="en")
                country = (
                    location.raw.get("display_name", "").split(",")[-1].strip()
                    if location
                    else ""
                )
            except Exception as exc:
                logger.warning("Geocoding failed for '%s': %s", city, exc)
                country = ""
            logger.info("Detected conference location: %s, %s", city, country)
            return city, country

    logger.warning("Could not detect conference city from slides — location unknown")
    return "", ""


def _find_talk(slide_index: int, talks: list[dict]) -> dict:
    for talk in talks:
        if talk["slide_start"] <= slide_index <= talk["slide_end"]:
            return talk
    return {}


def ingest_to_pinecone(
    aligned: list[dict], talks: list[dict], city: str, country: str
) -> None:
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pinecone import Pinecone, ServerlessSpec

    docs = []
    for entry in aligned:
        slide_text = entry["slide_text"].strip()
        transcript_text = entry["transcript_text"].strip()
        if not slide_text and not transcript_text:
            continue

        slide_title = slide_text.splitlines()[0] if slide_text else ""
        minutes = int(entry["slide_timestamp"]) // 60
        seconds = int(entry["slide_timestamp"]) % 60
        timestamp_fmt = f"{minutes}:{seconds:02d}"
        talk = _find_talk(entry["slide_index"], talks)

        location_str = f"{city}, {country}" if country else (city or "location unknown")
        header = f"[ML Prague 2026 — {location_str} — slide {entry['slide_index'] + 1}, at {timestamp_fmt}]"
        if talk:
            header += f"\nTalk: {talk['title']}"
            header += f"\nSpeaker: {talk['speaker']}"
            if talk.get("affiliation"):
                header += f" ({talk['affiliation']})"
            if talk.get("date"):
                header += f"\nDate: {talk['date']}"

        content = header + "\n\n"
        if slide_title:
            content += f"Slide title: {slide_title}\n\n"
        if slide_text:
            content += f"[Slide]\n{slide_text}\n\n"
        if transcript_text:
            content += f"[Talk]\n{transcript_text}"

        docs.append(
            Document(
                page_content=content.strip(),
                metadata={
                    "source": "mlprague",
                    "slide_index": entry["slide_index"],
                    "slide_title": slide_title,
                    "timestamp": entry["slide_timestamp"],
                    "timestamp_fmt": timestamp_fmt,
                    "talk_title": talk.get("title", ""),
                    "speaker": talk.get("speaker", ""),
                    "affiliation": talk.get("affiliation", ""),
                    "date": talk.get("date", ""),
                    "city": city,
                    "country": country,
                    "has_slide_text": bool(slide_text),
                    "has_transcript": bool(transcript_text),
                },
            )
        )

    # Add a synthetic overview document so "list all talks" queries have a direct hit
    if talks:
        overview_lines = [
            f"ML Prague 2026 — {city}, {country} — Conference Talk List",
            f"Date: {talks[0].get('date', '')}",
            "",
            "Talks presented at ML Prague 2026:",
        ]
        for t in talks:
            line = f"- \"{t['title']}\" by {t['speaker']}"
            if t.get("affiliation"):
                line += f" ({t['affiliation']})"
            overview_lines.append(line)

        docs.append(
            Document(
                page_content="\n".join(overview_lines),
                metadata={
                    "source": "mlprague",
                    "slide_index": -1,
                    "slide_title": "Conference Overview",
                    "timestamp": 0.0,
                    "timestamp_fmt": "0:00",
                    "talk_title": "",
                    "speaker": "",
                    "affiliation": "",
                    "date": talks[0].get("date", ""),
                    "city": city,
                    "country": country,
                    "has_slide_text": False,
                    "has_transcript": False,
                },
            )
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    logger.info(
        "Ingesting %d chunks from %d slides into Pinecone…", len(chunks), len(docs)
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]

    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws", region=os.getenv("PINECONE_ENV", "us-east-1")
            ),
        )

    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    vectorstore.add_documents(chunks)
    logger.info(
        "Done — %d chunks uploaded to Pinecone index '%s'", len(chunks), index_name
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skip-transcription", action="store_true", help="Use existing transcript.json"
    )
    parser.add_argument(
        "--skip-slides", action="store_true", help="Use existing slide_texts.json"
    )
    parser.add_argument(
        "--skip-ingest", action="store_true", help="Skip Pinecone upload"
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        help="WhisperX model (tiny/base/small/medium/large-v2/large-v3)",
    )
    args = parser.parse_args()

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1
    if not args.skip_transcription:
        transcript = transcribe(
            TALK_VIDEO, TRANSCRIPT_PATH, model_name=args.whisper_model
        )
    else:
        if not TRANSCRIPT_PATH.exists():
            sys.exit(f"--skip-transcription set but {TRANSCRIPT_PATH} not found")
        logger.info("Loading transcript from %s", TRANSCRIPT_PATH)
        transcript = json.loads(TRANSCRIPT_PATH.read_text())

    # Phase 2 + 3
    if not args.skip_slides:
        frames = extract_slide_frames(SLIDE_VIDEO, FRAMES_DIR)
        slide_texts = extract_slide_texts(frames, SLIDE_TEXTS_PATH)
    else:
        if not SLIDE_TEXTS_PATH.exists():
            sys.exit(f"--skip-slides set but {SLIDE_TEXTS_PATH} not found")
        logger.info("Loading slide texts from %s", SLIDE_TEXTS_PATH)
        slide_texts = json.loads(SLIDE_TEXTS_PATH.read_text())

    # Phase 4
    logger.info("Aligning %d slides with transcript…", len(slide_texts))
    aligned = align(transcript, slide_texts)
    ALIGNED_PATH.write_text(json.dumps(aligned, indent=2, ensure_ascii=False))
    logger.info("Aligned data saved → %s  (%d entries)", ALIGNED_PATH, len(aligned))

    # Load talk metadata
    talks = []
    if TALKS_PATH.exists():
        talks = json.loads(TALKS_PATH.read_text())
        logger.info("Loaded %d talks from %s", len(talks), TALKS_PATH)
    else:
        logger.warning(
            "No talks.json found at %s — speaker/title metadata will be empty",
            TALKS_PATH,
        )

    # Detect conference location from slides
    city, country = detect_conference_location(slide_texts)

    # Phase 5
    if not args.skip_ingest:
        ingest_to_pinecone(aligned, talks, city, country)


if __name__ == "__main__":
    main()

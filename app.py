"""
Audiobook Recap v4 – Recap + Position Sync + Dynamic Calibration
"""

import asyncio
import base64
import bisect
import html as html_mod
import io
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import zipfile
from collections import OrderedDict
from contextlib import asynccontextmanager
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ABS_URL = os.getenv("AUDIOBOOKSHELF_URL", "http://localhost:13378")
ABS_TOKEN = os.getenv("AUDIOBOOKSHELF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
SUMMARY_LANGUAGE = os.getenv("SUMMARY_LANGUAGE", "de")
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
DEFAULT_WORDS_PER_PAGE = 250
EPUB_MAX_SIZE_MB = int(os.getenv("EPUB_MAX_SIZE_MB", "100"))
AUTH_USER = os.getenv("AUTH_USER", "")
AUTH_PASS = os.getenv("AUTH_PASS", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audiobook-recap")


# ---------------------------------------------------------------------------
# Auth (optional – set AUTH_USER + AUTH_PASS env vars to enable)
# ---------------------------------------------------------------------------
async def verify_auth(request: Request):
    if not AUTH_USER:
        return
    # Allow health check without auth
    if request.url.path == "/api/health":
        return
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8")
            user, _, password = decoded.partition(":")
            if (secrets.compare_digest(user, AUTH_USER)
                    and secrets.compare_digest(password, AUTH_PASS)):
                return
        except Exception:
            pass
    raise HTTPException(
        status_code=401, detail="Unauthorized",
        headers={"WWW-Authenticate": 'Basic realm="Audiobook Recap"'})


# ---------------------------------------------------------------------------
# Shared HTTP client (lifespan)
# ---------------------------------------------------------------------------
_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(timeout=30)
    yield
    await _http_client.aclose()
    _http_client = None


app = FastAPI(title="Audiobook Recap", version="4.1.0",
              lifespan=lifespan, dependencies=[Depends(verify_auth)])
app.mount("/static", StaticFiles(directory="static"), name="static")

_epub_cache: OrderedDict[str, list[dict]] = OrderedDict()


# ---------------------------------------------------------------------------
# Calibration persistence (thread-safe)
# ---------------------------------------------------------------------------
_calibration_lock = threading.Lock()
_favorites_lock = threading.Lock()


def _calibration_path() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "calibration.json"


def load_calibrations() -> dict:
    with _calibration_lock:
        p = _calibration_path()
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return {}
        return {}


def save_calibrations(data: dict):
    with _calibration_lock:
        _calibration_path().write_text(json.dumps(data, indent=2))


def get_words_per_page(item_id: str) -> Optional[float]:
    cal = load_calibrations()
    entry = cal.get(item_id)
    if entry:
        return entry.get("words_per_page")
    return None


def get_chapter_offset(item_id: str) -> int:
    cal = load_calibrations()
    entry = cal.get(item_id)
    if entry:
        return entry.get("epub_chapter_offset", 0)
    return 0


def set_calibration(item_id: str, words_per_page: float, method: str, details: dict = None):
    with _calibration_lock:
        cal_path = _calibration_path()
        cal = {}
        if cal_path.exists():
            try:
                cal = json.loads(cal_path.read_text())
            except Exception:
                pass
        existing = cal.get(item_id, {})
        existing.update({
            "words_per_page": round(words_per_page, 1),
            "method": method,
            **(details or {}),
        })
        cal[item_id] = existing
        cal_path.write_text(json.dumps(cal, indent=2))


def set_chapter_offset(item_id: str, offset: int):
    with _calibration_lock:
        cal_path = _calibration_path()
        cal = {}
        if cal_path.exists():
            try:
                cal = json.loads(cal_path.read_text())
            except Exception:
                pass
        existing = cal.get(item_id, {})
        existing["epub_chapter_offset"] = max(0, offset)
        cal[item_id] = existing
        cal_path.write_text(json.dumps(cal, indent=2))


def get_anchor_points(item_id: str) -> list[dict]:
    """Get stored multi-point calibration anchors for a book."""
    cal = load_calibrations()
    entry = cal.get(item_id, {})
    return entry.get("anchor_points", [])


def set_anchor_points(item_id: str, points: list[dict]):
    """Store multi-point calibration anchors for a book."""
    with _calibration_lock:
        cal_path = _calibration_path()
        cal = {}
        if cal_path.exists():
            try:
                cal = json.loads(cal_path.read_text())
            except Exception:
                pass
        existing = cal.get(item_id, {})
        existing["anchor_points"] = points
        cal[item_id] = existing
        cal_path.write_text(json.dumps(cal, indent=2))


def get_whisper_anchors(item_id: str) -> list[dict]:
    """Get stored Whisper auto-sync anchors for a book."""
    cal = load_calibrations()
    entry = cal.get(item_id, {})
    return entry.get("whisper_anchors", [])


def set_whisper_anchors(item_id: str, anchors: list[dict]):
    """Store Whisper auto-sync anchors (audio_seconds, char_position, confidence)."""
    with _calibration_lock:
        cal_path = _calibration_path()
        cal = {}
        if cal_path.exists():
            try:
                cal = json.loads(cal_path.read_text())
            except Exception:
                pass
        existing = cal.get(item_id, {})
        existing["whisper_anchors"] = anchors
        existing["method"] = "whisper_auto"
        cal[item_id] = existing
        cal_path.write_text(json.dumps(cal, indent=2))


def _load_whisper_anchors(item_id: str) -> list[tuple[float, int]] | None:
    """Load stored Whisper anchors as (time, char_pos) tuples."""
    anchors = get_whisper_anchors(item_id)
    if not anchors:
        return None
    result = []
    for a in anchors:
        t = a.get("audio_seconds", 0)
        cp = a.get("char_position", 0)
        if t >= 0 and cp >= 0:
            result.append((float(t), int(cp)))
    return result if result else None


def _load_manual_anchors(
    item_id: str,
    epub_ch: list[dict],
    wpp: float,
) -> list[tuple[float, int]] | None:
    """Convert stored (audio_seconds, kindle_page) pairs to (time, char_pos) tuples."""
    points = get_anchor_points(item_id)
    if not points:
        return None
    full_text = _build_full_text(epub_ch, item_id)
    anchors: list[tuple[float, int]] = []
    for p in points:
        t = p["audio_seconds"]
        page = p["kindle_page"]
        # Convert page to approximate character position via word count
        word_target = (page - 1) * wpp
        wc, cp = 0, 0
        for i, ch in enumerate(full_text):
            if ch == ' ':
                wc += 1
            if wc >= word_target:
                cp = i
                break
        else:
            cp = len(full_text) - 1
        anchors.append((float(t), cp))
    return anchors if anchors else None


def _mapping_quality_label(
    mapping: list[tuple[int, int]],
    audio_ch: list[dict],
    epub_ch: list[dict],
    anchors: list[tuple[float, int]] | None = None,
    item_id: str | None = None,
) -> str:
    """Return a human-readable quality label for the chapter mapping."""
    if not audio_ch or not epub_ch:
        return "legacy"
    # Whisper anchors = highest quality automatic mapping
    if item_id and get_whisper_anchors(item_id):
        return "whisper"
    n_audio = len(audio_ch)
    n_matched = len(mapping)
    if n_matched == 0:
        if anchors and len(anchors) >= 3:
            return "wpm"
        return "legacy"
    ratio = n_matched / max(n_audio, 1)
    if ratio >= 0.7:
        return "high"
    if ratio >= 0.3:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Favorites persistence (thread-safe)
# ---------------------------------------------------------------------------
def _favorites_path() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "favorites.json"


def load_favorites() -> list[str]:
    with _favorites_lock:
        p = _favorites_path()
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return []
        return []


def save_favorites(data: list[str]):
    with _favorites_lock:
        _favorites_path().write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class RecapRequest(BaseModel):
    library_item_id: str
    start_minutes: float
    end_minutes: float
    summary_style: Optional[str] = "concise"


class RecapResponse(BaseModel):
    text_excerpt: str
    summary: str
    chapters_covered: list[str]
    duration_seconds: float
    cost_estimate_usd: float


class PositionRequest(BaseModel):
    library_item_id: str
    current_time_seconds: float


class PositionResponse(BaseModel):
    estimated_page: int
    total_pages: int
    percentage: float
    chapter_title: str
    chapter_progress_pct: float
    nearby_text: str
    is_calibrated: bool
    words_per_page: float
    mapping_quality: str = "legacy"  # "high", "medium", "low", "legacy"
    matched_chapters: int = 0


class CalibrateByPageRequest(BaseModel):
    """User says: 'I'm on page X and my audio is at Y seconds'"""
    library_item_id: str
    kindle_page: int
    audio_time_seconds: float


class CalibrateByTotalRequest(BaseModel):
    """User says: 'My Kindle shows X total pages for this book'"""
    library_item_id: str
    total_kindle_pages: int


class CalibrateResponse(BaseModel):
    words_per_page: float
    total_pages: int
    method: str


class TextSearchRequest(BaseModel):
    library_item_id: str
    query: str


class TextSearchResponse(BaseModel):
    audio_timestamp_seconds: float
    audio_timestamp_formatted: str
    chapter_title: str
    confidence: str
    context: str


class PageToAudioRequest(BaseModel):
    library_item_id: str
    page_number: int


class PageToAudioResponse(BaseModel):
    audio_timestamp_seconds: float
    audio_timestamp_formatted: str
    chapter_title: str
    nearby_text: str


class AnchorPoint(BaseModel):
    audio_seconds: float
    kindle_page: int


class MultiAnchorRequest(BaseModel):
    """Multi-point calibration: multiple (audio_time, kindle_page) pairs."""
    library_item_id: str
    points: list[AnchorPoint]


class SetOffsetRequest(BaseModel):
    library_item_id: str
    epub_chapter_offset: int


class WhisperSyncRequest(BaseModel):
    n_samples: int = 10
    language: str = "de"
    segment_duration: int = 20
    force: bool = False  # Re-run even if whisper anchors exist


# ---------------------------------------------------------------------------
# EPUB extraction (unchanged)
# ---------------------------------------------------------------------------
def extract_text_from_epub(epub_bytes: bytes) -> list[dict]:
    chapters = []
    with zipfile.ZipFile(io.BytesIO(epub_bytes)) as zf:
        container = zf.read("META-INF/container.xml")
        ct = ElementTree.fromstring(container)
        ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
        rootfile = ct.find(".//c:rootfile", ns)
        if rootfile is None:
            raise ValueError("No rootfile")
        opf_path = rootfile.attrib["full-path"]
        opf_dir = os.path.dirname(opf_path)
        opf_tree = ElementTree.fromstring(zf.read(opf_path))
        manifest = {}
        for item in opf_tree.findall(".//{http://www.idpf.org/2007/opf}item"):
            manifest[item.attrib.get("id", "")] = {
                "href": item.attrib.get("href", ""),
                "media_type": item.attrib.get("media-type", ""),
            }
        spine_items = []
        for itemref in opf_tree.findall(".//{http://www.idpf.org/2007/opf}itemref"):
            idref = itemref.attrib.get("idref", "")
            if idref in manifest:
                spine_items.append(manifest[idref])
        for idx, item in enumerate(spine_items):
            if "html" not in item["media_type"] and "xml" not in item["media_type"]:
                continue
            href = item["href"]
            fpath = f"{opf_dir}/{href}" if opf_dir else href
            fpath = fpath.replace("%20", " ")
            try:
                html_bytes = zf.read(fpath)
            except KeyError:
                try:
                    html_bytes = zf.read(href)
                except KeyError:
                    continue
            text = _strip_html(html_bytes)
            if not text or len(text.strip()) < 20:
                continue
            title = _extract_heading(html_bytes) or f"Abschnitt {idx + 1}"
            chapters.append({
                "title": title, "text": text.strip(), "index": idx,
                "char_count": len(text.strip()), "word_count": len(text.split()),
            })
    return chapters


def _strip_html(html_bytes: bytes) -> str:
    text = html_bytes.decode("utf-8", errors="replace")
    text = re.sub(r"<head[^>]*>.*?</head>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_mod.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_heading(html_bytes: bytes) -> Optional[str]:
    text = html_bytes.decode("utf-8", errors="replace")
    for tag in ["h1", "h2", "h3", "title"]:
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        if m:
            t = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            if t and len(t) < 200:
                return t
    return None


# ---------------------------------------------------------------------------
# Mapping helpers – anchor-point based (v2)
# ---------------------------------------------------------------------------
_full_text_cache: dict[str, str] = {}
_chapter_mapping_cache: dict[str, list[tuple[int, int]]] = {}
_anchor_cache: dict[str, list[tuple[float, int]]] = {}


def _build_full_text(ec, item_id: str | None = None) -> str:
    if item_id and item_id in _full_text_cache:
        return _full_text_cache[item_id]
    ft = " ".join(ch["text"] for ch in ec)
    if item_id:
        _full_text_cache[item_id] = ft
    return ft


def _total_words(ec): return sum(ch["word_count"] for ch in ec)


def _epub_char_starts(epub_ch) -> list[int]:
    """Cumulative character start position for each EPUB chapter."""
    starts = []
    cum = 0
    for e in epub_ch:
        starts.append(cum)
        cum += e["char_count"] + 1  # +1 for space between chapters
    return starts


# --- Legacy index-based mapping (used as fallback) ---

def _audio_to_epub_idx(audio_idx: int, n_audio: int, n_epub: int, offset: int = 0) -> int:
    """Map audio chapter index → EPUB chapter index, scaling proportionally."""
    if n_epub <= 0:
        return 0
    if n_audio <= 1:
        idx = offset
    elif n_audio == n_epub:
        idx = audio_idx + offset
    else:
        idx = round(audio_idx * max(n_epub - 1, 1) / max(n_audio - 1, 1)) + offset
    return max(0, min(idx, n_epub - 1))


def _epub_to_audio_idx(epub_idx: int, n_audio: int, n_epub: int, offset: int = 0) -> int:
    """Reverse: map EPUB chapter index → audio chapter index."""
    if n_audio <= 0:
        return 0
    adj = epub_idx - offset
    if n_audio == n_epub:
        idx = adj
    elif n_epub <= 1:
        idx = 0
    else:
        idx = round(adj * max(n_audio - 1, 1) / max(n_epub - 1, 1))
    return max(0, min(idx, n_audio - 1))


# --- Chapter title matching ---

_CHAPTER_PREFIX_RE = re.compile(
    r'^(chapter|kapitel|teil|part|abschnitt|prologue?|epilogue?|prolog|epilog)'
    r'\s*[\d.:;,\-–—ivxlcIVXLC]*\s*[-–—:.\s]*',
    re.IGNORECASE,
)
_CHAPTER_NUM_RE = re.compile(
    r'(?:chapter|kapitel|teil|part|abschnitt)\s*(\d+)',
    re.IGNORECASE,
)
_ROMAN_RE = re.compile(
    r'(?:chapter|kapitel|teil|part|abschnitt)\s+([IVXLC]+)\b',
    re.IGNORECASE,
)
_ROMAN_MAP = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}


def _roman_to_int(s: str) -> int | None:
    """Convert a simple Roman numeral to int, or None."""
    s = s.upper()
    if not all(c in _ROMAN_MAP for c in s):
        return None
    total, prev = 0, 0
    for c in reversed(s):
        v = _ROMAN_MAP[c]
        total += v if v >= prev else -v
        prev = v
    return total if total > 0 else None


def _extract_chapter_number(title: str) -> int | None:
    """Extract chapter number from title (Arabic or Roman numerals)."""
    m = _CHAPTER_NUM_RE.search(title)
    if m:
        return int(m.group(1))
    m = _ROMAN_RE.search(title)
    if m:
        return _roman_to_int(m.group(1))
    return None


def _normalize_title(title: str) -> str:
    """Normalize a chapter title for fuzzy comparison."""
    t = title.strip()
    if not t:
        return ""
    # Remove chapter-prefix patterns ("Chapter 5:", "Kapitel XII –", …)
    t = _CHAPTER_PREFIX_RE.sub("", t).strip()
    # Collapse whitespace, lowercase
    t = re.sub(r'\s+', ' ', t).lower()
    # Strip leading/trailing punctuation
    t = t.strip(' \t\n\r-–—:.,;!?()[]""„"\'')
    return t


def _title_similarity(a: str, b: str) -> float:
    """Similarity score between two chapter titles (0.0–1.0)."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Containment bonus
    if a in b or b in a:
        return max(0.85, SequenceMatcher(None, a, b).ratio())
    return SequenceMatcher(None, a, b).ratio()


def _match_chapters_by_title(
    audio_ch: list[dict],
    epub_ch: list[dict],
    offset: int = 0,
) -> list[tuple[int, int]]:
    """Match audio chapters to EPUB chapters by title similarity.

    Uses two strategies:
    1. Fuzzy title matching (normalized titles, similarity >= 0.55)
    2. Chapter number matching (e.g. "Kapitel 3" ↔ "Kapitel 3: Title")

    Returns a list of (audio_idx, epub_idx) pairs, ordered by audio_idx.
    Enforces monotonicity: if audio chapter i maps to EPUB chapter j,
    then audio chapter i+1 must map to EPUB chapter >= j.
    """
    THRESHOLD = 0.55

    epub_titles = [
        (i, _normalize_title(e["title"]))
        for i, e in enumerate(epub_ch)
        if i >= offset
    ]
    audio_titles = [
        (i, _normalize_title(a.get("title", "")))
        for i, a in enumerate(audio_ch)
    ]

    # Strategy 1: Fuzzy title matching on normalized titles
    raw_matches: list[tuple[float, int, int]] = []
    for ai, at in audio_titles:
        if not at:
            continue
        for ei, et in epub_titles:
            if not et:
                continue
            score = _title_similarity(at, et)
            if score >= THRESHOLD:
                raw_matches.append((score, ai, ei))

    # Strategy 2: Chapter number matching (for titles like "Kapitel 3" with no subtitle)
    audio_nums: dict[int, int] = {}  # audio_idx → chapter_number
    epub_nums: dict[int, int] = {}  # epub_idx → chapter_number
    for i, a in enumerate(audio_ch):
        n = _extract_chapter_number(a.get("title", ""))
        if n is not None:
            audio_nums[i] = n
    for i, e in enumerate(epub_ch):
        if i < offset:
            continue
        n = _extract_chapter_number(e["title"])
        if n is not None:
            epub_nums[i] = n

    # Build reverse lookup: number → epub_idx
    num_to_epub: dict[int, int] = {}
    for ei, num in epub_nums.items():
        if num not in num_to_epub:  # first occurrence wins
            num_to_epub[num] = ei

    for ai, num in audio_nums.items():
        if num in num_to_epub:
            ei = num_to_epub[num]
            # Add with score 0.75 (lower than exact title match but above threshold)
            raw_matches.append((0.75, ai, ei))

    # Sort by score descending, greedily select while maintaining monotonicity
    raw_matches.sort(key=lambda x: -x[0])
    used_audio: set[int] = set()
    used_epub: set[int] = set()
    selected: list[tuple[int, int]] = []

    for score, ai, ei in raw_matches:
        if ai in used_audio or ei in used_epub:
            continue
        selected.append((ai, ei))
        used_audio.add(ai)
        used_epub.add(ei)

    # Sort by audio index
    selected.sort(key=lambda x: x[0])

    # Enforce strict monotonicity (remove any inversions)
    monotonic: list[tuple[int, int]] = []
    for ai, ei in selected:
        if not monotonic or ei > monotonic[-1][1]:
            monotonic.append((ai, ei))

    return monotonic


def _match_by_boundary_alignment(
    audio_ch: list[dict],
    epub_ch: list[dict],
    offset: int = 0,
) -> list[tuple[int, int]]:
    """Match chapter boundaries by cumulative fractional position.

    If audio chapter 3 starts at 30% of total duration and EPUB chapter 5
    starts at 31% of total words, they likely correspond. This works even
    when chapter titles are completely different (e.g. "Track 01" vs
    "Der Anfang").

    The tolerance adapts: stricter when both sides have similar chapter
    counts, looser when they differ a lot.
    """
    if not audio_ch or not epub_ch:
        return []
    total_dur = audio_ch[-1].get("end", 0)
    epub_chs = epub_ch[offset:]
    total_words = sum(e["word_count"] for e in epub_chs)
    if total_dur <= 0 or total_words <= 0:
        return []

    # Compute cumulative fraction at each chapter START
    audio_fracs: list[tuple[float, int]] = []  # (fraction, original_idx)
    for i, a in enumerate(audio_ch):
        audio_fracs.append((a["start"] / total_dur, i))

    epub_fracs: list[tuple[float, int]] = []  # (fraction, original_idx)
    cum_w = 0
    for i, e in enumerate(epub_chs):
        epub_fracs.append((cum_w / total_words, i + offset))
        cum_w += e["word_count"]

    # Adaptive tolerance based on chapter count ratio
    n_ratio = min(len(audio_ch), len(epub_chs)) / max(len(audio_ch), len(epub_chs))
    tolerance = 0.03 if n_ratio > 0.8 else 0.06 if n_ratio > 0.5 else 0.10

    # Greedy matching: for each audio boundary, find closest EPUB boundary
    used_epub: set[int] = set()
    matches: list[tuple[float, int, int]] = []  # (distance, audio_idx, epub_idx)

    for a_frac, ai in audio_fracs:
        best_dist, best_ei = tolerance, -1
        for e_frac, ei in epub_fracs:
            if ei in used_epub:
                continue
            dist = abs(a_frac - e_frac)
            if dist < best_dist:
                best_dist, best_ei = dist, ei
        if best_ei >= 0:
            matches.append((best_dist, ai, best_ei))
            used_epub.add(best_ei)

    # Sort by audio index
    matches.sort(key=lambda x: x[1])

    # Enforce strict monotonicity
    monotonic: list[tuple[int, int]] = []
    for _, ai, ei in matches:
        if not monotonic or ei > monotonic[-1][1]:
            monotonic.append((ai, ei))

    return monotonic


def _build_wpm_anchors(
    audio_ch: list[dict],
    epub_ch: list[dict],
    total_dur: float,
    offset: int = 0,
) -> list[tuple[float, int]]:
    """Build anchor points using WPM estimation (no chapter matching needed).

    Assumes constant narration speed. Creates an anchor at each EPUB chapter
    boundary based on the estimated reading position at that word count.

    This is more accurate than proportional index mapping because it accounts
    for chapters of different lengths.
    """
    epub_chs = epub_ch[offset:]
    total_words = sum(e["word_count"] for e in epub_chs)
    if total_words <= 0 or total_dur <= 0:
        return []

    wpm = total_words / (total_dur / 60)  # words per minute

    # Pre-offset char count
    pre_offset_chars = sum(e["char_count"] + 1 for e in epub_ch[:offset])

    anchors: list[tuple[float, int]] = [(0.0, 0)]
    cum_words = 0
    cum_chars = pre_offset_chars
    for e in epub_chs:
        cum_words += e["word_count"]
        cum_chars += e["char_count"] + 1
        estimated_time = (cum_words / wpm) * 60  # seconds
        # Clamp to total duration
        estimated_time = min(estimated_time, total_dur)
        anchors.append((estimated_time, cum_chars))

    return anchors


def _get_chapter_mapping(
    audio_ch: list[dict],
    epub_ch: list[dict],
    offset: int = 0,
    item_id: str | None = None,
) -> list[tuple[int, int]]:
    """Get (cached) chapter mapping for a book.

    Tries strategies in order:
    1. Title matching (fuzzy + number-based)
    2. Boundary alignment (cumulative fraction matching)
    """
    cache_key = f"{item_id}:{offset}" if item_id else None
    if cache_key and cache_key in _chapter_mapping_cache:
        return _chapter_mapping_cache[cache_key]

    # Strategy 1: Title matching
    mapping = _match_chapters_by_title(audio_ch, epub_ch, offset)

    # Strategy 2: If title matching found < 2 pairs, try boundary alignment
    if len(mapping) < 2 and len(audio_ch) >= 3 and len(epub_ch) >= 3:
        boundary_mapping = _match_by_boundary_alignment(audio_ch, epub_ch, offset)
        if len(boundary_mapping) > len(mapping):
            mapping = boundary_mapping

    if cache_key:
        _chapter_mapping_cache[cache_key] = mapping
    return mapping


# --- Anchor-point system ---

def _build_anchor_points(
    audio_ch: list[dict],
    epub_ch: list[dict],
    mapping: list[tuple[int, int]],
    total_dur: float,
    manual_anchors: list[tuple[float, int]] | None = None,
) -> list[tuple[float, int]]:
    """Build sorted (time, char_pos) anchor points from chapter mapping.

    Each matched chapter pair contributes two anchors (start & end).
    Manual anchors (from multi-point calibration) are merged in.
    Book boundaries (0,0) and (total_dur, total_chars) are always included.
    """
    char_starts = _epub_char_starts(epub_ch)
    total_chars = sum(e["char_count"] + 1 for e in epub_ch)

    anchors: list[tuple[float, int]] = [(0.0, 0)]

    for ai, ei in mapping:
        a = audio_ch[ai]
        t_start = a["start"]
        t_end = a.get("end", a["start"])
        c_start = char_starts[ei]
        c_end = char_starts[ei] + epub_ch[ei]["char_count"]
        anchors.append((t_start, c_start))
        anchors.append((t_end, c_end))

    if manual_anchors:
        anchors.extend(manual_anchors)

    anchors.append((total_dur, total_chars))

    # Sort by time, deduplicate (keep last for same time)
    anchors.sort(key=lambda x: (x[0], x[1]))
    cleaned: list[tuple[float, int]] = [anchors[0]]
    for t, c in anchors[1:]:
        if t > cleaned[-1][0] + 0.5:  # >0.5s gap = new point
            # Ensure char_pos is also monotonically increasing
            if c >= cleaned[-1][1]:
                cleaned.append((t, c))
            else:
                # Non-monotonic char_pos → skip (bad match)
                pass
        else:
            # Same time point → update char_pos if larger
            if c > cleaned[-1][1]:
                cleaned[-1] = (t, c)

    return cleaned


def _interpolate_time_to_char(anchors: list[tuple[float, int]], time_sec: float) -> int:
    """Piecewise-linear interpolation: audio time → character position."""
    if not anchors:
        return 0
    if time_sec <= anchors[0][0]:
        return anchors[0][1]
    if time_sec >= anchors[-1][0]:
        return anchors[-1][1]

    # Binary search for the right interval
    times = [a[0] for a in anchors]
    idx = bisect.bisect_right(times, time_sec) - 1
    idx = max(0, min(idx, len(anchors) - 2))

    t0, c0 = anchors[idx]
    t1, c1 = anchors[idx + 1]
    dt = t1 - t0
    if dt <= 0:
        return c0
    ratio = (time_sec - t0) / dt
    return int(c0 + ratio * (c1 - c0))


def _interpolate_char_to_time(anchors: list[tuple[float, int]], char_pos: int) -> float:
    """Piecewise-linear interpolation: character position → audio time."""
    if not anchors:
        return 0.0
    if char_pos <= anchors[0][1]:
        return anchors[0][0]
    if char_pos >= anchors[-1][1]:
        return anchors[-1][0]

    # Binary search on char positions
    chars = [a[1] for a in anchors]
    idx = bisect.bisect_right(chars, char_pos) - 1
    idx = max(0, min(idx, len(anchors) - 2))

    t0, c0 = anchors[idx]
    t1, c1 = anchors[idx + 1]
    dc = c1 - c0
    if dc <= 0:
        return t0
    ratio = (char_pos - c0) / dc
    return t0 + ratio * (t1 - t0)


def _get_anchors(
    audio_ch: list[dict],
    epub_ch: list[dict],
    total_dur: float,
    offset: int = 0,
    item_id: str | None = None,
    manual_anchors: list[tuple[float, int]] | None = None,
) -> tuple[list[tuple[float, int]], list[tuple[int, int]]]:
    """Get anchor points and chapter mapping (with caching).

    Tries in order (merging where possible):
    1. Chapter mapping anchors (title + boundary matching)
    2. Whisper auto-sync anchors (persistent, from calibration.json)
    3. Manual anchors (from multi-point calibration)
    4. WPM-based anchors (word-count proportional, no matching needed)
    5. Simple start/end anchors (pure linear)

    Returns (anchors, mapping).
    """
    mapping = _get_chapter_mapping(audio_ch, epub_ch, offset, item_id)
    cache_key = f"{item_id}:{offset}" if item_id else None

    # Load whisper anchors (persistent, stored as char positions directly)
    whisper_anchors = _load_whisper_anchors(item_id) if item_id else None

    # Don't cache if manual or whisper anchors exist (they can change)
    has_extra = manual_anchors or whisper_anchors
    if cache_key and not has_extra and cache_key in _anchor_cache:
        return _anchor_cache[cache_key], mapping

    # Merge all anchor sources
    combined_manual = []
    if whisper_anchors:
        combined_manual.extend(whisper_anchors)
    if manual_anchors:
        combined_manual.extend(manual_anchors)

    anchors = _build_anchor_points(
        audio_ch, epub_ch, mapping, total_dur,
        combined_manual if combined_manual else None,
    )

    # If still too few anchors (just start+end), use WPM-based fallback
    if len(anchors) <= 2 and len(epub_ch) >= 3 and not has_extra:
        wpm_anchors = _build_wpm_anchors(audio_ch, epub_ch, total_dur, offset)
        if len(wpm_anchors) > len(anchors):
            anchors = wpm_anchors

    if cache_key and not has_extra:
        _anchor_cache[cache_key] = anchors

    return anchors, mapping


def _find_audio_chapter_at_time(audio_ch: list[dict], time_sec: float) -> tuple[str, float]:
    """Find audio chapter title and progress % at a given time."""
    if not audio_ch:
        return "(unbekannt)", 0.0
    cur, ci = audio_ch[0], 0
    for i, a in enumerate(audio_ch):
        if a["start"] <= time_sec < a.get("end", a["start"]):
            cur, ci = a, i
            break
        if a["start"] <= time_sec:
            cur, ci = a, i
    cs = cur["start"]
    ce = cur.get("end", cs)
    cd = ce - cs
    pct = max(0, min(1, (time_sec - cs) / max(cd, 1))) * 100
    return cur.get("title", f"Kapitel {ci + 1}"), pct


# --- Main mapping functions (v2: anchor-based with fallback) ---

def _time_to_char_position(
    audio_ch, epub_ch, time_sec, total_dur,
    offset: int = 0, item_id: str | None = None,
    manual_anchors: list[tuple[float, int]] | None = None,
):
    """Map audio time → character position in EPUB full text.

    Uses anchor-point interpolation when enough anchors are available
    (from chapter matching, boundary alignment, or WPM estimation).
    Falls back to legacy proportional mapping only as last resort.

    Returns: (char_offset, chapter_title, chapter_progress_pct)
    """
    full_text = _build_full_text(epub_ch, item_id)

    if not audio_ch:
        ratio = time_sec / max(total_dur, 1)
        return int(len(full_text) * ratio), "(geschätzt)", ratio * 100

    # Try anchor-based mapping
    anchors, mapping = _get_anchors(
        audio_ch, epub_ch, total_dur, offset, item_id, manual_anchors,
    )

    title, pct = _find_audio_chapter_at_time(audio_ch, time_sec)

    if len(anchors) >= 3:
        # Anchor-based interpolation (from title match, boundary align, or WPM)
        char_pos = _interpolate_time_to_char(anchors, time_sec)
        char_pos = max(0, min(char_pos, len(full_text) - 1))
        return char_pos, title, pct

    # Fallback: legacy proportional chapter-index mapping
    cur, ci = audio_ch[0], 0
    for i, a in enumerate(audio_ch):
        if a["start"] <= time_sec < a.get("end", a["start"]):
            cur, ci = a, i
            break
        if a["start"] <= time_sec:
            cur, ci = a, i
    cs, ce = cur["start"], cur.get("end", cur["start"])
    cd = ce - cs
    cp = max(0, min(1, (time_sec - cs) / max(cd, 1)))
    epub_idx = _audio_to_epub_idx(ci, len(audio_ch), len(epub_ch), offset)
    co = 0
    for i, e in enumerate(epub_ch):
        if i < epub_idx:
            co += e["char_count"] + 1
        elif i == epub_idx:
            co += int(e["char_count"] * cp)
            break
    return min(co, len(full_text) - 1), cur.get("title", ""), cp * 100


def _char_position_to_time(
    audio_ch, epub_ch, char_pos, total_dur,
    offset: int = 0, item_id: str | None = None,
    manual_anchors: list[tuple[float, int]] | None = None,
):
    """Map character position → audio time.

    Uses anchor-point interpolation when available, falls back to legacy.

    Returns: (audio_time_seconds, chapter_title)
    """
    if not audio_ch or not epub_ch:
        full = _build_full_text(epub_ch, item_id)
        return (char_pos / max(len(full), 1)) * total_dur, "(geschätzt)"

    # Try anchor-based mapping
    anchors, mapping = _get_anchors(
        audio_ch, epub_ch, total_dur, offset, item_id, manual_anchors,
    )

    if len(anchors) >= 3:
        t = _interpolate_char_to_time(anchors, char_pos)
        t = max(0.0, min(t, total_dur))
        title, _ = _find_audio_chapter_at_time(audio_ch, t)
        return t, title

    # Fallback: legacy proportional mapping
    cum, ti, ep = 0, 0, 0.0
    for i, e in enumerate(epub_ch):
        if cum + e["char_count"] >= char_pos:
            ti, ep = i, (char_pos - cum) / max(e["char_count"], 1)
            break
        cum += e["char_count"] + 1
        ti, ep = i, 1.0
    ai = _epub_to_audio_idx(ti, len(audio_ch), len(epub_ch), offset)
    a = audio_ch[ai]
    return a["start"] + (a.get("end", a["start"]) - a["start"]) * ep, a.get("title", "")


def _format_time(s):
    h, m, sec = int(s // 3600), int(s % 3600 // 60), int(s % 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _chapters_in_char_range(epub_ch, c_start: int, c_end: int) -> list[str]:
    """Find EPUB chapter titles that overlap a character range."""
    starts = _epub_char_starts(epub_ch)
    names = []
    for i, e in enumerate(epub_ch):
        ch_start = starts[i]
        ch_end = ch_start + e["char_count"]
        if ch_end > c_start and ch_start < c_end:
            names.append(e["title"])
    return names if names else ["(unbekannt)"]


def map_time_to_text(
    audio_ch, epub_ch, start_sec, end_sec,
    offset: int = 0, item_id: str | None = None,
):
    """Extract EPUB text for an audio time range.

    Uses anchor-based mapping when available for higher accuracy.
    """
    if not epub_ch:
        raise ValueError("No EPUB chapters")
    if not audio_ch:
        return _pfb(epub_ch, start_sec, end_sec, end_sec)

    total_dur = audio_ch[-1].get("end", 0) if audio_ch else end_sec

    # Try anchor-based mapping
    anchors, mapping = _get_anchors(
        audio_ch, epub_ch, total_dur, offset, item_id,
    )

    if len(anchors) >= 3:
        # Anchor-based: map start/end to char positions via interpolation
        full = _build_full_text(epub_ch, item_id)
        c0 = _interpolate_time_to_char(anchors, start_sec)
        c1 = _interpolate_time_to_char(anchors, end_sec)
        c0 = max(0, min(c0, len(full) - 1))
        c1 = max(c0, min(c1, len(full)))
        if c0 > 0:
            c0 = _snap_to_sentence_start(full, c0)
        if c1 < len(full):
            c1 = _snap_to_sentence_end(full, c1)
        text = full[c0:c1]
        if text.strip():
            names = _chapters_in_char_range(epub_ch, c0, c1)
            return text, names

    # Fallback: legacy per-audio-chapter mapping
    parts, names = [], []
    for i, a in enumerate(audio_ch):
        cs, ce = a.get("start", 0), a.get("end", 0)
        if cs >= end_sec or ce <= start_sec:
            continue
        ei = _audio_to_epub_idx(i, len(audio_ch), len(epub_ch), offset)
        e = epub_ch[ei] if ei < len(epub_ch) else None
        if not e:
            at = a.get("title", "").lower()
            for ec in epub_ch:
                if ec["title"].lower() in at or at in ec["title"].lower():
                    e = ec
                    break
        if not e:
            continue
        d = ce - cs
        if d <= 0:
            continue
        ps, pe = max(0, (start_sec - cs) / d), min(1, (end_sec - cs) / d)
        t = e["text"]
        c0, c1 = int(len(t) * ps), int(len(t) * pe)
        if c0 > 0:
            c0 = _snap_to_sentence_start(t, c0)
        if c1 < len(t):
            c1 = _snap_to_sentence_end(t, c1)
        p = t[c0:c1]
        if p.strip():
            parts.append(p)
            names.append(a.get("title", e["title"]))
    if not parts:
        return _pfb(epub_ch, start_sec, end_sec, total_dur)
    return "\n\n".join(parts), names


def _pfb(ec, ss, es, td):
    if td <= 0: td = 1
    ft = " ".join(c["text"] for c in ec)
    c0, c1 = int(len(ft) * ss / td), int(len(ft) * es / td)
    if c0 > 0:
        c0 = _snap_to_sentence_start(ft, c0)
    if c1 < len(ft):
        c1 = _snap_to_sentence_end(ft, c1)
    return ft[c0:c1], ["(geschätzt)"]


# ---------------------------------------------------------------------------
# ABS API (shared client, status checks, size limits)
# ---------------------------------------------------------------------------
def abs_headers():
    return {"Authorization": f"Bearer {ABS_TOKEN}"}


def _client() -> httpx.AsyncClient:
    if _http_client is None:
        raise RuntimeError("HTTP client not initialized (app not started?)")
    return _http_client


def _find_epub_file(item):
    """Find EPUB from ebookFile or supplementary libraryFiles."""
    media = item.get("media", {})
    ef = media.get("ebookFile")
    if ef and ef.get("metadata", {}).get("ext", "") == ".epub":
        return ef
    for f in item.get("libraryFiles", []):
        if f.get("metadata", {}).get("ext", "") == ".epub":
            return f
    return None


async def get_library_items():
    c = _client()
    r = await c.get(f"{ABS_URL}/api/libraries", headers=abs_headers())
    r.raise_for_status()
    libs = r.json().get("libraries", [])
    item_ids = []
    for lib in libs:
        r2 = await c.get(f"{ABS_URL}/api/libraries/{lib['id']}/items", headers=abs_headers(),
                         params={"limit": 100, "sort": "media.metadata.title"})
        r2.raise_for_status()
        for item in r2.json().get("results", []):
            item_ids.append(item["id"])
    # Fetch item details concurrently to access libraryFiles for epub detection
    detail_resps = await asyncio.gather(*(
        c.get(f"{ABS_URL}/api/items/{iid}", headers=abs_headers(),
              params={"expanded": 1, "include": "progress"}) for iid in item_ids
    ))
    items = []
    cal_data = load_calibrations()
    fav_data = load_favorites()
    for resp in detail_resps:
        resp.raise_for_status()
        item = resp.json()
        media = item.get("media", {}); md = media.get("metadata", {})
        prog = item.get("userMediaProgress") or {}
        cal = cal_data.get(item["id"])
        items.append({
            "id": item["id"], "title": md.get("title", "Unknown"),
            "author": md.get("authorName") or ", ".join(
                a.get("name", "") for a in md.get("authors", [])) or "Unknown",
            "duration": media.get("duration", 0),
            "current_time": prog.get("currentTime", 0),
            "cover": f"/api/cover/{item['id']}",
            "has_epub": _find_epub_file(item) is not None,
            "is_calibrated": cal is not None,
            "is_favorite": item["id"] in fav_data,
        })
    return items


async def get_item_details(item_id):
    c = _client()
    r = await c.get(f"{ABS_URL}/api/items/{item_id}", headers=abs_headers(),
                    params={"expanded": 1})
    r.raise_for_status()
    return r.json()


async def download_epub(item_id, ino):
    c = _client()
    r = await c.get(f"{ABS_URL}/api/items/{item_id}/file/{ino}/download",
                    headers=abs_headers(), timeout=120)
    r.raise_for_status()
    max_bytes = EPUB_MAX_SIZE_MB * 1024 * 1024
    if len(r.content) > max_bytes:
        raise HTTPException(status_code=413,
                            detail=f"EPUB zu groß ({len(r.content) // 1024 // 1024} MB, max {EPUB_MAX_SIZE_MB} MB)")
    return r.content


# ---------------------------------------------------------------------------
# EPUB cache (LRU in-memory + disk persistence)
# ---------------------------------------------------------------------------
def _epub_disk_cache_dir() -> Path:
    d = DATA_DIR / "epub_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_cached_epub_from_disk(item_id: str) -> list[dict] | None:
    p = _epub_disk_cache_dir() / f"{item_id}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _save_cached_epub_to_disk(item_id: str, chapters: list[dict]):
    try:
        p = _epub_disk_cache_dir() / f"{item_id}.json"
        p.write_text(json.dumps(chapters, ensure_ascii=False))
    except Exception as e:
        logger.warning("EPUB-Cache konnte nicht auf Disk geschrieben werden: %s", e)


async def get_epub_chapters(item_id):
    item = await get_item_details(item_id)
    media = item.get("media", {})
    audio_ch = [{"title": ch.get("title", ""), "start": ch.get("start", 0), "end": ch.get("end", 0)}
                for ch in media.get("chapters", [])]
    duration = media.get("duration", 0)
    ef = _find_epub_file(item)
    if not ef:
        raise HTTPException(status_code=400, detail="Kein EPUB gefunden.")
    # 1. LRU in-memory cache
    if item_id in _epub_cache:
        _epub_cache.move_to_end(item_id)
        ec = _epub_cache[item_id]
    else:
        # 2. Disk cache
        ec = _load_cached_epub_from_disk(item_id)
        if not ec:
            # 3. Download + parse
            eb = await download_epub(item_id, ef.get("ino", ef.get("metadata", {}).get("ino", "")))
            ec = extract_text_from_epub(eb)
            if not ec:
                raise HTTPException(status_code=500, detail="EPUB-Text konnte nicht extrahiert werden")
            _save_cached_epub_to_disk(item_id, ec)
        _epub_cache[item_id] = ec
        if len(_epub_cache) > 20:
            evicted_id, _ = _epub_cache.popitem(last=False)
            _full_text_cache.pop(evicted_id, None)
    return ec, audio_ch, duration


async def update_abs_progress(item_id, time_sec, duration):
    c = _client()
    r = await c.patch(f"{ABS_URL}/api/me/progress/{item_id}",
                      headers={**abs_headers(), "Content-Type": "application/json"},
                      json={"currentTime": time_sec, "duration": duration,
                            "progress": time_sec / max(duration, 1), "isFinished": False})
    r.raise_for_status()


# ---------------------------------------------------------------------------
# Whisper auto-sync: audio segment extraction + transcription + text matching
# ---------------------------------------------------------------------------

def _map_global_time_to_file(
    audio_files: list[dict], global_time: float,
) -> tuple[dict, float]:
    """Map a global audio time to a specific file and local offset.

    audio_files must be sorted by index. Each has 'duration' in seconds.
    Returns (audio_file_dict, local_offset_seconds).
    """
    cum = 0.0
    for af in audio_files:
        d = af.get("duration", 0)
        if cum + d > global_time:
            return af, global_time - cum
        cum += d
    # Past end — return last file at its end
    if audio_files:
        last = audio_files[-1]
        return last, last.get("duration", 0)
    raise ValueError("No audio files")


async def _download_audio_file(item_id: str, file_ino: str) -> Path:
    """Download a single audio file from ABS to a temp directory."""
    tmp = Path(tempfile.gettempdir()) / "whisper_sync" / item_id
    tmp.mkdir(parents=True, exist_ok=True)
    cached = tmp / f"{file_ino}"
    if cached.exists() and cached.stat().st_size > 0:
        return cached
    c = _client()
    r = await c.get(
        f"{ABS_URL}/api/items/{item_id}/file/{file_ino}/download",
        headers=abs_headers(), timeout=300,
    )
    r.raise_for_status()
    # Determine extension from content-type or default to .mp3
    ct = r.headers.get("content-type", "")
    ext = ".mp3"
    if "mp4" in ct or "m4a" in ct or "m4b" in ct:
        ext = ".m4b"
    elif "ogg" in ct:
        ext = ".ogg"
    out = tmp / f"{file_ino}{ext}"
    out.write_bytes(r.content)
    return out


async def _extract_audio_segment(
    item_id: str,
    audio_files: list[dict],
    global_time: float,
    duration: float = 20,
) -> Path:
    """Extract a short audio segment at the given global time using ffmpeg."""
    af, local_offset = _map_global_time_to_file(audio_files, global_time)
    ino = af.get("ino", af.get("metadata", {}).get("ino", ""))
    src = await _download_audio_file(item_id, ino)

    tmp = Path(tempfile.gettempdir()) / "whisper_sync" / item_id / "segments"
    tmp.mkdir(parents=True, exist_ok=True)
    out = tmp / f"seg_{global_time:.0f}.mp3"

    proc = await asyncio.to_thread(
        subprocess.run,
        [
            "ffmpeg", "-y", "-ss", str(local_offset), "-t", str(duration),
            "-i", str(src), "-vn", "-acodec", "libmp3lame", "-q:a", "5",
            str(out),
        ],
        capture_output=True, timeout=30,
    )
    if proc.returncode != 0:
        logger.warning("ffmpeg failed: %s", proc.stderr[:500])
        raise RuntimeError(f"ffmpeg error: {proc.stderr[:200]}")
    return out


def _select_sample_positions(
    audio_ch: list[dict], total_dur: float, n_samples: int = 10,
) -> list[float]:
    """Select evenly-spaced sample positions, avoiding chapter boundaries."""
    n_samples = max(3, min(n_samples, 20))
    # Evenly spaced across the book (skip first/last 30 seconds)
    margin = min(30, total_dur * 0.01)
    step = (total_dur - 2 * margin) / max(n_samples - 1, 1)
    positions = [margin + i * step for i in range(n_samples)]

    # Avoid chapter boundaries (±5s) — shift samples away
    boundaries = set()
    for ch in audio_ch:
        boundaries.add(ch["start"])
        if "end" in ch:
            boundaries.add(ch["end"])

    adjusted = []
    for pos in positions:
        for b in boundaries:
            if abs(pos - b) < 5:
                # Shift 8 seconds into the chapter
                pos = b + 8
                break
        pos = max(margin, min(pos, total_dur - margin))
        adjusted.append(pos)

    return adjusted


async def _transcribe_segment(audio_path: Path, language: str = "de") -> str | None:
    """Transcribe a short audio segment using OpenAI Whisper API."""
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        with open(audio_path, "rb") as f:
            resp = await client.audio.transcriptions.create(
                model="whisper-1", file=f, language=language,
            )
        text = resp.text.strip() if hasattr(resp, "text") else str(resp).strip()
        return text if text else None
    except Exception as e:
        logger.warning("Whisper transcription failed for %s: %s", audio_path.name, e)
        return None


async def _transcribe_all(
    segments: list[Path], language: str = "de",
) -> list[str | None]:
    """Transcribe multiple segments in parallel (limited concurrency)."""
    sem = asyncio.Semaphore(5)

    async def _do(path: Path) -> str | None:
        async with sem:
            return await _transcribe_segment(path, language)

    return await asyncio.gather(*(_do(s) for s in segments))


def _normalize_for_matching(text: str) -> list[str]:
    """Normalize text to word list for fuzzy matching."""
    t = text.lower()
    # Remove punctuation except hyphens within words
    t = re.sub(r'[^\w\s\-äöüàáâèéêìíîòóôùúûß]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t.split()


def _find_text_in_epub(
    whisper_text: str,
    full_text: str,
    expected_frac: float,
    window: float = 0.15,
) -> tuple[int, float] | None:
    """Find whisper-transcribed text in EPUB full text.

    Uses a sliding window on word-level with SequenceMatcher.
    Searches within ±window fraction around expected_frac.

    Returns (char_position, confidence_score) or None.
    """
    needle_words = _normalize_for_matching(whisper_text)
    if len(needle_words) < 3:
        return None

    # Determine search region in char space
    total_len = len(full_text)
    center = int(expected_frac * total_len)
    half_win = int(window * total_len)
    search_start = max(0, center - half_win)
    search_end = min(total_len, center + half_win)
    region = full_text[search_start:search_end]

    haystack_words = _normalize_for_matching(region)
    if len(haystack_words) < len(needle_words):
        return None

    needle_str = " ".join(needle_words)
    best_score, best_pos = 0.0, -1
    n = len(needle_words)
    step = max(1, n // 6)  # Slide by ~1/6 of needle length for speed

    # Track character positions: we need to map word index → char offset
    # Pre-compute word start positions in the region
    word_char_starts: list[int] = []
    for m in re.finditer(r'\S+', region.lower()):
        word_char_starts.append(m.start())

    for i in range(0, len(haystack_words) - n + 1, step):
        window_str = " ".join(haystack_words[i:i + n])
        score = SequenceMatcher(None, needle_str, window_str).ratio()
        if score > best_score:
            best_score = score
            best_pos = i

    if best_score < 0.45 or best_pos < 0:
        return None

    # Refine: check neighbors of best_pos with step=1
    refined_score, refined_pos = best_score, best_pos
    for j in range(max(0, best_pos - step), min(len(haystack_words) - n + 1, best_pos + step + 1)):
        window_str = " ".join(haystack_words[j:j + n])
        score = SequenceMatcher(None, needle_str, window_str).ratio()
        if score > refined_score:
            refined_score = score
            refined_pos = j

    # Convert word position to char position in the full text
    if refined_pos < len(word_char_starts):
        char_in_region = word_char_starts[refined_pos]
        char_pos = search_start + char_in_region
        return char_pos, refined_score

    return None


def _cleanup_whisper_temp(item_id: str):
    """Remove temp files for a whisper sync run."""
    tmp = Path(tempfile.gettempdir()) / "whisper_sync" / item_id
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# Known context window sizes (tokens) for common models.
# Conservative defaults – we only use ~80% of the window for input to leave
# headroom for the system prompt, output tokens, and tokenizer variance.
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16384,
    "gpt-3.5-turbo-16k": 16384,
}
_DEFAULT_CONTEXT_WINDOW = 16384  # safe fallback for unknown models
_CHARS_PER_TOKEN = 3  # conservative estimate (German text ≈ 3 chars/token)
_CONTEXT_USAGE_RATIO = 0.75  # use at most 75% of context for the text chunk
_SYSTEM_PROMPT_TOKENS = 200  # rough reservation for system + framing


def _max_chunk_chars() -> int:
    """Derive the maximum chunk size in characters from the configured model."""
    ctx = _MODEL_CONTEXT_WINDOWS.get(LLM_MODEL, _DEFAULT_CONTEXT_WINDOW)
    usable_tokens = int(ctx * _CONTEXT_USAGE_RATIO) - _SYSTEM_PROMPT_TOKENS
    return max(4000, usable_tokens * _CHARS_PER_TOKEN)


# Matches sentence-ending punctuation optionally followed by closing quotes/parens.
_SENTENCE_END_RE = re.compile(r'[.!?][»«""\'\)]*')


def _snap_to_sentence_start(text: str, pos: int, max_search: int = 500) -> int:
    """Snap *pos* forward to the beginning of the next sentence.

    Searches up to *max_search* characters ahead for a sentence-ending
    punctuation mark followed by whitespace, then returns the position of the
    first non-whitespace character after it (i.e. the start of the next
    sentence).  Falls back to a simple word-boundary snap when no sentence
    boundary is found.
    """
    search_end = min(len(text), pos + max_search)
    m = _SENTENCE_END_RE.search(text, pos, search_end)
    if m:
        new_pos = m.end()
        # skip whitespace to reach the actual start of the next sentence
        while new_pos < len(text) and text[new_pos] in " \t\n\r":
            new_pos += 1
        return new_pos
    # fallback: snap to next word boundary
    idx = text.find(" ", pos)
    if idx != -1 and idx - pos < max_search:
        return idx + 1
    return pos


def _snap_to_sentence_end(text: str, pos: int, max_search: int = 500) -> int:
    """Snap *pos* backward to the end of the last complete sentence.

    Searches up to *max_search* characters back for sentence-ending punctuation
    (including optional closing quotes) and returns the position right after it.
    Falls back to a simple word-boundary snap when no sentence boundary is
    found.
    """
    search_start = max(0, pos - max_search)
    last_end = -1
    for m in _SENTENCE_END_RE.finditer(text, search_start, pos):
        last_end = m.end()
    if last_end > search_start:
        return last_end
    # fallback: snap to previous word boundary
    idx = text.rfind(" ", 0, pos)
    if idx != -1:
        return idx
    return pos


def _split_into_chunks(text: str, max_chars: int | None = None) -> list[str]:
    """Split text into chunks of at most *max_chars*, preferring sentence boundaries."""
    if max_chars is None:
        max_chars = _max_chunk_chars()
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        # prefer splitting at a sentence boundary
        idx = _snap_to_sentence_end(text, end, max_search=max_chars // 2)
        if idx <= start:
            # no sentence boundary – fall back to word boundary
            idx = text.rfind(" ", start, end)
            if idx <= start:
                idx = end  # no space found – hard break (unlikely for prose)
        chunks.append(text[start:idx])
        # skip whitespace before the next chunk
        while idx < len(text) and text[idx] in " \t\n\r":
            idx += 1
        start = idx
    return chunks


_LLM_MAX_RETRIES = 2
_LLM_RETRY_DELAY = 2  # seconds


async def _call_llm(client: AsyncOpenAI, messages: list[dict], max_tokens: int = 1000) -> str:
    """Call LLM with retry on transient errors. Returns empty string on None content."""
    last_err = None
    for attempt in range(_LLM_MAX_RETRIES + 1):
        try:
            r = await client.chat.completions.create(
                model=LLM_MODEL, temperature=0.3, max_tokens=max_tokens, messages=messages)
            return r.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < _LLM_MAX_RETRIES:
                delay = _LLM_RETRY_DELAY * (2 ** attempt)
                logger.warning("LLM-Aufruf fehlgeschlagen (Versuch %d/%d): %s – Retry in %ds",
                               attempt + 1, _LLM_MAX_RETRIES + 1, e, delay)
                await asyncio.sleep(delay)
    raise HTTPException(status_code=502,
                        detail=f"LLM nicht erreichbar nach {_LLM_MAX_RETRIES + 1} Versuchen: {last_err}")


async def summarize_text(text: str, style: str, language: str = "de") -> str:
    styles = {"concise": "3-5 Sätze, Fokus Handlung.",
              "detailed": "Ausführlich: Ereignisse, Dialoge, Charaktere.",
              "bullet": "Aufzählung der Schlüsselereignisse."}
    lang = "Antworte auf Deutsch." if language == "de" else ("Answer in English." if language == "en" else "")
    system_msg = (f"Du fasst Hörbuch-Abschnitte zusammen. Nutzer ist eingeschlafen.\n"
                  f"{styles.get(style, styles['concise'])}\n{lang}\nKeine Spoiler.")

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    chunks = _split_into_chunks(text)

    if len(chunks) == 1:
        return await _call_llm(client, [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Hörbuch-Abschnitt:\n\n{chunks[0]}"}])

    # Summarize chunks in parallel
    logger.info("Text hat %d Zeichen – wird in %d Teile aufgeteilt", len(text), len(chunks))

    async def _summarize_chunk(i: int, chunk: str) -> str:
        logger.info("Zusammenfassung Teil %d/%d …", i, len(chunks))
        return await _call_llm(client, [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Hörbuch-Abschnitt (Teil {i} von {len(chunks)}):\n\n{chunk}"}])

    partial_summaries = await asyncio.gather(
        *(_summarize_chunk(i, chunk) for i, chunk in enumerate(chunks, 1)))

    # Merge partial summaries into one coherent summary
    merged_input = "\n\n---\n\n".join(
        f"Teil {i}: {s}" for i, s in enumerate(partial_summaries, 1))
    return await _call_llm(client, [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": (
            f"Die folgenden Teilzusammenfassungen stammen aus einem zusammenhängenden "
            f"Hörbuch-Abschnitt. Fasse sie zu einer einzigen, kohärenten Zusammenfassung "
            f"zusammen. Behalte alle wichtigen Details bei, entferne aber Redundanz.\n\n"
            f"{merged_input}")}],
        max_tokens=2000)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/index.html")

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "4.1.0"}

@app.get("/api/books")
async def list_books():
    try:
        return {"books": await get_library_items()}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/api/books/{item_id}/chapters")
async def get_chapters(item_id):
    item = await get_item_details(item_id)
    return {"chapters": item.get("media", {}).get("chapters", [])}


# --- Cover proxy (avoids leaking ABS token to browser) ---
@app.get("/api/cover/{item_id}")
async def cover_proxy(item_id: str):
    try:
        c = _client()
        r = await c.get(f"{ABS_URL}/api/items/{item_id}/cover", headers=abs_headers())
        r.raise_for_status()
        ct = r.headers.get("content-type", "image/jpeg")
        return Response(content=r.content, media_type=ct)
    except Exception:
        return Response(status_code=404)


# --- Favorites ---
@app.post("/api/favorites/{item_id}")
async def add_favorite(item_id: str):
    favs = load_favorites()
    if item_id not in favs:
        favs.append(item_id)
        save_favorites(favs)
    return {"is_favorite": True}


@app.delete("/api/favorites/{item_id}")
async def remove_favorite(item_id: str):
    favs = load_favorites()
    if item_id in favs:
        favs.remove(item_id)
        save_favorites(favs)
    return {"is_favorite": False}


# --- Calibration ---
@app.get("/api/calibration/{item_id}")
async def get_calibration(item_id: str):
    """Get calibration status for a book."""
    cal = load_calibrations().get(item_id)
    ec, _, _ = await get_epub_chapters(item_id)
    tw = _total_words(ec)
    wpp = cal["words_per_page"] if cal else DEFAULT_WORDS_PER_PAGE
    offset = cal.get("epub_chapter_offset", 0) if cal else 0
    return {
        "is_calibrated": cal is not None,
        "words_per_page": wpp,
        "total_words": tw,
        "total_pages": max(1, round(tw / wpp)),
        "method": cal.get("method", "default") if cal else "default",
        "default_words_per_page": DEFAULT_WORDS_PER_PAGE,
        "epub_chapter_offset": offset,
        "epub_chapter_count": len(ec),
    }


@app.post("/api/calibrate/by-page", response_model=CalibrateResponse)
async def calibrate_by_page(req: CalibrateByPageRequest):
    """Calibrate by: 'I'm on Kindle page X and audio is at Y seconds'."""
    ec, audio_ch, duration = await get_epub_chapters(req.library_item_id)
    tw = _total_words(ec)
    offset = get_chapter_offset(req.library_item_id)

    char_pos, _, _ = _time_to_char_position(
        audio_ch, ec, req.audio_time_seconds, duration, offset,
        item_id=req.library_item_id,
    )
    full = _build_full_text(ec, req.library_item_id)
    words_at_pos = len(full[:char_pos].split())

    if req.kindle_page <= 0:
        raise HTTPException(status_code=400, detail="Seitenzahl muss > 0 sein")
    wpp = words_at_pos / req.kindle_page
    wpp = max(50, min(600, wpp))

    set_calibration(req.library_item_id, wpp, "by_page",
                    {"kindle_page": req.kindle_page, "audio_seconds": req.audio_time_seconds})

    return CalibrateResponse(
        words_per_page=round(wpp, 1),
        total_pages=max(1, round(tw / wpp)),
        method="by_page",
    )


@app.post("/api/calibrate/by-total", response_model=CalibrateResponse)
async def calibrate_by_total(req: CalibrateByTotalRequest):
    """Calibrate by: 'My Kindle shows X total pages'."""
    ec, _, _ = await get_epub_chapters(req.library_item_id)
    tw = _total_words(ec)

    if req.total_kindle_pages <= 0:
        raise HTTPException(status_code=400, detail="Seitenanzahl muss > 0 sein")
    wpp = tw / req.total_kindle_pages
    wpp = max(50, min(600, wpp))

    set_calibration(req.library_item_id, wpp, "by_total",
                    {"total_kindle_pages": req.total_kindle_pages})

    return CalibrateResponse(
        words_per_page=round(wpp, 1),
        total_pages=max(1, round(tw / wpp)),
        method="by_total",
    )


@app.post("/api/calibrate/offset")
async def set_offset(req: SetOffsetRequest):
    """Set the EPUB front-matter chapter offset."""
    ec, _, _ = await get_epub_chapters(req.library_item_id)
    if req.epub_chapter_offset < 0 or req.epub_chapter_offset >= len(ec):
        raise HTTPException(status_code=400,
                            detail=f"Offset muss zwischen 0 und {len(ec) - 1} liegen")
    set_chapter_offset(req.library_item_id, req.epub_chapter_offset)
    # Clear mapping caches (offset affects chapter matching)
    for key in list(_chapter_mapping_cache):
        if key.startswith(f"{req.library_item_id}:"):
            del _chapter_mapping_cache[key]
    for key in list(_anchor_cache):
        if key.startswith(f"{req.library_item_id}:"):
            del _anchor_cache[key]
    return {"status": "ok", "epub_chapter_offset": req.epub_chapter_offset,
            "epub_chapter_count": len(ec)}


@app.delete("/api/calibration/{item_id}")
async def reset_calibration(item_id: str):
    cal = load_calibrations()
    cal.pop(item_id, None)
    save_calibrations(cal)
    _full_text_cache.pop(item_id, None)
    # Clear mapping caches for this item
    for key in list(_chapter_mapping_cache):
        if key.startswith(f"{item_id}:"):
            del _chapter_mapping_cache[key]
    for key in list(_anchor_cache):
        if key.startswith(f"{item_id}:"):
            del _anchor_cache[key]
    return {"status": "reset"}


@app.post("/api/calibrate/multi-anchor")
async def calibrate_multi_anchor(req: MultiAnchorRequest):
    """Multi-point calibration: store multiple (audio_time, kindle_page) pairs.

    These anchor points are merged with auto-detected chapter matches to
    create a more accurate piecewise-linear mapping.
    """
    if len(req.points) < 1:
        raise HTTPException(status_code=400, detail="Mindestens 1 Ankerpunkt nötig")
    if len(req.points) > 20:
        raise HTTPException(status_code=400, detail="Maximal 20 Ankerpunkte")
    ec, ac, dur = await get_epub_chapters(req.library_item_id)
    for p in req.points:
        if p.audio_seconds < 0 or p.audio_seconds > dur:
            raise HTTPException(status_code=400,
                                detail=f"Audio-Zeit {p.audio_seconds}s außerhalb 0–{dur:.0f}s")
        if p.kindle_page < 1:
            raise HTTPException(status_code=400, detail="Seitenzahl muss >= 1 sein")

    points = [{"audio_seconds": p.audio_seconds, "kindle_page": p.kindle_page}
              for p in sorted(req.points, key=lambda x: x.audio_seconds)]
    set_anchor_points(req.library_item_id, points)

    # Clear caches so next request uses updated anchors
    for key in list(_anchor_cache):
        if key.startswith(f"{req.library_item_id}:"):
            del _anchor_cache[key]

    return {
        "status": "ok",
        "anchor_count": len(points),
        "anchor_points": points,
    }


@app.post("/api/whisper-sync/{item_id}")
async def whisper_sync(item_id: str, req: WhisperSyncRequest):
    """Run Whisper auto-sync: transcribe short audio segments and find them in EPUB.

    Stores results persistently in calibration.json. Only needs to run once per book.
    """
    # Pre-checks
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY nicht konfiguriert")
    if not shutil.which("ffmpeg"):
        raise HTTPException(status_code=422,
                            detail="ffmpeg nicht gefunden. Bitte installieren: apt install ffmpeg")

    # Check if already done
    existing = get_whisper_anchors(item_id)
    if existing and not req.force:
        return {
            "status": "exists",
            "message": f"Whisper-Sync bereits durchgeführt ({len(existing)} Ankerpunkte). "
                       "force=true zum Erneuern.",
            "anchors": existing,
        }

    # Load book data
    ec, ac, dur = await get_epub_chapters(item_id)
    full_text = _build_full_text(ec, item_id)
    if not full_text:
        raise HTTPException(status_code=400, detail="EPUB-Text konnte nicht geladen werden")

    # Get audio files from ABS item details
    item = await get_item_details(item_id)
    media = item.get("media", {})
    audio_files = sorted(media.get("audioFiles", []), key=lambda f: f.get("index", 0))
    if not audio_files:
        raise HTTPException(status_code=400, detail="Keine Audio-Dateien gefunden")

    # Select sample positions
    positions = _select_sample_positions(ac, dur, req.n_samples)
    logger.info("Whisper-Sync %s: %d Samples geplant", item_id[:8], len(positions))

    try:
        # Extract audio segments
        segments: list[Path] = []
        for i, pos in enumerate(positions):
            logger.info("Extrahiere Segment %d/%d bei %.0fs", i + 1, len(positions), pos)
            try:
                seg = await _extract_audio_segment(item_id, audio_files, pos, req.segment_duration)
                segments.append(seg)
            except Exception as e:
                logger.warning("Segment-Extraktion fehlgeschlagen bei %.0fs: %s", pos, e)
                segments.append(None)

        # Transcribe all segments
        valid_segments = [s for s in segments if s is not None]
        logger.info("Transkribiere %d Segmente", len(valid_segments))
        transcriptions_raw = await _transcribe_all(valid_segments, req.language)

        # Map back to positions (filling None for failed extractions)
        transcriptions: list[str | None] = []
        t_idx = 0
        for s in segments:
            if s is not None:
                transcriptions.append(transcriptions_raw[t_idx])
                t_idx += 1
            else:
                transcriptions.append(None)

        # Match each transcription in EPUB text
        anchors: list[dict] = []
        failed: list[int] = []

        for i, (pos, text) in enumerate(zip(positions, transcriptions)):
            if text is None:
                failed.append(i)
                continue
            expected_frac = pos / max(dur, 1)
            result = _find_text_in_epub(text, full_text, expected_frac)
            if result is None:
                failed.append(i)
                logger.info("Sample %d (%.0fs): kein Match im EPUB", i, pos)
                continue

            char_pos, confidence = result
            anchors.append({
                "audio_seconds": round(pos, 1),
                "char_position": char_pos,
                "confidence": round(confidence, 3),
                "whisper_text": text[:100],
            })
            logger.info("Sample %d (%.0fs): Match bei char %d (confidence %.2f)",
                        i, pos, char_pos, confidence)

        # Store persistently
        if anchors:
            set_whisper_anchors(item_id, anchors)
            # Clear caches
            for key in list(_anchor_cache):
                if key.startswith(f"{item_id}:"):
                    del _anchor_cache[key]

        return {
            "status": "ok",
            "total_samples": len(positions),
            "matched": len(anchors),
            "failed_samples": failed,
            "anchors": anchors,
            "cost_seconds": len(valid_segments) * req.segment_duration,
        }

    finally:
        _cleanup_whisper_temp(item_id)


@app.delete("/api/whisper-sync/{item_id}")
async def delete_whisper_sync(item_id: str):
    """Delete stored Whisper anchors for a book."""
    with _calibration_lock:
        cal_path = _calibration_path()
        cal = {}
        if cal_path.exists():
            try:
                cal = json.loads(cal_path.read_text())
            except Exception:
                pass
        entry = cal.get(item_id, {})
        entry.pop("whisper_anchors", None)
        if entry.get("method") == "whisper_auto":
            entry.pop("method", None)
        cal[item_id] = entry
        cal_path.write_text(json.dumps(cal, indent=2))

    # Clear caches
    for key in list(_anchor_cache):
        if key.startswith(f"{item_id}:"):
            del _anchor_cache[key]

    return {"status": "deleted"}


@app.get("/api/chapter-mapping/{item_id}")
async def get_chapter_mapping_endpoint(item_id: str):
    """Show the auto-detected chapter mapping and anchor points.

    Helps users understand and verify the alignment quality.
    """
    ec, ac, dur = await get_epub_chapters(item_id)
    offset = get_chapter_offset(item_id)
    mapping = _get_chapter_mapping(ac, ec, offset, item_id)
    wpp = get_words_per_page(item_id) or DEFAULT_WORDS_PER_PAGE
    manual = _load_manual_anchors(item_id, ec, wpp)
    anchors, _ = _get_anchors(ac, ec, dur, offset, item_id, manual)

    # Build detailed mapping info
    matched = []
    for ai, ei in mapping:
        a = ac[ai]
        e = ec[ei]
        matched.append({
            "audio_chapter": {
                "index": ai,
                "title": a.get("title", ""),
                "start": round(a["start"], 1),
                "end": round(a.get("end", a["start"]), 1),
            },
            "epub_chapter": {
                "index": ei,
                "title": e["title"],
                "word_count": e["word_count"],
            },
            "similarity": round(
                _title_similarity(
                    _normalize_title(a.get("title", "")),
                    _normalize_title(e["title"]),
                ), 2
            ),
        })

    # Unmatched audio chapters
    matched_audio = {ai for ai, _ in mapping}
    unmatched_audio = [
        {"index": i, "title": a.get("title", ""), "start": round(a["start"], 1)}
        for i, a in enumerate(ac)
        if i not in matched_audio
    ]

    # Unmatched epub chapters
    matched_epub = {ei for _, ei in mapping}
    unmatched_epub = [
        {"index": i, "title": e["title"], "word_count": e["word_count"]}
        for i, e in enumerate(ec)
        if i not in matched_epub
    ]

    return {
        "quality": _mapping_quality_label(mapping, ac, ec, anchors, item_id),
        "matched_count": len(mapping),
        "audio_chapter_count": len(ac),
        "epub_chapter_count": len(ec),
        "epub_chapter_offset": offset,
        "matched_chapters": matched,
        "unmatched_audio": unmatched_audio,
        "unmatched_epub": unmatched_epub,
        "anchor_count": len(anchors),
        "anchors": [{"time": round(t, 1), "char_pos": c} for t, c in anchors],
        "manual_anchor_points": get_anchor_points(item_id),
        "whisper_anchor_points": get_whisper_anchors(item_id),
    }


# --- Position ---
@app.post("/api/position", response_model=PositionResponse)
async def get_position(req: PositionRequest):
    ec, ac, dur = await get_epub_chapters(req.library_item_id)
    full = _build_full_text(ec, req.library_item_id)
    wpp = get_words_per_page(req.library_item_id) or DEFAULT_WORDS_PER_PAGE
    tp = max(1, round(_total_words(ec) / wpp))
    tw = _total_words(ec)
    offset = get_chapter_offset(req.library_item_id)
    manual = _load_manual_anchors(req.library_item_id, ec, wpp)
    cp, ct, cpct = _time_to_char_position(
        ac, ec, req.current_time_seconds, dur, offset,
        item_id=req.library_item_id, manual_anchors=manual,
    )
    wb = len(full[:cp].split())
    page = max(1, min(tp, round(wb / wpp) + 1))
    pct = round(wb / max(tw, 1) * 100, 1)
    s, e = max(0, cp - 200), min(len(full), cp + 200)
    if s > 0:
        i = full.find(" ", s)
        if i != -1: s = i + 1
    i = full.rfind(" ", 0, e)
    if i != -1: e = i
    # Determine mapping quality
    mapping = _get_chapter_mapping(ac, ec, offset, req.library_item_id)
    pos_anchors, _ = _get_anchors(ac, ec, dur, offset, req.library_item_id, manual)
    return PositionResponse(estimated_page=page, total_pages=tp, percentage=pct,
                            chapter_title=ct, chapter_progress_pct=round(cpct, 1),
                            nearby_text=full[s:e],
                            is_calibrated=get_words_per_page(req.library_item_id) is not None,
                            words_per_page=wpp,
                            mapping_quality=_mapping_quality_label(mapping, ac, ec, pos_anchors, req.library_item_id),
                            matched_chapters=len(mapping))


# --- Recap ---
@app.post("/api/recap", response_model=RecapResponse)
async def create_recap(req: RecapRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    ss, es = req.start_minutes * 60, req.end_minutes * 60
    d = es - ss
    if d <= 0: raise HTTPException(status_code=400, detail="End > Start")
    if d > 7200: raise HTTPException(status_code=400, detail="Max 120 min")
    ec, ac, _ = await get_epub_chapters(req.library_item_id)
    offset = get_chapter_offset(req.library_item_id)
    text, names = map_time_to_text(ac, ec, ss, es, offset, item_id=req.library_item_id)
    if not text or len(text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Kein Text gefunden")
    summary = await summarize_text(text, req.summary_style, SUMMARY_LANGUAGE)
    n_chunks = len(_split_into_chunks(text))
    ti = len(text.split()) * 1.3
    to = len(summary.split()) * 1.3
    if n_chunks > 1:
        ti += n_chunks * 200 * 1.3
        to += n_chunks * 200 * 1.3
    cost = (ti * 0.00015 + to * 0.0006) / 1000
    return RecapResponse(text_excerpt=text[:5000] + ("…" if len(text) > 5000 else ""),
                         summary=summary, chapters_covered=names,
                         duration_seconds=d, cost_estimate_usd=round(cost, 6))


# --- Find ---
@app.post("/api/find-text", response_model=TextSearchResponse)
async def find_text(req: TextSearchRequest):
    ec, ac, dur = await get_epub_chapters(req.library_item_id)
    full = _build_full_text(ec, req.library_item_id)
    offset = get_chapter_offset(req.library_item_id)
    q = req.query.strip()
    if len(q) < 3: raise HTTPException(status_code=400, detail="Min 3 Zeichen")
    idx = full.lower().find(q.lower())
    conf = "exact"
    if idx == -1:
        words = q.split()
        bi, bl = -1, 0
        for wc in range(len(words), 0, -1):
            for sw in range(len(words) - wc + 1):
                sub = " ".join(words[sw:sw + wc])
                p = full.lower().find(sub.lower())
                if p != -1 and len(sub) > bl:
                    bi, bl = p, len(sub)
            if bi != -1: break
        if bi != -1: idx, conf = bi, "approximate"
        else: raise HTTPException(status_code=404, detail="Text nicht gefunden")
    ts, ct = _char_position_to_time(ac, ec, idx, dur, offset, item_id=req.library_item_id)
    cs, ce = max(0, idx - 80), min(len(full), idx + len(q) + 80)
    return TextSearchResponse(audio_timestamp_seconds=round(ts, 1),
                              audio_timestamp_formatted=_format_time(ts),
                              chapter_title=ct, confidence=conf,
                              context="…" + full[cs:ce] + "…")


@app.post("/api/page-to-audio", response_model=PageToAudioResponse)
async def page_to_audio(req: PageToAudioRequest):
    ec, ac, dur = await get_epub_chapters(req.library_item_id)
    full = _build_full_text(ec, req.library_item_id)
    offset = get_chapter_offset(req.library_item_id)
    wpp = get_words_per_page(req.library_item_id) or DEFAULT_WORDS_PER_PAGE
    tp = max(1, round(_total_words(ec) / wpp))
    if req.page_number < 1 or req.page_number > tp:
        raise HTTPException(status_code=400, detail=f"Seite 1–{tp}")
    wt = (req.page_number - 1) * wpp
    wc, cp = 0, 0
    for i, ch in enumerate(full):
        if ch == ' ': wc += 1
        if wc >= wt: cp = i; break
    else: cp = len(full) - 1
    manual = _load_manual_anchors(req.library_item_id, ec, wpp)
    ts, ct = _char_position_to_time(
        ac, ec, cp, dur, offset,
        item_id=req.library_item_id, manual_anchors=manual,
    )
    s, e = max(0, cp - 200), min(len(full), cp + 200)
    if s > 0:
        i = full.find(" ", s)
        if i != -1: s = i + 1
    i = full.rfind(" ", 0, e)
    if i != -1: e = i
    return PageToAudioResponse(audio_timestamp_seconds=round(ts, 1),
                               audio_timestamp_formatted=_format_time(ts),
                               chapter_title=ct, nearby_text=full[s:e])


@app.post("/api/sync-progress")
async def sync_progress(library_item_id: str, time_seconds: float):
    if not ABS_TOKEN: raise HTTPException(status_code=500, detail="No ABS token")
    item = await get_item_details(library_item_id)
    dur = item.get("media", {}).get("duration", 0)
    await update_abs_progress(library_item_id, time_seconds, dur)
    return {"status": "ok", "new_position": _format_time(time_seconds)}

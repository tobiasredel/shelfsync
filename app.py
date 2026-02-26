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
    # Prefetch EPUB for currently-reading book in the background
    asyncio.create_task(_prefetch_currently_reading())
    yield
    await _http_client.aclose()
    _http_client = None


async def _prefetch_currently_reading():
    """Background task: pre-warm EPUB cache for all currently-reading books."""
    cr_ids = load_currently_reading()
    if not cr_ids:
        return
    for cr_id in cr_ids:
        try:
            logger.info("Prefetching EPUB for currently-reading book %s", cr_id[:8])
            await get_epub_chapters(cr_id)
            logger.info("Prefetch complete for %s", cr_id[:8])
        except Exception as e:
            logger.warning("Prefetch failed for %s: %s", cr_id[:8], e)


app = FastAPI(title="Audiobook Recap", version="4.1.0",
              lifespan=lifespan, dependencies=[Depends(verify_auth)])
app.mount("/static", StaticFiles(directory="static"), name="static")

_epub_cache: OrderedDict[str, list[dict]] = OrderedDict()


# ---------------------------------------------------------------------------
# Calibration persistence (thread-safe)
# ---------------------------------------------------------------------------
_calibration_lock = threading.Lock()
_currently_reading_lock = threading.Lock()


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


def has_whisper_sync(item_id: str) -> bool:
    """Check if a book has Whisper-Sync anchors."""
    return bool(get_whisper_anchors(item_id))


# ---------------------------------------------------------------------------
# Currently-reading persistence (thread-safe, single book)
# ---------------------------------------------------------------------------
def _currently_reading_path() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "currently_reading.json"


def load_currently_reading() -> list[str]:
    """Return the list of currently-reading item IDs."""
    with _currently_reading_lock:
        p = _currently_reading_path()
        if p.exists():
            try:
                data = json.loads(p.read_text())
                # Migrate from old single-item format
                if isinstance(data, dict) and "item_id" in data and "item_ids" not in data:
                    old_id = data.get("item_id")
                    return [old_id] if old_id else []
                if isinstance(data, dict):
                    return data.get("item_ids", [])
                return []
            except Exception:
                return []
        return []


def save_currently_reading(item_ids: list[str]):
    with _currently_reading_lock:
        _currently_reading_path().write_text(
            json.dumps({"item_ids": item_ids}, indent=2)
        )


def add_currently_reading(item_id: str):
    ids = load_currently_reading()
    if item_id not in ids:
        ids.append(item_id)
        save_currently_reading(ids)


def remove_currently_reading(item_id: str):
    ids = load_currently_reading()
    ids = [i for i in ids if i != item_id]
    save_currently_reading(ids)


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


def _build_full_text(ec, item_id: str | None = None) -> str:
    if item_id and item_id in _full_text_cache:
        return _full_text_cache[item_id]
    ft = " ".join(ch["text"] for ch in ec)
    if item_id:
        _full_text_cache[item_id] = ft
    return ft


def _epub_char_starts(epub_ch) -> list[int]:
    """Cumulative character start position for each EPUB chapter."""
    starts = []
    cum = 0
    for e in epub_ch:
        starts.append(cum)
        cum += e["char_count"] + 1  # +1 for space between chapters
    return starts


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
    epub_ch: list[dict],
    total_dur: float,
    item_id: str | None = None,
) -> list[tuple[float, int]]:
    """Get anchor points from Whisper-Sync.

    Uses only Whisper auto-sync anchors plus start/end boundaries.

    Returns sorted list of (audio_seconds, char_position) anchors.
    """
    whisper_anchors = _load_whisper_anchors(item_id) if item_id else None

    total_chars = sum(e["char_count"] + 1 for e in epub_ch)
    anchors: list[tuple[float, int]] = [(0.0, 0)]
    if whisper_anchors:
        anchors.extend(whisper_anchors)
    anchors.append((total_dur, total_chars))

    # Sort by time, deduplicate (keep last for same time)
    anchors.sort(key=lambda x: (x[0], x[1]))
    cleaned: list[tuple[float, int]] = [anchors[0]]
    dropped = 0
    for t, c in anchors[1:]:
        if t > cleaned[-1][0] + 0.5:
            if c >= cleaned[-1][1]:
                cleaned.append((t, c))
            else:
                dropped += 1
                logger.warning("Whisper-Anker bei %.0fs verworfen: char_pos %d < vorheriger %d",
                               t, c, cleaned[-1][1])
        else:
            if c > cleaned[-1][1]:
                cleaned[-1] = (t, c)
    if dropped:
        logger.warning("Whisper-Sync: %d von %d Ankern wegen Monotonie verworfen",
                       dropped, len(anchors) - 2)

    return cleaned


def _find_epub_chapter_at_char(epub_ch: list[dict], char_pos: int) -> tuple[str, float]:
    """Find the EPUB chapter and progress % at a given character position."""
    cum = 0
    for i, e in enumerate(epub_ch):
        ch_end = cum + e["char_count"]
        if char_pos <= ch_end:
            pct = (char_pos - cum) / max(e["char_count"], 1) * 100
            return e.get("title", f"Kapitel {i + 1}"), max(0, min(100, pct))
        cum = ch_end + 1  # +1 for space between chapters
    # Past the end
    if epub_ch:
        return epub_ch[-1].get("title", f"Kapitel {len(epub_ch)}"), 100.0
    return "(unbekannt)", 0.0


def _time_to_char_position(
    epub_ch, time_sec, total_dur,
    item_id: str | None = None,
):
    """Map audio time → character position in EPUB full text.

    Uses Whisper anchor interpolation with start/end boundaries.

    Returns: (char_offset, epub_chapter_title, epub_chapter_progress_pct)
    """
    full_text = _build_full_text(epub_ch, item_id)
    anchors = _get_anchors(epub_ch, total_dur, item_id)

    char_pos = _interpolate_time_to_char(anchors, time_sec)
    char_pos = max(0, min(char_pos, len(full_text) - 1))

    title, pct = _find_epub_chapter_at_char(epub_ch, char_pos)
    return char_pos, title, pct


def _char_position_to_time(
    epub_ch, char_pos, total_dur,
    item_id: str | None = None,
):
    """Map character position → audio time.

    Uses Whisper anchor interpolation with start/end boundaries.

    Returns: (audio_time_seconds, chapter_title)
    """
    anchors = _get_anchors(epub_ch, total_dur, item_id)

    t = _interpolate_char_to_time(anchors, char_pos)
    t = max(0.0, min(t, total_dur))
    title, _ = _find_epub_chapter_at_char(epub_ch, char_pos)
    return t, title


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
    epub_ch, start_sec, end_sec,
    total_dur: float, item_id: str | None = None,
):
    """Extract EPUB text for an audio time range using Whisper-Sync anchors."""
    if not epub_ch:
        raise ValueError("No EPUB chapters")

    anchors = _get_anchors(epub_ch, total_dur, item_id)
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
    names = _chapters_in_char_range(epub_ch, c0, c1) if text.strip() else ["(geschätzt)"]
    return text, names


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
    """Book listing with lightweight metadata.

    Includes has_epub (free check on listing data) and currently-reading flags.
    Calibration status is loaded lazily via /api/books/{id}/details.
    """
    c = _client()
    r = await c.get(f"{ABS_URL}/api/libraries", headers=abs_headers())
    r.raise_for_status()
    libs = r.json().get("libraries", [])
    items: list[dict] = []
    cr_ids = set(load_currently_reading())
    for lib in libs:
        r2 = await c.get(
            f"{ABS_URL}/api/libraries/{lib['id']}/items",
            headers=abs_headers(),
            params={"limit": 100, "sort": "media.metadata.title",
                    "include": "rssfeed,numEpisodesIncomplete,progress",
                    "expanded": 1},
        )
        r2.raise_for_status()
        for item in r2.json().get("results", []):
            media = item.get("media", {})
            md = media.get("metadata", {})
            prog = item.get("userMediaProgress") or {}
            items.append({
                "id": item["id"],
                "title": md.get("title", "Unknown"),
                "author": (md.get("authorName")
                           or ", ".join(a.get("name", "")
                                        for a in md.get("authors", []))
                           or "Unknown"),
                "duration": media.get("duration", 0),
                "current_time": prog.get("currentTime", 0),
                "cover": f"/api/cover/{item['id']}",
                "has_epub": _find_epub_file(item) is not None,
                "is_currently_reading": item["id"] in cr_ids,
            })
    return items


async def get_item_details(item_id):
    c = _client()
    r = await c.get(f"{ABS_URL}/api/items/{item_id}", headers=abs_headers(),
                    params={"expanded": 1, "include": "progress"})
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
    """Select evenly-spaced sample positions, avoiding chapter boundaries.

    For longer books, automatically increases sample count for better coverage.
    Shifts samples away from chapter boundaries where narration may be unclear.
    """
    # Adaptive: at least 1 sample per 30 min, clamped to user request
    auto_min = max(3, int(total_dur / 1800))
    n_samples = max(auto_min, min(n_samples, 20))

    # Evenly spaced across the book (skip first/last 60 seconds for intros/outros)
    margin = min(60, total_dur * 0.02)
    step = (total_dur - 2 * margin) / max(n_samples - 1, 1)
    positions = [margin + i * step for i in range(n_samples)]

    # Avoid chapter boundaries (±8s) — shift samples into middle of chapter
    boundaries = sorted(set(
        [ch["start"] for ch in audio_ch] +
        [ch["end"] for ch in audio_ch if "end" in ch]
    ))

    adjusted = []
    for pos in positions:
        for b in boundaries:
            if abs(pos - b) < 8:
                # Shift 12 seconds into the chapter (past transition)
                pos = b + 12
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
) -> tuple[int, float] | None:
    """Find whisper-transcribed text in EPUB full text.

    Uses a sliding window on word-level with SequenceMatcher.
    Adaptive search window: narrower for longer snippets (more unique),
    wider for shorter ones.

    Returns (char_position, confidence_score) or None.
    """
    needle_words = _normalize_for_matching(whisper_text)
    if len(needle_words) < 3:
        return None

    # Adaptive window: longer transcriptions are more unique → search narrower
    n = len(needle_words)
    if n >= 15:
        window = 0.10
    elif n >= 8:
        window = 0.15
    else:
        window = 0.25

    # Determine search region in char space
    total_len = len(full_text)
    center = int(expected_frac * total_len)
    half_win = int(window * total_len)
    search_start = max(0, center - half_win)
    search_end = min(total_len, center + half_win)
    region = full_text[search_start:search_end]

    haystack_words = _normalize_for_matching(region)
    if len(haystack_words) < n:
        return None

    needle_str = " ".join(needle_words)
    best_score, best_pos = 0.0, -1
    # Finer step for better match precision (1/10 of needle length)
    step = max(1, n // 10)

    # Track character positions: map word index → char offset in region
    normalized_region = re.sub(r'[^\w\s\-äöüàáâèéêìíîòóôùúûß]', ' ', region.lower())
    word_char_starts: list[int] = []
    for m in re.finditer(r'\S+', normalized_region):
        word_char_starts.append(m.start())

    for i in range(0, len(haystack_words) - n + 1, step):
        window_str = " ".join(haystack_words[i:i + n])
        score = SequenceMatcher(None, needle_str, window_str).ratio()
        if score > best_score:
            best_score = score
            best_pos = i

    if best_score < 0.50 or best_pos < 0:
        return None

    # Refine: check neighbors of best_pos with step=1
    refined_score, refined_pos = best_score, best_pos
    refine_range = max(step, 3)
    for j in range(max(0, best_pos - refine_range),
                   min(len(haystack_words) - n + 1, best_pos + refine_range + 1)):
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

@app.get("/api/books/{item_id}/details")
async def get_book_details(item_id: str):
    """Lazy-loaded book details: EPUB availability + calibration status.

    Called when a user selects a book, not during initial listing.
    """
    item = await get_item_details(item_id)
    has_epub = _find_epub_file(item) is not None
    cal_data = load_calibrations()
    cal = cal_data.get(item_id)
    prog = item.get("userMediaProgress") or {}
    return {
        "item_id": item_id,
        "has_epub": has_epub,
        "is_calibrated": cal is not None and bool(cal.get("whisper_anchors")),
        "current_time": prog.get("currentTime", 0),
        "duration": item.get("media", {}).get("duration", 0),
    }


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


# --- Currently Reading ---
@app.post("/api/currently-reading/{item_id}")
async def set_currently_reading_route(item_id: str):
    add_currently_reading(item_id)
    return {"is_currently_reading": True, "item_id": item_id}


@app.delete("/api/currently-reading/{item_id}")
async def remove_currently_reading_route(item_id: str):
    remove_currently_reading(item_id)
    return {"is_currently_reading": False, "item_id": item_id}


@app.get("/api/currently-reading")
async def get_currently_reading_route():
    return {"item_ids": load_currently_reading()}


# --- Calibration ---
@app.get("/api/calibration/{item_id}")
async def get_calibration(item_id: str):
    """Get Whisper-Sync status for a book."""
    whisper = get_whisper_anchors(item_id)
    return {
        "has_whisper_sync": bool(whisper),
        "anchor_count": len(whisper),
    }


@app.delete("/api/calibration/{item_id}")
async def reset_calibration(item_id: str):
    """Reset all calibration data for a book."""
    cal = load_calibrations()
    cal.pop(item_id, None)
    save_calibrations(cal)
    _full_text_cache.pop(item_id, None)
    return {"status": "reset"}


@app.put("/api/books/{item_id}/kindle-pages")
async def set_kindle_pages(item_id: str, pages: int):
    """Store the Kindle total page count for a book."""
    if pages < 1:
        raise HTTPException(status_code=400, detail="Seitenzahl muss >= 1 sein")
    with _calibration_lock:
        cal_path = _calibration_path()
        cal = {}
        if cal_path.exists():
            try:
                cal = json.loads(cal_path.read_text())
            except Exception:
                pass
        entry = cal.get(item_id, {})
        entry["kindle_pages"] = pages
        cal[item_id] = entry
        cal_path.write_text(json.dumps(cal, indent=2))
    return {"item_id": item_id, "kindle_pages": pages}


@app.get("/api/books/{item_id}/kindle-pages")
async def get_kindle_pages(item_id: str):
    cal = load_calibrations()
    pages = (cal.get(item_id) or {}).get("kindle_pages")
    return {"item_id": item_id, "kindle_pages": pages}


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

    return {"status": "deleted"}


@app.get("/api/chapter-mapping/{item_id}")
async def get_chapter_mapping_endpoint(item_id: str):
    """Show Whisper-Sync anchor points for a book."""
    ec, ac, dur = await get_epub_chapters(item_id)
    anchors = _get_anchors(ec, dur, item_id)
    whisper = get_whisper_anchors(item_id)

    return {
        "quality": "whisper" if whisper else "none",
        "audio_chapter_count": len(ac),
        "epub_chapter_count": len(ec),
        "anchor_count": len(anchors),
        "anchors": [{"time": round(t, 1), "char_pos": c} for t, c in anchors],
        "whisper_anchor_points": whisper,
    }


# --- Position ---
@app.post("/api/position", response_model=PositionResponse)
async def get_position(req: PositionRequest):
    ec, ac, dur = await get_epub_chapters(req.library_item_id)
    full = _build_full_text(ec, req.library_item_id)
    tw = sum(ch["word_count"] for ch in ec)
    wpp = DEFAULT_WORDS_PER_PAGE
    tp = max(1, round(tw / wpp))
    cp, ct, cpct = _time_to_char_position(
        ec, req.current_time_seconds, dur,
        item_id=req.library_item_id,
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
    # Use stored Kindle page count if available
    cal_data = load_calibrations()
    kindle_pages = (cal_data.get(req.library_item_id) or {}).get("kindle_pages")
    if kindle_pages and kindle_pages > 0:
        tp = kindle_pages
        wpp = tw / max(tp, 1)
        page = max(1, min(tp, round(wb / max(wpp, 1)) + 1))

    synced = has_whisper_sync(req.library_item_id)
    return PositionResponse(estimated_page=page, total_pages=tp, percentage=pct,
                            chapter_title=ct, chapter_progress_pct=round(cpct, 1),
                            nearby_text=full[s:e],
                            is_calibrated=synced,
                            words_per_page=round(wpp, 1),
                            mapping_quality="whisper" if synced else "none",
                            matched_chapters=0)


# --- Recap ---
@app.post("/api/recap", response_model=RecapResponse)
async def create_recap(req: RecapRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    ss, es = req.start_minutes * 60, req.end_minutes * 60
    d = es - ss
    if d <= 0: raise HTTPException(status_code=400, detail="End > Start")
    if d > 7200: raise HTTPException(status_code=400, detail="Max 120 min")
    ec, ac, dur = await get_epub_chapters(req.library_item_id)
    total_dur = ac[-1].get("end", 0) if ac else dur
    text, names = map_time_to_text(ec, ss, es, total_dur, item_id=req.library_item_id)
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
    ts, ct = _char_position_to_time(ec, idx, dur, item_id=req.library_item_id)
    cs, ce = max(0, idx - 80), min(len(full), idx + len(q) + 80)
    return TextSearchResponse(audio_timestamp_seconds=round(ts, 1),
                              audio_timestamp_formatted=_format_time(ts),
                              chapter_title=ct, confidence=conf,
                              context="…" + full[cs:ce] + "…")


@app.post("/api/page-to-audio", response_model=PageToAudioResponse)
async def page_to_audio(req: PageToAudioRequest):
    ec, ac, dur = await get_epub_chapters(req.library_item_id)
    full = _build_full_text(ec, req.library_item_id)
    wpp = DEFAULT_WORDS_PER_PAGE
    tw = sum(ch["word_count"] for ch in ec)
    tp = max(1, round(tw / wpp))
    if req.page_number < 1 or req.page_number > tp:
        raise HTTPException(status_code=400, detail=f"Seite 1–{tp}")
    wt = (req.page_number - 1) * wpp
    wc, cp = 0, 0
    for i, ch in enumerate(full):
        if ch == ' ': wc += 1
        if wc >= wt: cp = i; break
    else: cp = len(full) - 1
    ts, ct = _char_position_to_time(ec, cp, dur, item_id=req.library_item_id)
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

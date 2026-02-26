"""
Audiobook Recap v4 – Recap + Position Sync + Dynamic Calibration
"""

import asyncio
import base64
import html as html_mod
import io
import json
import logging
import os
import re
import secrets
import threading
import zipfile
from collections import OrderedDict
from contextlib import asynccontextmanager
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


class SetOffsetRequest(BaseModel):
    library_item_id: str
    epub_chapter_offset: int


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
# Mapping helpers (with front-matter offset support)
# ---------------------------------------------------------------------------
_full_text_cache: dict[str, str] = {}


def _build_full_text(ec, item_id: str | None = None) -> str:
    if item_id and item_id in _full_text_cache:
        return _full_text_cache[item_id]
    ft = " ".join(ch["text"] for ch in ec)
    if item_id:
        _full_text_cache[item_id] = ft
    return ft


def _total_words(ec): return sum(ch["word_count"] for ch in ec)


def _time_to_char_position(audio_ch, epub_ch, time_sec, total_dur, offset: int = 0):
    full_text = _build_full_text(epub_ch)
    if not audio_ch:
        ratio = time_sec / max(total_dur, 1)
        return int(len(full_text) * ratio), "(geschätzt)", ratio * 100
    cur, ci = audio_ch[0], 0
    for i, a in enumerate(audio_ch):
        if a["start"] <= time_sec < a.get("end", a["start"]):
            cur, ci = a, i; break
        if a["start"] <= time_sec:
            cur, ci = a, i
    cs, ce = cur["start"], cur.get("end", cur["start"])
    cd = ce - cs
    cp = max(0, min(1, (time_sec - cs) / max(cd, 1)))
    # Map audio chapter ci → EPUB chapter ci + offset
    epub_idx = ci + offset
    co = 0
    for i, e in enumerate(epub_ch):
        if i < epub_idx:
            co += e["char_count"] + 1
        elif i == epub_idx:
            co += int(e["char_count"] * cp); break
    return min(co, len(full_text) - 1), cur.get("title", ""), cp * 100


def _char_position_to_time(audio_ch, epub_ch, char_pos, total_dur, offset: int = 0):
    if not audio_ch or not epub_ch:
        full = _build_full_text(epub_ch)
        return (char_pos / max(len(full), 1)) * total_dur, "(geschätzt)"
    cum, ti, ep = 0, 0, 0.0
    for i, e in enumerate(epub_ch):
        if cum + e["char_count"] >= char_pos:
            ti, ep = i, (char_pos - cum) / max(e["char_count"], 1); break
        cum += e["char_count"] + 1; ti, ep = i, 1.0
    # EPUB chapter ti → audio chapter ti - offset
    ai = min(max(0, ti - offset), len(audio_ch) - 1)
    a = audio_ch[ai]
    return a["start"] + (a.get("end", a["start"]) - a["start"]) * ep, a.get("title", "")


def _format_time(s):
    h, m, sec = int(s // 3600), int(s % 3600 // 60), int(s % 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def map_time_to_text(audio_ch, epub_ch, start_sec, end_sec, offset: int = 0):
    if not epub_ch:
        raise ValueError("No EPUB chapters")
    if not audio_ch:
        return _pfb(epub_ch, start_sec, end_sec, end_sec)
    parts, names = [], []
    for i, a in enumerate(audio_ch):
        cs, ce = a.get("start", 0), a.get("end", 0)
        if cs >= end_sec or ce <= start_sec:
            continue
        # Apply front-matter offset: audio chapter i → EPUB chapter i + offset
        ei = i + offset
        e = epub_ch[ei] if ei < len(epub_ch) else None
        if not e:
            at = a.get("title", "").lower()
            for ec in epub_ch:
                if ec["title"].lower() in at or at in ec["title"].lower():
                    e = ec; break
        if not e:
            continue
        d = ce - cs
        if d <= 0:
            continue
        ps, pe = max(0, (start_sec - cs) / d), min(1, (end_sec - cs) / d)
        t = e["text"]; c0, c1 = int(len(t) * ps), int(len(t) * pe)
        if c0 > 0:
            idx = t.find(" ", c0)
            if idx != -1 and idx - c0 < 50: c0 = idx + 1
        if c1 < len(t):
            idx = t.rfind(" ", 0, c1)
            if idx != -1: c1 = idx
        p = t[c0:c1]
        if p.strip():
            parts.append(p); names.append(a.get("title", e["title"]))
    if not parts:
        td = audio_ch[-1].get("end", 0) if audio_ch else end_sec
        return _pfb(epub_ch, start_sec, end_sec, td)
    return "\n\n".join(parts), names


def _pfb(ec, ss, es, td):
    if td <= 0: td = 1
    ft = " ".join(c["text"] for c in ec)
    c0, c1 = int(len(ft) * ss / td), int(len(ft) * es / td)
    i = ft.find(" ", c0)
    if i != -1 and i - c0 < 100: c0 = i + 1
    i = ft.rfind(" ", 0, c1)
    if i != -1: c1 = i
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
        c.get(f"{ABS_URL}/api/items/{iid}", headers=abs_headers()) for iid in item_ids
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
            "author": md.get("authorName", "Unknown"),
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
    r = await c.get(f"{ABS_URL}/api/items/{item_id}", headers=abs_headers())
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


def _split_into_chunks(text: str, max_chars: int | None = None) -> list[str]:
    """Split text into chunks of at most *max_chars*, breaking at word boundaries."""
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
        # find last space before the limit so we don't split mid-word
        idx = text.rfind(" ", start, end)
        if idx <= start:
            idx = end  # no space found – hard break (unlikely for prose)
        chunks.append(text[start:idx])
        start = idx + 1  # skip the space
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

    char_pos, _, _ = _time_to_char_position(audio_ch, ec, req.audio_time_seconds, duration, offset)
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
    return {"status": "ok", "epub_chapter_offset": req.epub_chapter_offset,
            "epub_chapter_count": len(ec)}


@app.delete("/api/calibration/{item_id}")
async def reset_calibration(item_id: str):
    cal = load_calibrations()
    cal.pop(item_id, None)
    save_calibrations(cal)
    _full_text_cache.pop(item_id, None)
    return {"status": "reset"}


# --- Position ---
@app.post("/api/position", response_model=PositionResponse)
async def get_position(req: PositionRequest):
    ec, ac, dur = await get_epub_chapters(req.library_item_id)
    full = _build_full_text(ec, req.library_item_id)
    wpp = get_words_per_page(req.library_item_id) or DEFAULT_WORDS_PER_PAGE
    tp = max(1, round(_total_words(ec) / wpp))
    tw = _total_words(ec)
    offset = get_chapter_offset(req.library_item_id)
    cp, ct, cpct = _time_to_char_position(ac, ec, req.current_time_seconds, dur, offset)
    wb = len(full[:cp].split())
    page = max(1, min(tp, round(wb / wpp) + 1))
    pct = round(wb / max(tw, 1) * 100, 1)
    s, e = max(0, cp - 60), min(len(full), cp + 60)
    if s > 0:
        i = full.find(" ", s)
        if i != -1: s = i + 1
    i = full.rfind(" ", 0, e)
    if i != -1: e = i
    return PositionResponse(estimated_page=page, total_pages=tp, percentage=pct,
                            chapter_title=ct, chapter_progress_pct=round(cpct, 1),
                            nearby_text=full[s:e],
                            is_calibrated=get_words_per_page(req.library_item_id) is not None,
                            words_per_page=wpp)


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
    text, names = map_time_to_text(ac, ec, ss, es, offset)
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
    ts, ct = _char_position_to_time(ac, ec, idx, dur, offset)
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
    ts, ct = _char_position_to_time(ac, ec, cp, dur, offset)
    s, e = max(0, cp - 60), min(len(full), cp + 60)
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

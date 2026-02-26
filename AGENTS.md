# AGENTS.md – Audiobook Recap

## Project Overview

**Audiobook Recap** is a self-hosted web tool that bridges audiobooks and eBooks. It connects to a user's **Audiobookshelf** instance, extracts text from EPUBs, and provides four core features:

1. **Recap** – Summarize a missed audiobook segment (e.g. "what happened between minute 45 and 55?") using EPUB text + GPT-4o-mini
2. **Position Sync** – Show which Kindle page / percentage corresponds to the current audio position
3. **Find & Jump** – Enter a Kindle page number or text passage → get the audio timestamp → optionally update the ABS listening position
4. **Search & Favorites** – Filter the book list by title/author as you type; star books to pin them at the top, persisted across sessions

The key insight is that EPUB text extraction replaces expensive Speech-to-Text (Whisper), making the tool ~14x cheaper (~$0.005 vs $0.07 per recap).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (Single-page HTML/JS)                              │
│  static/index.html                                          │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP (JSON)
┌──────────────────────▼──────────────────────────────────────┐
│  FastAPI Backend (app.py)                                   │
│  - /api/books            → list ABS library items           │
│  - /api/books/:id/chapters → audio chapter markers          │
│  - /api/position         → audio time → page/percentage     │
│  - /api/recap            → time range → EPUB text → GPT     │
│  - /api/find-text        → text query → audio timestamp     │
│  - /api/page-to-audio    → page number → audio timestamp    │
│  - /api/sync-progress    → update ABS listening position    │
│  - /api/calibrate/*      → per-book Kindle page calibration │
│  - /api/calibration/:id  → get/delete calibration           │
│  - /api/favorites/:id    → add/remove book favorites        │
└──────┬──────────┬───────────────────────────────────────────┘
       │          │
       ▼          ▼
  Audiobookshelf  OpenAI API
  REST API        (GPT-4o-mini only)
  - libraries     - chat.completions
  - items
  - file download
  - user progress
```

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12, FastAPI, uvicorn |
| HTTP client | httpx (async) |
| EPUB parsing | stdlib zipfile + xml.etree.ElementTree + regex HTML stripping |
| LLM | OpenAI API (gpt-4o-mini default) |
| Frontend | Vanilla HTML/CSS/JS (single file, no build step) |
| Fonts | Google Fonts: DM Sans + DM Serif Display |
| Persistence | JSON files in `DATA_DIR` (`calibration.json`, `favorites.json`, `epub_cache/`) |
| Deployment | Docker, docker-compose, targeted at Synology NAS |

## File Structure

```
audiobook-recap/
├── app.py                 # FastAPI application (all backend logic)
├── static/
│   └── index.html         # Single-file frontend (HTML + CSS + JS)
├── Dockerfile             # Slim Python image, no ffmpeg needed
├── docker-compose.yml     # Production config with volume mount
├── requirements.txt       # Python dependencies
├── README.md              # User-facing setup guide
└── AGENTS.md              # This file
```

## Key Design Decisions

### EPUB-based instead of Audio Transcription
- EPUBs are parsed with stdlib (zipfile + ElementTree), no external dependencies
- Text is mapped to audio position via ABS chapter markers
- Fallback: proportional mapping (char position ≈ time ratio) when chapters don't align

### Single-file Frontend
- No build tools, no npm, no React – just one HTML file with embedded CSS/JS
- Dark theme, mobile-responsive, designed for sleepy users
- All API calls use fetch() with JSON

### Per-book Kindle Calibration
- Words-per-page varies by Kindle model, font size, book formatting
- Two calibration methods: total page count or current page position
- Stored in `/data/calibration.json`, keyed by ABS library_item_id
- Sanity-bounded to 50–600 words/page
- Default fallback: 250 words/page

### Favorites Persistence
- Stored as a flat `list[str]` of item IDs in `/data/favorites.json`
- Thread-safe via `_favorites_lock` (same pattern as calibrations)
- Frontend sorts favorites to the top, then alphabetical within each group
- Search filters across both favorited and non-favorited books

### EPUB Cache (Memory + Disk)
- `_epub_cache` (OrderedDict) holds parsed EPUB chapters in memory, keyed by item_id
- Bounded to 20 entries (LRU: oldest evicted)
- Backed by disk cache in `DATA_DIR/epub_cache/{item_id}.json`
- Lookup order: memory → disk → download + parse (then populate both caches)

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `AUDIOBOOKSHELF_URL` | ✅ | `http://localhost:13378` | ABS instance URL |
| `AUDIOBOOKSHELF_TOKEN` | ✅ | – | ABS API bearer token |
| `OPENAI_API_KEY` | ✅ | – | OpenAI API key (for GPT-4o-mini) |
| `LLM_MODEL` | – | `gpt-4o-mini` | OpenAI model for summaries |
| `SUMMARY_LANGUAGE` | – | `de` | Summary language (`de`, `en`) |
| `DATA_DIR` | – | `/data` | Path for calibration.json, favorites.json, epub_cache/ |
| `EPUB_MAX_SIZE_MB` | – | `100` | Max EPUB download size in MB |
| `AUTH_USER` | – | – | Basic auth username (leave empty to disable auth) |
| `AUTH_PASS` | – | – | Basic auth password |

## API Endpoints Reference

### `GET /api/books`
Returns all library items with `has_epub`, `is_calibrated`, and `is_favorite` flags.

### `GET /api/books/{item_id}/chapters`
Returns audio chapter markers (title, start, end in seconds).

### `POST /api/position`
**Body:** `{library_item_id, current_time_seconds}`
**Returns:** estimated_page, total_pages, percentage, chapter_title, nearby_text, is_calibrated, words_per_page

### `POST /api/recap`
**Body:** `{library_item_id, start_minutes, end_minutes, summary_style}`
**summary_style:** `"concise"` | `"detailed"` | `"bullet"`
**Returns:** summary, text_excerpt, chapters_covered, cost_estimate_usd

### `POST /api/find-text`
**Body:** `{library_item_id, query}`
Exact match first, then fuzzy (longest matching substring).
**Returns:** audio_timestamp_seconds, audio_timestamp_formatted, chapter_title, confidence, context

### `POST /api/page-to-audio`
**Body:** `{library_item_id, page_number}`
Uses calibrated or default words_per_page.
**Returns:** audio_timestamp_seconds, audio_timestamp_formatted, chapter_title, nearby_text

### `POST /api/calibrate/by-total`
**Body:** `{library_item_id, total_kindle_pages}`
Calculates: total_words / total_kindle_pages = words_per_page

### `POST /api/calibrate/by-page`
**Body:** `{library_item_id, kindle_page, audio_time_seconds}`
Maps audio position to word count, divides by kindle_page.

### `DELETE /api/calibration/{item_id}`
Resets calibration to default.

### `POST /api/favorites/{item_id}`
Adds the book to the favorites list. Returns `{is_favorite: true}`. Idempotent.

### `DELETE /api/favorites/{item_id}`
Removes the book from the favorites list. Returns `{is_favorite: false}`. Idempotent.

### `POST /api/sync-progress?library_item_id=...&time_seconds=...`
Updates the user's listening position in Audiobookshelf via `PATCH /api/me/progress/{id}`.

## Core Algorithms

### Time → Text Mapping (`map_time_to_text`)
1. Iterate audio chapters, find those overlapping [start_sec, end_sec]
2. For each overlapping audio chapter, find corresponding EPUB chapter by index
3. Calculate portion of chapter text based on time ratio within chapter
4. Snap to word boundaries
5. Fallback: proportional mapping across entire text if chapter mapping fails

### Time → Character Position (`_time_to_char_position`)
1. Find current audio chapter by timestamp
2. Calculate progress within that chapter (0.0–1.0)
3. Sum character counts of previous EPUB chapters + fraction of current
4. Returns: char_offset, chapter_title, chapter_progress_pct

### Character Position → Time (`_char_position_to_time`)
Reverse of above: find which EPUB chapter contains the char position, calculate progress ratio, apply to corresponding audio chapter's time range.

### Text Search (`find_text`)
1. Exact case-insensitive search in full EPUB text
2. If no match: sliding window over query words, find longest matching substring
3. Convert match position to audio timestamp via `_char_position_to_time`

## Coding Conventions

- **Language:** Python 3.12+, type hints used but not strict
- **Async:** All ABS API calls are async (httpx.AsyncClient), LLM calls are sync (openai SDK)
- **Error handling:** FastAPI HTTPException with German error messages for user-facing errors
- **Variable names in app.py:** Abbreviated in mapping functions for density (ec=epub_chapters, ac=audio_chapters, etc.), descriptive in route handlers
- **Frontend:** Minimal vanilla JS, no framework. CSS variables for theming. Short function/variable names in JS. Book list uses DOM API (createElement/textContent) instead of innerHTML to avoid XSS.
- **No tests yet** – this is a prototype

## Known Limitations & Issues

1. **Chapter mapping accuracy:** The index-based mapping (audio chapter i = EPUB chapter i) breaks when the EPUB has front matter (title page, copyright, TOC) that the audiobook skips. Title-based fallback matching exists but is imperfect.

2. **EPUB compatibility:** Only handles standard EPUB2/3 with OPF spine. DRM-protected EPUBs won't work. Some unusual EPUB structures may fail to parse.

3. **No EPUB3 Navigation Document support:** Currently relies on spine order + heading extraction for chapter titles, doesn't parse the nav document.

4. **Calibration assumes linear reading speed:** The audio narration speed is assumed constant. Variable-speed narration or books with many illustrations will reduce accuracy.

5. **Sync is one-way:** Can push audio position to ABS, but doesn't read/push Kindle position (Kindle doesn't have an open API).

## Roadmap / Open Tasks

### Completed
- [x] **Offset correction for front matter**: UI + API to set EPUB chapter offset, persisted in calibration.json.
- [x] **EPUB cache persistence**: Disk cache in `DATA_DIR/epub_cache/{item_id}.json`, with LRU in-memory layer.
- [x] **Support multiple EPUB files per book**: `_find_epub_file()` checks both `ebookFile` and `libraryFiles`.
- [x] **Basic auth**: Optional HTTP Basic auth via `AUTH_USER`/`AUTH_PASS` env vars.
- [x] **Search & Favorites**: Filter books by title/author; star books to pin them at the top.
- [x] **LLM retry with backoff**: `_call_llm()` retries transient errors with exponential backoff.
- [x] **Model-aware text chunking**: Long texts are split into chunks based on model context window, summarized in parallel, then merged.

### Medium Priority
- [ ] **Whispersync-style auto-detection**: If user's ABS progress changes, auto-show the new position without manual page load.
- [ ] **Reading speed calibration**: Instead of words-per-page, allow calibration by narration speed (words/minute audio) for better proportional mapping.

### Low Priority / Nice-to-Have
- [ ] **Bookmark integration**: Read/write ABS bookmarks for "I fell asleep here" markers.
- [ ] **History**: Store past recaps so users can review what was summarized.
- [ ] **Alternative LLM support**: Add Anthropic Claude, Ollama, or local LLM support as alternatives to OpenAI.
- [ ] **Mobile PWA**: Add manifest.json and service worker for installable PWA.
- [ ] **Localization**: UI is German-only; add i18n support.
- [ ] **Tests**: pytest for EPUB parsing, mapping logic, and API endpoints.
- [ ] **Split app.py**: Once the file grows beyond ~500 lines, split into modules (epub.py, mapping.py, abs_client.py, routes.py).

## Development Setup

```bash
# Clone / copy project
cd audiobook-recap

# Create venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set env vars
export AUDIOBOOKSHELF_URL=http://your-nas:13378
export AUDIOBOOKSHELF_TOKEN=your-token
export OPENAI_API_KEY=sk-...
export DATA_DIR=./data

# Run
uvicorn app:app --reload --host 0.0.0.0 --port 8765
```

## External API References

- **Audiobookshelf API**: https://api.audiobookshelf.org/ (note: docs are outdated but still useful)
- **OpenAI API**: https://platform.openai.com/docs/api-reference
- **EPUB spec**: https://www.w3.org/TR/epub-33/

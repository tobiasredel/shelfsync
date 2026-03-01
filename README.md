# ShelfSync

**Sync your audiobook and eBook positions** — like WhisperSync, but self-hosted for [Audiobookshelf](https://www.audiobookshelf.org/).

ShelfSync connects to your Audiobookshelf instance and uses the EPUB file of each audiobook to map between audio timestamps and text positions. Know exactly which Kindle page matches your current listening position — and vice versa.

## Features

### Position Sync (Primary)
- **Audio → Page**: See which Kindle page/percentage matches your current audiobook position
- **Page → Audio**: Enter a Kindle page number and jump to the corresponding audio timestamp
- **Text → Audio**: Enter a passage from your Kindle and find where it is in the audiobook
- **Kindle Calibration**: Set your total Kindle pages or calibrate with your current page for device-accurate mapping
- **Whisper Auto-Sync**: One-time per book — transcribes short audio segments via OpenAI Whisper to create high-accuracy anchor points (~$0.02/book)

### Find & Jump
- **Text Search**: Type any passage and find the exact audio position
- **OCR**: Take a photo of your Kindle page, extract text via vision API, and find the audio position
- **Sync to Audiobookshelf**: Jump the ABS player to any found position with one tap

### Recap (Bonus)
- **Summarize missed segments**: Fell asleep? Select a time range and get a GPT summary of what happened
- **Multiple styles**: Concise, detailed, or bullet points
- **Cost-efficient**: Uses EPUB text instead of Whisper transcription — ~$0.005 per recap (~14x cheaper)

### General
- **Currently Reading**: Pin a book as hero card for instant access
- **Mobile-first PWA**: Dark theme, bottom sheet UI, touch-optimized — designed for bedside use
- **Self-hosted**: Runs on Docker, targeted at Synology NAS and similar home servers

## How It Works

```
┌──────────────────────────────┐
│  ShelfSync (Browser PWA)     │
└──────────────┬───────────────┘
               │ HTTP/JSON
┌──────────────▼───────────────┐
│  FastAPI Backend (app.py)    │
│  - Position mapping          │
│  - Text search               │
│  - EPUB parsing              │
│  - Calibration management    │
└──────┬──────────┬────────────┘
       │          │
       ▼          ▼
  Audiobookshelf  OpenAI API
  (Library +      (GPT-4o-mini for
   EPUB files)     recaps + OCR)
```

1. Connects to your **Audiobookshelf** instance via API
2. Downloads and parses the **EPUB** file for each audiobook
3. Maps audio chapter markers to EPUB chapters using fuzzy title matching, boundary alignment, or WPM estimation
4. Interpolates between anchor points to convert any audio timestamp ↔ text position
5. Optional: **Whisper Auto-Sync** creates high-precision anchor points by transcribing short audio samples

## Prerequisites

- **Docker** (on Synology NAS, Raspberry Pi, or any server)
- **Audiobookshelf** with audiobooks that have matching EPUB files
- **OpenAI API Key** — [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Audiobookshelf API Token** — Settings → Users → your user → API Token

### EPUB + Audiobook Setup in Audiobookshelf

The EPUB file must be in the same folder as the audio files:

```
/audiobooks/
  Author Name/
    Book Title/
      chapter01.mp3
      chapter02.mp3
      booktitle.epub       ← this file
      cover.jpg
```

Audiobookshelf will automatically detect the EPUB and show it as an eBook option.

## Installation

### Docker Compose (Recommended)

```bash
mkdir -p shelfsync && cd shelfsync

# Create docker-compose.yml (see below) and configure your values
docker compose up -d
```

**docker-compose.yml:**

```yaml
services:
  shelfsync:
    image: ghcr.io/tobiasredel/shelfsync:latest
    container_name: shelfsync
    restart: unless-stopped
    ports:
      - "8765:8765"
    volumes:
      - ./data:/data
    environment:
      - AUDIOBOOKSHELF_URL=http://your-nas-ip:13378
      - AUDIOBOOKSHELF_TOKEN=your-abs-api-token
      - OPENAI_API_KEY=sk-your-openai-key
      # Optional:
      - LLM_MODEL=gpt-4o-mini
      - SUMMARY_LANGUAGE=de
```

Then open: `http://your-server:8765`

### Build from Source

```bash
git clone https://github.com/tobiasredel/shelfsync.git
cd shelfsync
docker compose up -d --build
```

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `AUDIOBOOKSHELF_URL` | Yes | `http://localhost:13378` | URL of your Audiobookshelf instance |
| `AUDIOBOOKSHELF_TOKEN` | Yes | — | Audiobookshelf API bearer token |
| `OPENAI_API_KEY` | Yes | — | OpenAI API key (for recaps, OCR, and Whisper sync) |
| `LLM_MODEL` | No | `gpt-4o-mini` | OpenAI model for summaries |
| `SUMMARY_LANGUAGE` | No | `de` | Summary language (`de`, `en`, etc.) |
| `DATA_DIR` | No | `/data` | Path for persistent data (calibration, cache) |
| `EPUB_MAX_SIZE_MB` | No | `100` | Maximum EPUB download size in MB |
| `AUTH_USER` | No | — | Basic auth username (leave empty to disable) |
| `AUTH_PASS` | No | — | Basic auth password |

## Usage

1. **Open ShelfSync** in your browser
2. **Select a book** from the library — it's added as "Currently Reading"
3. **Position tab** (default): See your current Kindle page and percentage based on your ABS listening progress
4. **Calibrate** (optional): Enter your total Kindle pages for accurate page numbers
5. **Find tab**: Enter text or a page number from your Kindle to find the audio position
6. **Recap tab**: Select a time range to summarize what you missed

### Whisper Auto-Sync (Optional)

For the most accurate position mapping:
1. Open the **Position tab** for your book
2. Scroll to the **Whisper-Sync** section
3. Select the book's language and click **Auto-Sync starten**
4. ShelfSync transcribes ~10 short audio segments and matches them to the EPUB
5. This creates high-precision anchor points — costs ~$0.02 per book, one-time

## Costs

| Feature | Cost | Notes |
|---|---|---|
| Position Sync | Free | No API calls needed |
| Recap (GPT-4o-mini) | ~$0.005 / recap | Uses EPUB text, not audio transcription |
| Whisper Auto-Sync | ~$0.02 / book | One-time calibration |
| OCR (Vision API) | ~$0.01 / photo | Kindle page text extraction |

**Estimated monthly cost: ~$0.50–1.00** with daily use.

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python 3.12, FastAPI, uvicorn |
| HTTP Client | httpx (async) |
| EPUB Parsing | Python stdlib (zipfile + xml.etree) |
| LLM | OpenAI API (GPT-4o-mini default) |
| Frontend | Vanilla HTML/CSS/JS (single file, no build step) |
| Deployment | Docker, multi-arch (amd64 + arm64) |

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/books` | GET | List all library items |
| `/api/books/{id}/details` | GET | Book details with EPUB/calibration status |
| `/api/position` | POST | Audio time → Kindle page/percentage |
| `/api/find-text` | POST | Text query → audio timestamp |
| `/api/page-to-audio` | POST | Kindle page → audio timestamp |
| `/api/recap` | POST | Time range → GPT summary |
| `/api/sync-progress` | POST | Update ABS listening position |
| `/api/whisper-sync/{id}` | POST | Start Whisper auto-calibration |
| `/api/calibrate/*` | POST | Manual calibration endpoints |
| `/api/health` | GET | Health check |

## Mapping Quality

ShelfSync uses a cascade of strategies to map audio timestamps to text positions:

1. **Whisper anchors** — Highest accuracy (if auto-sync was run)
2. **Chapter title matching** — Fuzzy matching of audio and EPUB chapter names
3. **Boundary alignment** — Matches chapters by relative position in the book
4. **WPM estimation** — Estimates narration speed for proportional mapping

Quality indicator badges in the UI show which strategy is active for each book.

## Troubleshooting

| Problem | Solution |
|---|---|
| "No EPUB found" | Place the EPUB in the same folder as audio files in Audiobookshelf |
| "Could not extract text" | Some EPUBs have unusual structure; DRM-protected EPUBs won't work |
| Position seems off | Calibrate with your Kindle page count, or run Whisper Auto-Sync |
| Connection error | Verify `AUDIOBOOKSHELF_URL` is reachable from the Docker container |

## License

MIT

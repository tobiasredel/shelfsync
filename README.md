# 🎧 Audiobook Recap – "Was hab ich verpasst?"

Eingeschlafen beim Hörbuch? Dieses Tool fasst zusammen, was du verpasst hast.

## Wie es funktioniert

1. Verbindet sich mit deiner **Audiobookshelf**-Instanz
2. Holt die **EPUB**-Datei des Hörbuchs
3. Nutzt die **Audio-Kapitelmarker**, um den Zeitraum auf den EPUB-Text zu mappen
4. Schickt den Text an **GPT-4o-mini** für eine Zusammenfassung

Kein Whisper/Speech-to-Text nötig → **~14x günstiger** als Audio-Transkription.

## Kosten

| Komponente | Preis | Beispiel (10 min) |
|---|---|---|
| GPT-4o-mini | ~$0.00015/1K tokens | **~$0.005** |
| Whisper (v1, zum Vergleich) | $0.006/min | $0.06 |

**Geschätzt: ~$0.50–1.00/Monat** bei täglicher Nutzung.

## Voraussetzungen

- Docker auf Synology NAS (oder anderem Server)
- Audiobookshelf mit Hörbüchern **und zugehörigen EPUBs**
- OpenAI API Key → [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- ABS API Token (Settings → Users → API Token)

### EPUB + Hörbuch in Audiobookshelf

Die EPUB muss im selben Ordner wie die Audiodateien liegen:

```
/audiobooks/
  Autor Name/
    Buchtitel/
      chapter01.mp3
      chapter02.mp3
      buchtitel.epub       ← diese Datei
      cover.jpg
```

Audiobookshelf erkennt die EPUB automatisch und zeigt sie als eBook-Option an.

## Installation

```bash
mkdir -p /volume1/docker/audiobook-recap
cd /volume1/docker/audiobook-recap

# Dateien hierhin kopieren
# docker-compose.yml anpassen (ABS URL, Token, OpenAI Key)

docker-compose up -d --build
```

Dann im Browser: `http://dein-nas:8765`

## Konfiguration

| Variable | Beschreibung | Default |
|---|---|---|
| `AUDIOBOOKSHELF_URL` | URL der ABS-Instanz | `http://localhost:13378` |
| `AUDIOBOOKSHELF_TOKEN` | ABS API Token | – |
| `OPENAI_API_KEY` | OpenAI API Key | – |
| `LLM_MODEL` | LLM-Modell | `gpt-4o-mini` |
| `SUMMARY_LANGUAGE` | Sprache | `de` |

## Features

- **Buch-Auswahl** mit Cover, Fortschritt und EPUB-Status
- **Kapitel-Übersicht** – klicke ein Kapitel um den Zeitraum zu setzen
- **Schnellauswahl** – "Letzte 5/10/15/20/30 min"
- **3 Zusammenfassungs-Stile** – Kurz, Ausführlich, Stichpunkte
- **Kostenlos lesbar** – der extrahierte Text ist aufklappbar
- **Docker-Image** nur ~70MB (kein ffmpeg nötig)

## Genauigkeit des Kapitel-Mappings

Das Mapping von Audio-Zeit → EPUB-Text basiert auf den Kapitelmarkern in ABS. Es ist nicht perfekt zeichengenau, aber für "Was ist passiert?" völlig ausreichend. Bei Büchern ohne Kapitelmarker wird ein proportionales Mapping genutzt (Position im Buch ≈ Position im Text).

## Troubleshooting

**"Kein EPUB gefunden"**: Die EPUB muss im selben Ordner wie die Audiodateien liegen und von ABS erkannt worden sein.

**"Konnte keinen Text extrahieren"**: Manche EPUBs haben ungewöhnliche Strukturen. DRM-geschützte EPUBs funktionieren nicht.

**Zusammenfassung passt nicht genau**: Das Zeit-zu-Text-Mapping ist eine Schätzung. Bei Büchern mit vielen Kapitelmarkern ist es genauer.

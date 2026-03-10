"""
FR-Moshi — Étape 1 : Téléchargement de podcasts/interviews français
Utilise yt-dlp pour télécharger de l'audio conversationnel français.

Prérequis: pip install yt-dlp
"""

import subprocess
import json
import os
from pathlib import Path

OUTPUT_DIR = Path("data/raw_audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SOURCES DE PODCASTS FRANÇAIS CONVERSATIONNELS
# Critères : dialogues naturels, 2+ locuteurs, français spontané
# =============================================================================

SOURCES = {
    # --- Podcasts conversationnels populaires ---
    "france_inter_interviews": [
        # France Inter - Le 7/9 (interviews)
        # Ajouter des URLs de playlists ou vidéos spécifiques ici
    ],
    "france_culture_discussions": [
        # France Culture - Les Matins, La Grande Table
    ],
    "thinkerview": [
        # Interviews longues, 2 locuteurs, très naturel
        # https://www.youtube.com/@Thinkerview/videos
    ],
    "clique_tv": [
        # Interviews conversationnelles
    ],
    "podcasts_independants": [
        # Podcasts indépendants français avec discussions
    ],
}

# =============================================================================
# PLAYLISTS / CHAÎNES À SCRAPER
# Format: (URL, max_videos, description)
# =============================================================================

CHANNELS = [
    # Interviews longues et naturelles (priorité haute)
    ("https://www.youtube.com/@Thinkerview/videos", 200, "Thinkerview - interviews longues"),
    ("https://www.youtube.com/@music2music/videos", 50, "Conversations musicales"),

    # Débats et discussions (plusieurs locuteurs)
    ("https://www.youtube.com/@franceculture/videos", 100, "France Culture"),
    ("https://www.youtube.com/@FranceInter/videos", 100, "France Inter"),

    # Talk-shows et discussions informelles
    ("https://www.youtube.com/@Quotidien/videos", 50, "Quotidien - discussions"),
    ("https://www.youtube.com/@Brut/videos", 50, "Brut - interviews courtes"),
]


def download_channel(url: str, max_videos: int, description: str):
    """Télécharge l'audio d'une chaîne YouTube."""
    print(f"\n{'='*60}")
    print(f"Téléchargement: {description}")
    print(f"URL: {url}")
    print(f"Max vidéos: {max_videos}")
    print(f"{'='*60}")

    safe_name = description.replace(" ", "_").replace("-", "_")[:30]
    output_template = str(OUTPUT_DIR / f"{safe_name}_%(title).50s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--max-downloads", str(max_videos),
        # Filtrer: vidéos de 5-120 minutes (conversations)
        "--match-filter", "duration > 300 & duration < 7200",
        # Langue française
        "--match-filter", "language = fr OR language = French",
        "--output", output_template,
        "--no-overwrites",
        "--ignore-errors",
        "--quiet",
        "--progress",
        url,
    ]

    try:
        subprocess.run(cmd, check=False)
        print(f"  ✓ Terminé: {description}")
    except FileNotFoundError:
        print("  ✗ yt-dlp non trouvé. Installer avec: pip install yt-dlp")
        return


def convert_to_24khz(input_dir: Path, output_dir: Path):
    """Convertit tous les WAV en 24kHz mono (requis par Mimi)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_files = list(input_dir.glob("*.wav"))
    print(f"\nConversion de {len(wav_files)} fichiers en 24kHz mono...")

    for wav in wav_files:
        output_path = output_dir / wav.name
        if output_path.exists():
            continue
        cmd = [
            "ffmpeg", "-i", str(wav),
            "-ar", "24000",    # 24kHz (requis par Mimi)
            "-ac", "1",        # Mono (sera converti en stéréo après diarisation)
            "-y",              # Écraser si existe
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  ✗ Erreur conversion {wav.name}: {e}")

    print(f"  ✓ {len(wav_files)} fichiers convertis")


def create_manifest(audio_dir: Path, output_path: Path):
    """Crée le manifeste des fichiers audio avec durées."""
    import wave

    manifest = []
    for wav in sorted(audio_dir.glob("*.wav")):
        try:
            with wave.open(str(wav), 'r') as w:
                duration = w.getnframes() / w.getframerate()
                if duration >= 60:  # Minimum 1 minute
                    manifest.append({
                        "path": str(wav),
                        "duration": round(duration, 2),
                        "filename": wav.name,
                    })
        except Exception as e:
            print(f"  ✗ Erreur lecture {wav.name}: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in manifest:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    total_hours = sum(e["duration"] for e in manifest) / 3600
    print(f"\nManifeste créé: {output_path}")
    print(f"  {len(manifest)} fichiers, {total_hours:.1f} heures au total")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Télécharger des podcasts français")
    parser.add_argument("--channels-only", type=int, default=None,
                        help="Ne télécharger que N chaînes (pour test)")
    parser.add_argument("--convert", action="store_true",
                        help="Convertir en 24kHz après téléchargement")
    parser.add_argument("--manifest", action="store_true",
                        help="Créer le manifeste des fichiers")
    args = parser.parse_args()

    channels = CHANNELS[:args.channels_only] if args.channels_only else CHANNELS

    # Téléchargement
    for url, max_vids, desc in channels:
        download_channel(url, max_vids, desc)

    # Conversion 24kHz
    if args.convert:
        convert_dir = Path("data/audio_24khz")
        convert_to_24khz(OUTPUT_DIR, convert_dir)

    # Manifeste
    if args.manifest:
        audio_dir = Path("data/audio_24khz") if args.convert else OUTPUT_DIR
        create_manifest(audio_dir, Path("data/raw_manifest.jsonl"))

    print("\n✓ Étape 1 terminée.")
    print("Prochaine étape: python scripts/02_diarize_stereo.py")

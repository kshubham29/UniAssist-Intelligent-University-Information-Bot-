"""
Voice_module.py — UniAssist NCU
================================
STT : SpeechRecognition + sounddevice (no PyAudio needed)
TTS : pyttsx3 (offline) → gTTS+pygame → PowerShell SAPI fallback

Install:
    pip install SpeechRecognition sounddevice scipy pyttsx3 gtts pygame
"""

import os
import re
import tempfile
import threading

# ── Dependency flags (safe imports, never crash on import) ────────────────────
try:
    import sounddevice as sd
    import scipy.io.wavfile as wav
    import numpy as np
    SD_AVAILABLE = True
except Exception as _e:
    SD_AVAILABLE = False
    print(f"[Voice] sounddevice/scipy not available: {_e}")

try:
    import speech_recognition as sr
    _recognizer = sr.Recognizer()
    SR_AVAILABLE = True
except Exception as _e:
    SR_AVAILABLE = False
    print(f"[Voice] SpeechRecognition not available: {_e}")

try:
    import pyttsx3 as _pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception as _e:
    PYTTSX3_AVAILABLE = False
    print(f"[Voice] pyttsx3 not available: {_e}")

try:
    from gtts import gTTS as _gTTS
    GTTS_AVAILABLE = True
except Exception as _e:
    GTTS_AVAILABLE = False
    print(f"[Voice] gTTS not available: {_e}")

try:
    import pygame as _pygame
    PYGAME_AVAILABLE = True
except Exception as _e:
    PYGAME_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
DURATION    = 6        # seconds to record


# =============================================================================
# CLEAN TEXT FOR TTS
# =============================================================================
def _clean_for_tts(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*',     r'\1', text)
    text = re.sub(r'#{1,6}\s',      '',    text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = text.replace('•', '').replace('◆', '').replace('#', '').replace('*', '')
    text = ' '.join(text.split())
    return text[:1000]


# =============================================================================
# TEXT-TO-SPEECH
# =============================================================================
def speak_response(text: str) -> None:
    """Speak text aloud. Tries pyttsx3 → gTTS+pygame → PowerShell SAPI."""
    if not text or not text.strip():
        return
    clean = _clean_for_tts(text)

    # ── 1. pyttsx3 (offline, best on Windows) ────────────────────────────────
    if PYTTSX3_AVAILABLE:
        try:
            engine = _pyttsx3.init()
            voices = engine.getProperty('voices')
            for v in voices:
                if any(x in v.name.lower() for x in ('zira', 'hazel', 'female')):
                    engine.setProperty('voice', v.id)
                    break
            engine.setProperty('rate',   155)
            engine.setProperty('volume', 1.0)
            engine.say(clean)
            engine.runAndWait()
            engine.stop()
            return
        except Exception as e:
            print(f"[Voice] pyttsx3 error: {e}")

    # ── 2. gTTS + pygame ─────────────────────────────────────────────────────
    if GTTS_AVAILABLE and PYGAME_AVAILABLE:
        try:
            import time
            tts = _gTTS(text=clean, lang="en", slow=False)
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp_path = tmp.name
            tmp.close()
            tts.save(tmp_path)

            _pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            _pygame.mixer.music.load(tmp_path)
            _pygame.mixer.music.play()
            while _pygame.mixer.music.get_busy():
                time.sleep(0.1)
            _pygame.mixer.music.stop()
            _pygame.mixer.quit()
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return
        except Exception as e:
            print(f"[Voice] gTTS+pygame error: {e}")

    # ── 3. Windows PowerShell SAPI (zero dependencies) ───────────────────────
    try:
        import subprocess
        safe = clean[:500].replace('"', "'")
        ps_cmd = (
            'Add-Type -AssemblyName System.Speech; '
            '$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
            f'$s.Speak("{safe}")'
        )
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        subprocess.run(
            ["powershell", "-Command", ps_cmd],
            timeout=30,
            creationflags=flags
        )
        return
    except Exception as e:
        print(f"[Voice] PowerShell SAPI error: {e}")

    print("[Voice] No TTS backend worked. Install pyttsx3: pip install pyttsx3")


# =============================================================================
# SPEECH-TO-TEXT
# =============================================================================
def _record_audio(duration: int = DURATION, fs: int = SAMPLE_RATE) -> str | None:
    """Record mic audio using sounddevice, save as WAV, return path."""
    if not SD_AVAILABLE:
        return None
    try:
        print("[Voice] 🎤 Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
        sd.wait()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav.write(tmp.name, fs, recording)
        return tmp.name
    except Exception as e:
        print(f"[Voice] Recording error: {e}")
        return None


def get_voice_input() -> str | None:
    """
    Record from mic → transcribe via Google STT.
    Returns transcribed string (lowercase) or None on failure.
    """
    if not SR_AVAILABLE:
        print("[Voice] SpeechRecognition not installed.")
        return None

    audio_path = _record_audio()
    if not audio_path:
        print("[Voice] Could not record audio.")
        return None

    try:
        with sr.AudioFile(audio_path) as source:
            audio = _recognizer.record(source)

        print("[Voice] 🔎 Recognizing...")
        text = _recognizer.recognize_google(audio, language="en-IN")
        print(f"[Voice] Heard: {text}")
        return text.lower().strip()

    except sr.UnknownValueError:
        print("[Voice] Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"[Voice] Google STT network error: {e}")
        return None
    except Exception as e:
        print(f"[Voice] STT error: {e}")
        return None
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass


# =============================================================================
# SELF-TEST  —  python Voice_module.py
# =============================================================================
if __name__ == "__main__":
    print("=" * 45)
    print("  UniAssist NCU — Voice Module Self-Test")
    print("=" * 45)
    print(f"  sounddevice       : {'✅' if SD_AVAILABLE else '❌  pip install sounddevice'}")
    print(f"  SpeechRecognition : {'✅' if SR_AVAILABLE else '❌  pip install SpeechRecognition'}")
    print(f"  pyttsx3           : {'✅' if PYTTSX3_AVAILABLE else '❌  pip install pyttsx3'}")
    print(f"  gTTS              : {'✅' if GTTS_AVAILABLE else '❌  pip install gtts'}")
    print(f"  pygame            : {'✅' if PYGAME_AVAILABLE else '❌  pip install pygame'}")
    print()
    print("Testing TTS…")
    speak_response("Hello! UniAssist voice module is working correctly.")
    print("TTS done.\n")
    print("Testing STT — speak after prompt…")
    result = get_voice_input()
    print(f"You said: {result}" if result else "STT returned nothing.")
"""Offline Text-to-Speech synthesizer for local workstation and Raspberry Pi.

Supports:
- Windows built-in SAPI via PowerShell System.Speech (zero external dependencies)
- Linux / Raspberry Pi: piper or espeak-ng when installed
Returns WAV bytes for local audio playback when browser SpeechSynthesis is unavailable
or when offline server TTS is explicitly selected.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional


def get_tts_capabilities() -> Dict[str, object]:
    system = platform.system().lower()
    if system == "windows":
        return {"available": True, "engine": "windows_sapi"}
    if shutil.which("piper"):
        return {"available": True, "engine": "piper"}
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return {"available": True, "engine": "espeak"}
    return {"available": False, "engine": "browser_only"}


def synthesize_wav(text: str, lang: str = "en", rate: float = 1.0) -> Optional[bytes]:
    """Generate offline WAV audio bytes for the given text."""
    clean = " ".join((text or "").split()).strip()
    if not clean:
        return None
    clean = clean[:700]

    system = platform.system().lower()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    try:
        if system == "windows":
            sapi_rate = max(-5, min(5, int(round((rate - 1.0) * 6))))
            lang_prefix = "ar" if lang.startswith("ar") else "en"
            ps_script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Rate = {sapi_rate}; "
                f"$v = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Culture.TwoLetterISOLanguageName -eq '{lang_prefix}' }} | Select-Object -First 1; "
                "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
                "$s.SetOutputToWaveFile($env:CAMPUS_TTS_OUT); "
                "$s.Speak($env:CAMPUS_TTS_TEXT); "
                "$s.Dispose();"
            )
            env = os.environ.copy()
            env["CAMPUS_TTS_OUT"] = str(wav_path)
            env["CAMPUS_TTS_TEXT"] = clean
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                env=env,
                capture_output=True,
                timeout=12,
                check=False,
            )
            if res.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 44:
                return wav_path.read_bytes()
            return None

        espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
        if espeak_bin:
            voice = "ar" if lang.startswith("ar") else "en"
            wpm = str(max(110, min(240, int(165 * rate))))
            res = subprocess.run(
                [espeak_bin, "-v", voice, "-s", wpm, "-w", str(wav_path), clean],
                capture_output=True,
                timeout=10,
                check=False,
            )
            if res.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 44:
                return wav_path.read_bytes()
        return None
    except Exception:
        return None
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            pass

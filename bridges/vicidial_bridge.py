"""
Vicidial & Asterisk Audio Bridge Helper.
Provides dialplan generators, channel configuration utilities, and health monitors.
"""

import sys
import json
import asyncio
import urllib.request
from typing import Dict, Any, Optional


class VicidialAudioBridgeHelper:
    """
    Assists in configuring and bridging Vicidial Call Center campaigns with OmniVoice TTS.
    """

    @staticmethod
    def generate_asterisk_dialplan(
        context: str = "vicidial-ai-bot",
        extension: str = "8300",
        audiosocket_host: str = "127.0.0.1",
        audiosocket_port: int = 9092,
        tts_http_url: str = "http://127.0.0.1:8000/v1/tts/synthesize"
    ) -> str:
        """
        Generates standard Asterisk extensions.conf dialplan snippet for Vicidial.
        """
        dialplan = f"""
; ====================================================================
; OmniVoice TTS & AI Voice Bot Integration Dialplan for Vicidial
; Place in /etc/asterisk/extensions.conf or included custom context
; ====================================================================

[{context}]
; Route inbound/transferred Vicidial call to AI Voice Bot via AudioSocket
exten => {extension},1,NoOp(Starting OmniVoice AI Voice Bot for Vicidial Channel: ${{CHANNEL}})
 same => n,Answer()
 same => n,Set(CHANNEL(audioreadformat)=slin)
 same => n,Set(CHANNEL(audiowriteformat)=slin)
 same => n,Set(CALL_UUID=${{UUID}})
 ; Connect bidirectional 8kHz Linear PCM audio channel to OmniVoice AudioSocket server
 same => n,AudioSocket(${{CALL_UUID}},{audiosocket_host}:{audiosocket_port})
 same => n,Hangup()

; Fallback HTTP playback macro for simple survey or IVR prompt playout
exten => 8301,1,NoOp(Playing cached/synthesized greeting via HTTP)
 same => n,Answer()
 same => n,EAGI(/var/lib/asterisk/agi-bin/omnivoice_play.py,"Hello! Thank you for calling Clara support. How can I help you today?")
 same => n,Hangup()
"""
        return dialplan.strip()

    @staticmethod
    def check_tts_service_health(tts_url: str = "http://127.0.0.1:8000/health") -> Dict[str, Any]:
        """Checks the reachability and GPU/CPU health of the OmniVoice TTS service."""
        try:
            req = urllib.request.Request(tts_url, headers={"User-Agent": "VicidialBridge/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {"status": "ok", "response": data}
        except Exception as e:
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    helper = VicidialAudioBridgeHelper()
    print("--- Sample Vicidial Asterisk Dialplan ---")
    print(helper.generate_asterisk_dialplan())
    print("\n--- Checking Local TTS Engine Health ---")
    health = helper.check_tts_service_health()
    print(json.dumps(health, indent=2))

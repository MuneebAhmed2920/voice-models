#!/usr/bin/env python3
"""
examples/generate_nayatel_american_technical_suite.py

Official Production Suite: High-Complexity American English Technical Audio Suite for Nayatel.
Clean dedicated folder: outputs/nayatel_american_technical_suite/

Each recording covers a distinct domain with difficult, industry-standard American English technical vocabulary:
1. Optical Fiber & ONT Hardware Diagnostics (ONT, PON light, LOS light, yellow patch cord, sharp bend, power cycle, optical power meter, field tech team)
2. Dual-Band Wi-Fi Optimization & Latency (Dual-band Wi-Fi router, 5 GHz SSID, 2.4 GHz channel congestion, 80 MHz channel width, backend auto channel, ping spikes, jitter, packet loss, speed throttle, Ookla speed test)
3. Enterprise Static IP & CCTV Port Forwarding Configuration (Public Static IP, subnet mask, default gateway, Google DNS 8.8.8.8, CCTV NVR, port forwarding rules, high-definition camera streams, cell phone app)
4. Value Added Services — JoyBox 4K Streaming & NWatch Cloud Surveillance (JoyBox 4K, video-on-demand buffer, 4K HDMI cable, app cache clear, NWatch cloud surveillance, 30 days HD recording backup, real-time AI motion detection alerts, 24 hours accessible)
5. Corporate Invoicing, FBR/PRA Tax Exemption & Digital Payments (Monthly billing statement, corporate account, FBR and PRA Section 236 withholding tax exemption certificate, Nayatel Customer Portal, 6-digit 1Link Consumer ID, Kuickpay, cell phone banking app)
6. Optical Backbone Fiber Cut Outage Advisory (Unplanned network outage, civil road construction, main optical backbone, emergency technical team, OTDR testing van, fusion splicing, 2-hour SLA restoration timeline, real-time SMS status updates)

Acoustic Specifications:
- American English technical terminology & pronunciation
- Speed: 0.88 (deliberate, polite, empathetic Pakistani helpline cadence)
- Diffusion Steps: 36 (natural vocal warmth, breath texture, acoustic clarity)
- Guidance Scale: 1.85 (balanced micro-prosody)
- Both 24kHz Studio Master HD & 8kHz PSTN Telephony Filtered outputs
"""

import os
import sys
import time
import shutil
import whisper

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_ROOT)

from runtime.engine import UrduVoiceModel

ARTIFACT_DIR = "/Users/haseebahmed/.gemini/antigravity-ide/brain/633db7cf-406a-4a03-8bd5-e832e3f6af11/nayatel_american_technical_suite"

AMERICAN_TECH_CATEGORIES = {
    "nayatel_american_tech_ont_diagnostics": {
        "title": "Category 1: Optical Fiber & ONT Diagnostics (LOS / Red Blink / Patch Cord / Power Cycle)",
        "domain": "Optical Hardware, Physical Layer & Attenuation Diagnostics",
        "american_terms": [
            "ONT router", "PON light", "LOS light red blink", "Signal drop",
            "Yellow fiber patch cord", "Sharp bend", "Power cycle (30 seconds)",
            "Optical power drop", "Field tech team", "Optical power meter"
        ],
        "script": (
            "اَلسَّلَامُ عَلَیْکُم، Nayatel Technical Support۔ "
            "اگر آپ کے ONT router پر PON light آف ہے اور LOS light red blink کر رہی ہے، "
            "تو آپ کی optical line میں signal drop ہو چکا ہے۔ "
            "برائے مہربانی yellow fiber patch cord کو check کریں کہ کہیں اس میں sharp bend یا twist نہ ہو، "
            "اور ONT کو 30 seconds کے لیے power cycle کریں۔ "
            "اگر optical power drop رہے، تو ہماری field tech team optical power meter کے ساتھ فوری طور پر تشریف لائے گی۔"
        )
    },
    "nayatel_american_tech_wifi_latency_optimization": {
        "title": "Category 2: Dual-Band Wi-Fi Optimization (5 GHz vs 2.4 GHz / Auto Channel / Ookla Speed Test)",
        "domain": "Wireless RF, Channel Congestion, Gaming Latency & Speed Optimization",
        "american_terms": [
            "Dual-band Wi-Fi router", "5 GHz SSID", "2.4 GHz frequency",
            "Channel congestion", "Speed throttle", "Backend auto channel",
            "80 MHz channel width", "Ping spikes", "Jitter", "Packet loss", "Ookla speed test"
        ],
        "script": (
            "جی، اگر آپ کو gaming یا video conferencing کے دوران ping spikes، jitter یا packet loss کا سامنا ہے، "
            "تو verify کریں کہ آپ 5 GHz Wi-Fi SSID سے connect ہیں۔ "
            "2.4 GHz frequency پر channel congestion کی وجہ سے speed throttle ہو سکتی ہے۔ "
            "ہم نے backend سے آپ کے router کا auto channel 80 MHz channel width پر optimize کر دیا ہے۔ "
            "آپ Ookla speed test run کر کے download اور upload bandwidth confirm کر لیں۔"
        )
    },
    "nayatel_american_tech_static_ip_cctv_configuration": {
        "title": "Category 3: Enterprise Static IP & CCTV Configuration (Port Forwarding / NVR Streams / Cell Phone App)",
        "domain": "Enterprise Networking, Public IP Allocation & Remote Video Surveillance",
        "american_terms": [
            "Enterprise Support", "Public Static IP", "Subnet mask",
            "Default gateway", "Google DNS (8.8.8.8)", "CCTV NVR",
            "Port forwarding rules", "Cell phone app", "High-definition camera streams"
        ],
        "script": (
            "اَلسَّلَامُ عَلَیْکُم، Nayatel Enterprise Support۔ "
            "آپ کے router کے لیے public Static IP، subnet mask اور default gateway configure ہو چکے ہیں، "
            "اور primary DNS Google 8.8.8.8 پر set کر دیا گیا ہے۔ "
            "آپ کے CCTV NVR کے لیے port forwarding rules بھی successfully apply ہو گئے ہیں۔ "
            "اب آپ office network کے باہر سے cell phone app پر تمام high-definition camera streams securely monitor کر سکتے ہیں۔"
        )
    },
    "nayatel_american_tech_vas_joybox_nwatch": {
        "title": "Category 4: Value Added Services — JoyBox 4K Streaming & NWatch Cloud Surveillance",
        "domain": "Smart Hardware, Video-on-Demand & AI Cloud Video Security",
        "american_terms": [
            "JoyBox 4K", "Video-on-demand buffer", "4K HDMI cable",
            "App cache clear", "NWatch cloud surveillance", "30 days HD recording backup",
            "Real-time AI motion detection alerts", "Cell phone app", "24 hours accessible"
        ],
        "script": (
            "جی، Nayatel JoyBox اور NWatch Cloud Services کے حوالے سے رہنمائی حاضر ہے۔ "
            "اگر JoyBox پر Live TV یا video-on-demand content buffer کر رہا ہے، "
            "تو 4K HDMI cable check کریں اور settings میں جا کر app cache clear کریں۔ "
            "مزید یہ کہ NWatch cloud surveillance کی 30 days HD recording backup اور real-time AI motion detection alerts آپ کے cell phone app پر 24 hours accessible ہیں۔"
        )
    },
    "nayatel_american_tech_corporate_billing_tax": {
        "title": "Category 5: Corporate Invoicing & FBR/PRA Tax Exemption (1Link Consumer ID / Kuickpay)",
        "domain": "Financial Administration, Corporate Invoicing, Tax Compliance & Digital Payments",
        "american_terms": [
            "Monthly billing statement", "Corporate account", "FBR and PRA Section 236 withholding tax",
            "Exemption certificate upload", "Nayatel Customer Portal", "6-digit 1Link Consumer ID",
            "Kuickpay", "Cell phone banking app"
        ],
        "script": (
            "آپ کے corporate account کا monthly billing statement generate ہو چکا ہے۔ "
            "FBR اور PRA Section 236 withholding tax exemption کے لیے آپ exemption certificate Nayatel Customer Portal پر upload کر سکتے ہیں۔ "
            "Invoice کی payment کے لیے اپنا 6-digit 1Link Consumer ID استعمال کریں، "
            "اور Kuickpay یا cell phone banking app کے ذریعے بغیر کسی extra service charge کے payment clear کر لیں۔"
        )
    },
    "nayatel_american_tech_backbone_outage_advisory": {
        "title": "Category 6: Backbone Fiber Cut Outage Advisory (Unplanned Outage / OTDR / Fusion Splicing)",
        "domain": "Critical Network Emergency, Backbone Cable Severance & Emergency Field Restoration",
        "american_terms": [
            "Unplanned network outage (American telecom standard)", "Civil road construction",
            "Main optical backbone fiber", "Emergency technical team", "OTDR testing van",
            "Core fiber fusion splicing", "SLA restoration timeline", "Real-time SMS updates"
        ],
        "script": (
            "اہم نیٹ ورک الرٹ: آپ کے ایریا میں civil road construction کے دوران Nayatel کی main optical backbone fiber کٹ جانے کی وجہ سے unplanned network outage کا سامنا ہے۔ "
            "ہماری emergency technical team OTDR testing van کے ساتھ spot پر موجود ہے اور core fiber کی fusion splicing جاری ہے۔ "
            "ہمارا estimated SLA restoration timeline 2 گھنٹے ہے۔ "
            "آپ کو cell phone پر SMS کے ذریعے real-time status update ملتی رہے گی۔"
        )
    }
}

def main():
    print("=" * 88)
    print("🎙️ Generating Nayatel American English Technical Audio Suite (36 Steps, Speed 0.88)")
    print("=" * 88)

    outputs_dir = os.path.join(PACKAGE_ROOT, "outputs", "nayatel_american_technical_suite")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print(f"\n[1/3] Initializing Mahrukh Urdu Voice Model...")
    model = UrduVoiceModel()

    print(f"\n[2/3] Initializing Whisper ASR Model for Verification...")
    whisper_asr = whisper.load_model("base")

    results = []

    print(f"\n[3/3] Generating 6 Advanced American English Technical Domain Scenarios...")
    for key, data in AMERICAN_TECH_CATEGORIES.items():
        title = data["title"]
        domain = data["domain"]
        script = data["script"]
        terms = ", ".join(data["american_terms"])

        print("\n" + "=" * 78)
        print(f">>> {title}")
        print(f"    Domain:         {domain}")
        print(f"    American Terms: {terms}")
        print(f"    Script:         {script}")

        # 1. 24kHz HD Studio Master (Studio broadcast audio)
        out_24k = os.path.join(outputs_dir, f"{key}_24k_hd.wav")
        t0 = time.time()
        aud_24k, sr_24k = model.synthesize(
            script,
            speed=0.88,
            num_step=36,
            guidance_scale=1.85,
            sample_rate=24000,
            output_file=out_24k
        )
        dur_24k = len(aud_24k) / float(sr_24k)
        t_gen_24k = time.time() - t0

        asr_res_24k = whisper_asr.transcribe(out_24k)
        asr_24k = asr_res_24k["text"].strip()
        print(f"    [HD 24kHz]  Saved: {out_24k} ({dur_24k:.2f}s, gen: {t_gen_24k:.2f}s)")
        print(f"                Whisper ASR: {asr_24k}")

        # Sync to artifacts
        art_24k = os.path.join(ARTIFACT_DIR, f"{key}_24k_hd.wav")
        shutil.copyfile(out_24k, art_24k)

        # 2. 8kHz PSTN Telephony Filtered Audio (Call center line standard)
        out_8k = os.path.join(outputs_dir, f"{key}_8k_telephony.wav")
        t1 = time.time()
        aud_8k, sr_8k = model.synthesize(
            script,
            speed=0.88,
            num_step=36,
            guidance_scale=1.85,
            sample_rate=8000,
            telephony_filter=True,
            output_file=out_8k
        )
        dur_8k = len(aud_8k) / float(sr_8k)
        t_gen_8k = time.time() - t1

        asr_res_8k = whisper_asr.transcribe(out_8k)
        asr_8k = asr_res_8k["text"].strip()
        print(f"    [PSTN 8kHz] Saved: {out_8k} ({dur_8k:.2f}s, gen: {t_gen_8k:.2f}s)")
        print(f"                Whisper ASR: {asr_8k}")

        # Sync to artifacts
        art_8k = os.path.join(ARTIFACT_DIR, f"{key}_8k_telephony.wav")
        shutil.copyfile(out_8k, art_8k)

        results.append({
            "key": key,
            "title": title,
            "domain": domain,
            "terms": data["american_terms"],
            "script": script,
            "file_24k": out_24k,
            "art_24k": art_24k,
            "dur_24k": dur_24k,
            "asr_24k": asr_24k,
            "file_8k": out_8k,
            "art_8k": art_8k,
            "dur_8k": dur_8k,
            "asr_8k": asr_8k
        })

    print("\n" + "=" * 88)
    print("✅ All 6 American English Technical Nayatel Recordings Successfully Generated & Synced!")
    print("=" * 88)
    for r in results:
        print(f"• {r['title']}")
        print(f"    24kHz HD:       {r['file_24k']} ({r['dur_24k']:.2f}s)")
        print(f"    8kHz Telephony: {r['file_8k']} ({r['dur_8k']:.2f}s)")

if __name__ == "__main__":
    main()

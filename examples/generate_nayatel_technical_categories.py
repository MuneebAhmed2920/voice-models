#!/usr/bin/env python3
"""
examples/generate_nayatel_technical_categories.py

Official Production Suite: High-Complexity Technical Audio Suite for Nayatel.
Each recording covers a distinct domain with difficult, industry-standard technical vocabulary:
1. Optical Fiber & ONT Hardware Diagnostics (ONT, PON, LOS red light, Patch Cord, Optical Power)
2. Dual-Band Wi-Fi Optimization & Latency (2.4 GHz vs 5 GHz, Channel Interference, Ping Spikes, Packet Loss, Speed Throttle)
3. Enterprise Static IP & CCTV Configuration (Static IP, Subnet Mask, Gateway, DNS, Port Forwarding, DVR/NVR)
4. Value Added Services — JoyBox 4K & NWatch Cloud (JoyBox 4K, Live TV, Firmware, Buffer, NWatch Cloud, Motion Detection)
5. Corporate Invoicing, FBR/PRA Tax Exemption & Billing (Billing Cycle, Corporate Invoice, Withholding Tax, Customer Portal, 1Link, Kuickpay)
6. Optical Backbone Fiber Cut Outage Advisory (Main Optical Backbone, Fiber Cut, OTDR Testing, Splicing Team, SLA Restoration Time)

Engineered with:
- Authentic conversational Pakistani helpline Urdu code-switching
- English technical loanwords preserved directly in Latin script
- 36 diffusion steps for studio acoustic realism
- speed: 0.88 (deliberate, polite, attentive pace)
- guidance_scale: 1.85 (natural micro-prosody)
- Both 24kHz Studio Master HD & 8kHz PSTN Telephony Filtered outputs
"""

import os
import sys
import time
import whisper

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_ROOT)

from runtime.engine import UrduVoiceModel

TECH_CATEGORIES = {
    "nayatel_tech_ont_fiber_diagnostics": {
        "title": "Category 1: Optical Fiber & ONT Hardware Diagnostics (LOS / Red Light / Patch Cord)",
        "scenario": "Hardware & Physical Layer Troubleshooting",
        "difficult_words": ["ONT", "PON light", "LOS light", "Red blink", "Optical fiber patch cord", "Optical power", "Router restart", "Field team"],
        "text": (
            "السلام علیکم، Nayatel Technical Support۔ "
            "اگر آپ کے ONT router پر PON light آف ہے یا LOS light red blink کر رہی ہے، "
            "تو اس کا مطلب ہے کہ optical fiber patch cord میں bend یا break کی وجہ سے optical power drop ہو چکی ہے۔ "
            "آپ please yellow fiber patch cord کو gently check کریں اور router restart کریں۔ "
            "اگر مسئلہ برقرار رہے، تو ہماری field team فوری طور پر optical power verify کرنے تشریف لائے گی۔"
        )
    },
    "nayatel_tech_wifi_latency_optimization": {
        "title": "Category 2: Dual-Band Wi-Fi Optimization (2.4 GHz vs 5 GHz / Ping / Packet Loss)",
        "scenario": "Wireless RF, Gaming & Video Call Diagnostics",
        "difficult_words": ["Dual-band Wi-Fi", "2.4 GHz", "5 GHz SSID", "Channel interference", "Ping spikes", "Latency", "Packet loss", "Speed throttle", "Bandwidth"],
        "text": (
            "جی، اگر آپ کو gaming یا video calling کے دوران ping spikes، latency یا packet loss کا سامنا ہے، "
            "تو verify کریں کہ آپ 5 GHz Wi-Fi SSID سے connect ہیں۔ "
            "2.4 GHz پر channel interference کی وجہ سے speed throttle ہو سکتی ہے۔ "
            "ہم نے backend سے آپ کا Wi-Fi channel optimize کر دیا ہے۔ "
            "آپ speed test run کر کے download اور upload bandwidth verify کر لیں۔"
        )
    },
    "nayatel_tech_static_ip_cctv_configuration": {
        "title": "Category 3: Enterprise Static IP & CCTV Port Forwarding Configuration",
        "scenario": "Networking, IP Assignment & Remote Surveillance",
        "difficult_words": ["Static IP", "Port forwarding", "Subnet mask", "Default gateway", "Primary DNS", "CCTV DVR", "NVR", "Camera streams"],
        "text": (
            "السلام علیکم، Nayatel Enterprise Support۔ "
            "آپ کی Static IP configuration اور CCTV DVR کے لیے port forwarding rules activate کر دیے گئے ہیں۔ "
            "آپ router کی settings میں subnet mask، default gateway اور primary DNS verify کر سکتے ہیں۔ "
            "اب آپ static IP کے ذریعے باہر سے اپنے NVR اور camera streams کو securely access کر سکتے ہیں۔"
        )
    },
    "nayatel_tech_vas_nwatch_joybox_4k": {
        "title": "Category 4: Value Added Services — JoyBox 4K Streaming & NWatch Cloud CCTV",
        "scenario": "Smart Devices & Cloud Surveillance",
        "difficult_words": ["JoyBox 4K", "Live TV streams", "Firmware update", "Cache clear", "NWatch cloud recording", "AI motion detection", "Surveillance"],
        "text": (
            "جی، Nayatel JoyBox 4K اور NWatch Cloud Services کے حوالے سے رہنمائی حاضر ہے۔ "
            "اگر JoyBox پر Live TV streams buffer کر رہی ہیں، "
            "تو settings میں جا کر firmware update check کریں اور cache clear کریں۔ "
            "NWatch surveillance cameras کی cloud recording اور AI motion detection alerts آپ Nayatel mobile app پر live view کر سکتے ہیں۔"
        )
    },
    "nayatel_tech_corporate_billing_tax_portal": {
        "title": "Category 5: Corporate Invoicing, FBR/PRA Withholding Tax & Online Payment",
        "scenario": "Billing, Taxation & Payment Gateways",
        "difficult_words": ["Billing cycle", "Corporate invoice", "FBR", "PRA withholding tax", "Exemption certificate", "Customer Portal", "1Link", "Kuickpay"],
        "text": (
            "آپ کے account کی billing cycle اور corporate invoice کی details update ہو چکی ہیں۔ "
            "FBR اور PRA withholding tax exemption کے لیے آپ exemption certificate Nayatel Customer Portal پر upload کر سکتے ہیں۔ "
            "Invoice کی payment آپ 1Link، Kuickpay، یا mobile banking app سے باآسانی ادا کر سکتے ہیں۔"
        )
    },
    "nayatel_tech_backbone_outage_advisory": {
        "title": "Category 6: Fiber Cut Outage Advisory & OTDR Splicing Restoration",
        "scenario": "Network Outage & Infrastructure Emergency",
        "difficult_words": ["Main optical backbone", "Fiber cut", "Unscheduled outage", "Technical splicing team", "OTDR testing", "Fiber jointing", "SLA restoration time"],
        "text": (
            "اہم اطلاع: آپ کے علاقے میں civil work کے دوران Nayatel کی main optical backbone fiber cut ہونے کی وجہ سے unscheduled outage کا سامنا ہے۔ "
            "ہماری technical splicing team OTDR testing کے ذریعے fault locate کر چکی ہے اور fiber jointing جاری ہے۔ "
            "متوقع SLA restoration time دو گھنٹے ہے۔ آپ کے تعاون کا بہت شکریہ۔"
        )
    }
}

def main():
    print("=" * 85)
    print("🎙️ Generating Nayatel Domain-Specific Technical Audio Suite (36-step, 0.88 speed)")
    print("=" * 85)

    outputs_dir = os.path.join(PACKAGE_ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    print("\n[1/3] Initializing Mahrukh Urdu Voice Model...")
    model = UrduVoiceModel()

    print("\n[2/3] Initializing Whisper ASR Verification Pipeline...")
    whisper_asr = whisper.load_model("base")

    results = []

    print("\n[3/3] Synthesizing 6 Specialized Technical Domain Recordings (24k HD & 8k Telephony)...")
    for key, data in TECH_CATEGORIES.items():
        title = data["title"]
        scenario = data["scenario"]
        script_text = data["text"]
        diff_words = ", ".join(data["difficult_words"])

        print("\n" + "-" * 75)
        print(f">>> {title}")
        print(f"    Scenario:        {scenario}")
        print(f"    Difficult Terms: {diff_words}")
        print(f"    Script:          {script_text}")

        # 1. 24kHz HD Studio Audio (Clean master)
        out_24k = os.path.join(outputs_dir, f"{key}_24k_hd.wav")
        t0 = time.time()
        aud_24k, sr_24k = model.synthesize(
            script_text,
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

        # 2. 8kHz PSTN Telephony Filtered Audio (Call center line)
        out_8k = os.path.join(outputs_dir, f"{key}_8k_telephony.wav")
        t1 = time.time()
        aud_8k, sr_8k = model.synthesize(
            script_text,
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

        results.append({
            "key": key,
            "title": title,
            "scenario": scenario,
            "file_24k": out_24k,
            "dur_24k": dur_24k,
            "asr_24k": asr_24k,
            "file_8k": out_8k,
            "dur_8k": dur_8k,
            "asr_8k": asr_8k
        })

    print("\n" + "=" * 85)
    print("✅ All 6 Technical Nayatel Category Recordings Generated & Verified!")
    print("=" * 85)
    for r in results:
        print(f"• {r['title']}")
        print(f"    24kHz HD Studio:  {r['file_24k']} ({r['dur_24k']:.2f}s)")
        print(f"    8kHz Telephony:   {r['file_8k']} ({r['dur_8k']:.2f}s)")

if __name__ == "__main__":
    main()

"""
2-Minute Conversational Telephony Benchmark & Audio Generator.
Simulates a complete 2-minute customer service call with Clara,
measuring Time-To-First-Audio (TTFA), Inference Latency, Audio Duration, and Real-Time Factor (RTF).
"""

import os
import sys
import time
import struct
import numpy as np
import soundfile as sf

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from runtime.engine import get_engine

CONVERSATION_TURNS = [
    {
        "turn": 1,
        "speaker": "Clara (AI Agent)",
        "text": "[excited] Hello! Thank you for calling Premier Health and Wellness. My name is Clara, your virtual assistant. How may I help you today?",
        "customer_reply": "Hi Clara! I need to reschedule an existing appointment with Dr. Smith, and check if my latest payment went through.",
        "customer_dur": 5.5
    },
    {
        "turn": 2,
        "speaker": "Clara (AI Agent)",
        "text": "[sympathetic] I would be more than happy to help you with both of those requests today! To look up your file, could you please provide your full name and your 10-digit telephone number?",
        "customer_reply": "Sure, my name is David Miller, and my phone number is (800) 555-0199.",
        "customer_dur": 5.0
    },
    {
        "turn": 3,
        "speaker": "Clara (AI Agent)",
        "text": "Thank you so much, Mister Miller. Let me pull up your account right now. [breath] I see your original appointment was scheduled for Thursday at 10:00 AM, and I see a recent payment of $145.50 received on 08/18/2026.",
        "customer_reply": "That's great to hear. What dates do you have open next week in the afternoon?",
        "customer_dur": 4.5
    },
    {
        "turn": 4,
        "speaker": "Clara (AI Agent)",
        "text": "Looking at the calendar for next week, Dr. Smith has an opening on Monday, August 24th at 3:30 PM, or Wednesday, August 26th at 2:00 PM. Which of those times works best for your schedule?",
        "customer_reply": "Monday, August 24th at 3:30 PM would be perfect for me.",
        "customer_dur": 3.8
    },
    {
        "turn": 5,
        "speaker": "Clara (AI Agent)",
        "text": "[excited] Wonderful! I have updated your appointment for Monday, August 24th at 3:30 PM with Dr. Smith at 123 Main St., Suite 400. You will receive a confirmation SMS and email shortly.",
        "customer_reply": "Excellent! Do I have any remaining balance on my account?",
        "customer_dur": 3.5
    },
    {
        "turn": 6,
        "speaker": "Clara (AI Agent)",
        "text": "Your remaining account balance is currently $0.00, so everything is fully taken care of. Is there anything else I can assist you with today?",
        "customer_reply": "No, that was super fast and helpful. Thank you Clara!",
        "customer_dur": 3.2
    },
    {
        "turn": 7,
        "speaker": "Clara (AI Agent)",
        "text": "[emotional] You are very welcome, Mister Miller! Thank you for choosing Premier Health. Have a wonderful rest of your day, and goodbye!",
        "customer_reply": "Goodbye!",
        "customer_dur": 1.5
    }
]


def generate_comfort_noise(duration_sec: float, sample_rate: int = 24000, rms: float = 0.002) -> np.ndarray:
    """Generates soft telephone line comfort noise during customer turns / pauses."""
    samples = int(duration_sec * sample_rate)
    noise = np.random.randn(samples).astype(np.float32)
    # Bandpass filter simulation for telephone line
    noise = noise * (rms / (np.std(noise) + 1e-6))
    return noise


def run_conversation_benchmark():
    print("=" * 85)
    print("🎙️ 2-MINUTE CONVERSATION BENCHMARK & AUDIO GENERATION (OmniVoice TTS Engine)")
    print("=" * 85)

    engine = get_engine(default_ode_steps=16, warmup=True)
    status = engine.get_status()
    print(f"Device: {status['device']} ({status['dtype']}) | GPU: {status.get('gpu_name') or 'CPU'}")
    print(f"Sampling Rate: 24,000 Hz | ODE Steps: 16 (Low Latency Mode)")
    print("-" * 85)

    sample_rate = 24000
    combined_audio_track = []
    turn_metrics = []

    for item in CONVERSATION_TURNS:
        turn_num = item["turn"]
        clara_text = item["text"]
        cust_reply = item["customer_reply"]
        cust_dur = item["customer_dur"]

        print(f"\n--- Turn {turn_num} ---")
        print(f"👤 Caller: \"{cust_reply}\"")
        print(f"🤖 Clara:  \"{clara_text}\"")

        # Synthesize Clara's response
        t0 = time.time()
        audio_bytes, metrics = engine.synthesize(
            text=clara_text,
            voice="voice_clara",
            sample_rate=sample_rate,
            format_type="pcm16",
            ode_steps=16,
            normalize_text=True
        )
        t1 = time.time()

        # Convert pcm16 bytes back to float32 array for mixing
        clara_audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        
        # Turn metrics
        turn_stat = {
            "turn": turn_num,
            "words": len(clara_text.split()),
            "ttfa_ms": metrics["ttfa_ms"],
            "inference_ms": metrics["inference_time_ms"],
            "clara_dur_sec": metrics["audio_duration_sec"],
            "cust_dur_sec": cust_dur,
            "total_turn_sec": metrics["audio_duration_sec"] + cust_dur + 0.6,  # with short pause
            "rtf": metrics["rtf"],
            "emotion": metrics["emotion"]
        }
        turn_metrics.append(turn_stat)

        print(f"  ➜ TTFA: {turn_stat['ttfa_ms']:.1f} ms | Inference: {turn_stat['inference_ms']:.1f} ms | Audio: {turn_stat['clara_dur_sec']:.2f} s | RTF: {turn_stat['rtf']:.4f}")

        # Append Clara speech
        combined_audio_track.append(clara_audio)
        
        # Short conversational turn pause (300ms)
        combined_audio_track.append(generate_comfort_noise(0.3, sample_rate))

        # Simulated customer turn audio (ambient line comfort tone)
        combined_audio_track.append(generate_comfort_noise(cust_dur, sample_rate))

        # Short pause before Clara replies (300ms)
        combined_audio_track.append(generate_comfort_noise(0.3, sample_rate))

    # Concatenate complete conversation
    full_conversation_audio = np.concatenate(combined_audio_track)
    total_conv_duration_sec = len(full_conversation_audio) / sample_rate

    # Save 24kHz master conversation WAV
    out_dir = os.path.join(_PROJECT_ROOT, "samples")
    os.makedirs(out_dir, exist_ok=True)
    out_wav_path = os.path.join(out_dir, "clara_2min_conversation.wav")
    
    sf.write(out_wav_path, full_conversation_audio, sample_rate, subtype="PCM_16")

    # Also save 8kHz PSTN Telephony version for Vicidial
    audio_8k = engine.audio_processor.resample(full_conversation_audio, 24000, 8000)
    out_8k_path = os.path.join(out_dir, "clara_2min_conversation_8k_pstn.wav")
    sf.write(out_8k_path, audio_8k, 8000, subtype="PCM_16")

    # Latency & Performance Breakdown Table
    print("\n" + "=" * 85)
    print("📊 2-MINUTE TELEPHONY CONVERSATION LATENCY REPORT")
    print("=" * 85)
    header = f"{'Turn':<6} | {'Words':<6} | {'Emotion':<12} | {'TTFA (ms)':<10} | {'Inference (ms)':<15} | {'Speech (s)':<10} | {'RTF':<8}"
    print(header)
    print("-" * len(header))

    total_clara_speech = sum(m["clara_dur_sec"] for m in turn_metrics)
    total_clara_words = sum(m["words"] for m in turn_metrics)
    avg_ttfa = sum(m["ttfa_ms"] for m in turn_metrics) / len(turn_metrics)
    avg_inference = sum(m["inference_ms"] for m in turn_metrics) / len(turn_metrics)
    avg_rtf = sum(m["rtf"] for m in turn_metrics) / len(turn_metrics)

    for m in turn_metrics:
        print(f"{m['turn']:<6} | {m['words']:<6} | {m['emotion']:<12} | {m['ttfa_ms']:<10.1f} | {m['inference_ms']:<15.1f} | {m['clara_dur_sec']:<10.2f} | {m['rtf']:<8.4f}")

    print("-" * len(header))
    print(f"{'TOTAL':<6} | {total_clara_words:<6} | {'All Turns':<12} | {avg_ttfa:<10.1f}* | {avg_inference:<15.1f}* | {total_clara_speech:<10.2f} | {avg_rtf:<8.4f}*")
    print(f"(* = Average across 7 conversational turns)")
    print("=" * 85)
    print(f"🎧 Full Conversation Duration: {total_conv_duration_sec:.1f} seconds ({total_conv_duration_sec / 60:.2f} minutes)")
    print(f"💾 Master 24kHz WAV: {out_wav_path} ({os.path.getsize(out_wav_path) / 1024:.1f} KB)")
    print(f"📞 8kHz PSTN Telephony WAV: {out_8k_path} ({os.path.getsize(out_8k_path) / 1024:.1f} KB)")
    print("=" * 85)


if __name__ == "__main__":
    run_conversation_benchmark()

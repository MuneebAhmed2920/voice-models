# 📞 Vicidial & Asterisk Telephony Integration Guide

This guide details how to integrate the **OmniVoice Real-Time TTS Service** with **Vicidial Call Centers** and **Asterisk PBX** for outbound predictive dialing campaigns, inbound customer service IVRs, and AI Voice Bot agents.

---

## 🏗️ 1. Architecture Overview

```
                                  ┌───────────────────────────────┐
                                  │   PSTN / SIP Trunk Provider   │
                                  └───────────────┬───────────────┘
                                                  │ SIP / G.711u RTP Audio
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │        Vicidial Server        │
                                  │   (Asterisk PBX + MeetMe)     │
                                  └───────┬───────────────▲───────┘
                     AudioSocket / TCP    │               │  8kHz PCM16 / μ-law
                     (Inbound caller mic) │               │  (Outbound TTS speech)
                                          ▼               │
                        ┌─────────────────────────────────┴─────────────┐
                        │              AI Voice Bot Platform            │
                        │    (Deepgram / Whisper STT + LLM Reasoning)  │
                        └─────────────────┬─────────────────────────────┘
                                          │ Streaming Response Text Tokens
                                          ▼
                        ┌───────────────────────────────────────────────┐
                        │   OmniVoice TTS Engine (Windows CUDA FP16)    │
                        │      • Flow-matching ODE in < 150ms           │
                        │      • Real-time 8kHz μ-law / PCM16 Framing   │
                        │      • Instant Caller Barge-In Cancellation   │
                        └───────────────────────────────────────────────┘
```

---

## ⚙️ 2. Asterisk PBX Configuration

### A. Load AudioSocket Module
Ensure `app_audiosocket.so` or `res_audiosocket.so` is enabled in `/etc/asterisk/modules.conf`:
```ini
load => res_audiosocket.so
load => app_audiosocket.so
```

### B. Configure Asterisk Dialplan (`/etc/asterisk/extensions.conf`)
Add the following context for Vicidial AI Voice Bot routing:

```ini
; ====================================================================
; OmniVoice Real-Time AI Voice Bot Dialplan for Vicidial
; ====================================================================

[vicidial-ai-bot]
; Extension 8300: Connect caller audio directly to AI Voice Bot & OmniVoice AudioSocket
exten => 8300,1,NoOp(Vicidial Call Connected to AI Voice Bot on Channel ${CHANNEL})
 same => n,Answer()
 same => n,Set(CHANNEL(audioreadformat)=slin)
 same => n,Set(CHANNEL(audiowriteformat)=slin)
 same => n,Set(CALL_UUID=${UUID})
 ; Connect bidirectional 8kHz Linear PCM audio stream to OmniVoice AudioSocket server (Port 9092)
 same => n,AudioSocket(${CALL_UUID},10.0.0.50:9092)
 same => n,Hangup()

; Extension 8301: Fast MeetMe Conference Bridge for Vicidial Blind Transfer / 3-Way Conference
exten => 8301,1,NoOp(Connecting Vicidial Lead to AI Agent Conference: ${EXTEN})
 same => n,Answer()
 same => n,MeetMe(${EXTEN:4},qM)
 same => n,Hangup()
```

---

## 📋 3. Vicidial Administration Configuration

### Step 1: Create a Remote Agent (AI Bot Agent)
1. Navigate to **Vicidial Admin ➔ Remote Agents ➔ Add A New Remote Agent**.
2. Configure settings:
   - **User ID**: `9901`
   - **Campaign ID**: Select your target predictive/transfer campaign.
   - **Number of Lines**: Set to concurrency limit (e.g., `20` lines per GPU).
   - **Server IP**: Select your Vicidial Telephony Server.
   - **External Extension**: `8300` (points to the dialplan extension above).
   - **Status**: `ACTIVE`.

### Step 2: Configure Campaign Routing
1. In **Campaigns ➔ Detail View**:
   - **Dial Method**: `RATIO` or `ADAPT_HARD`
   - **Auto Dial Level**: `1.5` to `3.0` (depending on agent capacity)
   - **Next Agent Call**: `longest_wait_time`
   - **Drop Action**: `AUDIO` or `TRANSFER`
   - **Safe Harbor Audio**: Set to silence or brand intro.

### Step 3: Configure Inbound Ingroup (Optional for Inbound Calls)
1. In **Inbound ➔ In-Groups ➔ Add A New In-Group**:
   - **Group ID**: `CLARA_AI`
   - **Group Name**: `Clara Customer Support AI`
   - **Next Agent Call**: `longest_wait_time`
   - **Script**: Set to AI Script if monitoring via agent screen.

---

## ⚡ 4. Low-Latency Telephony & Audio Optimization

### 1. Codec Matching & Transcoding Elimination
Always configure SIP trunks and Vicidial channels to native **G.711 μ-law (`ulaw`)** or **8kHz Signed Linear PCM (`slin`)**:
```ini
; In /etc/asterisk/sip.conf or pjsip.conf
disallow=all
allow=ulaw
allow=alaw
```

### 2. Asterisk Jitter Buffer Configuration
To prevent audio latency buildup during live telephony streaming, configure low-latency adaptive jitter buffers:
```ini
; In /etc/asterisk/sip.conf
jbenable=yes
jbforce=yes
jbmaxsize=60
jbimpl=adaptive
jbtargetextra=20
```

### 3. Barge-In (Interruption) Handling
When a caller interrupts the AI bot:
1. The Voice Bot STT detects caller speech (VAD voice onset).
2. The Voice Bot sends `{"event": "interrupt"}` over the WebSocket `/ws/tts` to OmniVoice.
3. OmniVoice immediately drops all in-flight audio frames and clears the queue, stopping playback in `< 20ms`.

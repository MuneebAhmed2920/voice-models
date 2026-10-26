"""FastAPI REST and Streaming Server for Arabic OmniVoice TTS."""

import io
import os
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
import soundfile as sf

from ..arabic_nlp.diacritizer import ArabicDiacritizer
from ..arabic_nlp.normalizer import ArabicNormalizer
from ..models.omnivoice_engine import ArabicOmniVoiceEngine
from ..models.voice_manager import BUILTIN_VOICES, VoiceProfile
from ..pipeline.normalizer_unified import UnifiedNormalizer

app = FastAPI(
    title="Arabic OmniVoice TTS API",
    description="High-fidelity text-to-speech API for Modern Standard Arabic (الفصحى) & Egyptian Arabic (اللهجة المصرية) with zero-shot voice cloning.",
    version="1.0.0"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine lazily
engine = ArabicOmniVoiceEngine()
normalizer = ArabicNormalizer()
diacritizer = ArabicDiacritizer()
unified_normalizer_msa = UnifiedNormalizer(dialect="msa")
unified_normalizer_egy = UnifiedNormalizer(dialect="egyptian")


class SynthesisRequest(BaseModel):
    text: str = Field(..., description="Arabic text to synthesize")
    voice_id: Optional[str] = Field("egy_mohamed", description="ID of preset voice")
    dialect: Optional[str] = Field(None, description="'msa', 'egyptian', 'gulf', etc.")
    speed: Optional[float] = Field(1.0, ge=0.5, le=2.0, description="Speech rate multiplier")
    num_step: Optional[int] = Field(24, ge=8, le=64, description="Diffusion steps")
    guidance_scale: Optional[float] = Field(2.5, description="Guidance strength")
    instruct: Optional[str] = Field(None, description="Optional voice design prompt")
    auto_preprocess: Optional[bool] = Field(True, description="Enable automatic Tashkeel & Normalization")


class PipelineSynthesisRequest(BaseModel):
    text: str = Field(..., description="Arabic/English/mixed text to synthesize")
    voice_id: Optional[str] = Field(None, description="ID of preset voice")
    dialect: Optional[str] = Field(None, description="'msa', 'egyptian', 'gulf', 'levantine'")
    emotion_override: Optional[str] = Field(None, description="'neutral', 'excited', 'sympathetic', 'emotional'")
    telephony_sr: Optional[int] = Field(None, description="Target telephony sample rate: 8000 (G.711) or 16000 (G.722). None = 24kHz.")
    simulate_codec: Optional[bool] = Field(False, description="Simulate G.711 μ-law codec for QA testing")
    num_step: Optional[int] = Field(None, ge=8, le=64, description="Diffusion steps (overrides intent-based default)")


class ConversationalSynthesisRequest(BaseModel):
    text: str = Field(..., description="Arabic conversational text with optional tags [متحمس], [متعاطف], [مؤثر], [تنفس]")
    voice_id: Optional[str] = Field("egy_mohamed", description="ID of preset voice")
    dialect: Optional[str] = Field(None, description="'msa', 'egyptian', etc.")
    emotion_override: Optional[str] = Field(None, description="'neutral', 'excited', 'sympathetic', 'emotional'")
    include_breath: Optional[bool] = Field(True, description="Inject physiological airway breath")
    split_clauses: Optional[bool] = Field(True, description="Split syntactic Arabic clauses with micro-breaths")
    num_step: Optional[int] = Field(20, ge=8, le=64, description="Diffusion steps")


class NormalizationRequest(BaseModel):
    text: str = Field(..., description="Raw Arabic text")
    dialect: Optional[str] = Field("msa", description="'msa' or 'egyptian'")
    add_tashkeel: Optional[bool] = Field(True, description="Apply automatic diacritics")


@app.get("/health")
def health_check():
    """Returns engine health and active device."""
    return {
        "status": "healthy",
        "device": engine.device,
        "sample_rate": 24000,
        "available_voices": len(engine.voice_manager.list_voices())
    }


@app.get("/speakers", response_model=List[dict])
def get_speakers(dialect: Optional[str] = None, gender: Optional[str] = None):
    """Returns list of all available Arabic voices."""
    voices = engine.voice_manager.list_voices(dialect=dialect, gender=gender)
    return [v.__dict__ for v in voices]


@app.post("/normalize")
def normalize_text(req: NormalizationRequest):
    """Normalizes numerals, dates, currencies and adds Tashkeel."""
    norm = normalizer if req.dialect == "msa" else ArabicNormalizer(dialect="egyptian")
    normalized = norm.normalize(req.text)
    if req.add_tashkeel:
        vocalized = diacritizer.diacritize(normalized)
    else:
        vocalized = normalized

    return {
        "original": req.text,
        "normalized": normalized,
        "vocalized": vocalized,
        "dialect": req.dialect
    }


@app.post("/synthesize")
def synthesize(req: SynthesisRequest):
    """Synthesizes Arabic text and streams back WAV audio."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    audio, sr = engine.synthesize(
        text=req.text,
        voice_id=req.voice_id,
        dialect=req.dialect,
        instruct=req.instruct,
        speed=req.speed or 1.0,
        num_step=req.num_step or 24,
        guidance_scale=req.guidance_scale or 2.5,
        auto_preprocess=req.auto_preprocess if req.auto_preprocess is not None else True
    )

    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="arabic_speech.wav"'}
    )


@app.post("/synthesize_conversational")
def synthesize_conversational(req: ConversationalSynthesisRequest):
    """Synthesizes humanized conversational Arabic speech with respiratory dynamics, clause micro-breaths, and emotion coloration."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    audio, sr = engine.synthesize_conversational(
        text=req.text,
        voice_id=req.voice_id,
        dialect=req.dialect,
        emotion_override=req.emotion_override,
        include_breath=req.include_breath if req.include_breath is not None else True,
        split_clauses=req.split_clauses if req.split_clauses is not None else True,
        num_step=req.num_step or 20
    )

    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="arabic_conversational.wav"'}
    )


@app.post("/clone")
async def clone_and_synthesize(
    text: str = Form(..., description="Target Arabic text to synthesize"),
    audio_file: UploadFile = File(..., description="Reference audio file (3-15 seconds)"),
    ref_transcript: Optional[str] = Form(None, description="Transcript of the reference audio")
):
    """Zero-shot voice cloning from uploaded audio clip."""
    suffix = os.path.splitext(audio_file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_ref:
        content = await audio_file.read()
        tmp_ref.write(content)
        tmp_ref_path = tmp_ref.name

    try:
        audio, sr = engine.synthesize(
            text=text,
            ref_audio=tmp_ref_path,
            ref_text=ref_transcript
        )
    finally:
        if os.path.exists(tmp_ref_path):
            os.remove(tmp_ref_path)

    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="cloned_speech.wav"'}
    )


@app.post("/synthesize_pipeline")
def synthesize_pipeline(req: PipelineSynthesisRequest):
    """
    Full 12-stage humanized telephony TTS pipeline.

    Stages: text cleanup → language detection → unified normalization →
    diacritization → context/prosody analysis → breathing gate →
    TTS inference → EQ coloration → loudness normalization →
    DC removal → telephony resampling → quality verification.

    Use telephony_sr=8000 for G.711 narrowband or telephony_sr=16000 for G.722 wideband.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    audio, sr = engine.synthesize_pipeline(
        text=req.text,
        voice_id=req.voice_id,
        dialect=req.dialect,
        emotion_override=req.emotion_override,
        telephony_sr=req.telephony_sr,
        simulate_codec=req.simulate_codec if req.simulate_codec is not None else False,
        num_step=req.num_step,
    )

    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="pipeline_speech.wav"',
            "X-Sample-Rate": str(sr),
        }
    )


@app.post("/normalize_unified")
def normalize_unified(req: dict):
    """
    Full telephony-grade normalization preview (no TTS).
    Returns the normalized text that would be sent to TTS.
    Accepts: {text, dialect, add_tashkeel}
    """
    text = req.get("text", "").strip()
    dialect = req.get("dialect", "msa")
    add_tashkeel = req.get("add_tashkeel", True)

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    norm = unified_normalizer_egy if dialect == "egyptian" else unified_normalizer_msa
    normalized = norm.normalize(text)

    if add_tashkeel and dialect in ("msa", "standard"):
        vocalized = diacritizer.diacritize(normalized)
    else:
        vocalized = normalized

    return {
        "original": text,
        "normalized": normalized,
        "vocalized": vocalized,
        "dialect": dialect,
    }

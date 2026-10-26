"""Arabic OmniVoice Interactive Web Studio.

A modern, Arabic-first web studio for:
- Text-to-Speech in Modern Standard Arabic (الفصحى) & Egyptian (المصرية)
- Live Tashkeel (Diacritics) editor & Normalization inspector
- Zero-Shot Voice Cloning from microphone or audio file
- Voice Design via natural language descriptions
- Built-in Studio Voices (12+ male & female speakers)
- Low-latency synthesis & export
"""

import os
import tempfile
import time
from typing import Optional, Tuple

import gradio as gr
import numpy as np

from ..arabic_nlp.diacritizer import ArabicDiacritizer
from ..arabic_nlp.normalizer import ArabicNormalizer
from ..models.omnivoice_engine import ArabicOmniVoiceEngine
from ..models.voice_manager import BUILTIN_VOICES, VoiceManager

# Custom CSS for modern Arabic aesthetic
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Outfit:wght@400;500;600;700&display=swap');

* {
    font-family: 'Cairo', 'Outfit', system-ui, -apple-system, sans-serif !important;
}

body, .gradio-container {
    direction: rtl;
    text-align: right;
    background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0d1527 100%) !important;
    color: #f3f4f6 !important;
}

.title-container {
    text-align: center;
    padding: 2.2rem 1.5rem 1.2rem;
    background: radial-gradient(circle at center, rgba(37, 99, 235, 0.15) 0%, transparent 70%);
    border-radius: 18px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.title-container h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}

.title-container p {
    color: #9ca3af;
    font-size: 1.1rem;
    max-width: 700px;
    margin: 0 auto;
}

.badge-tag {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0 0.25rem;
    background: rgba(59, 130, 246, 0.2);
    color: #93c5fd;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.gr-button-primary {
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39) !important;
    transition: all 0.2s ease-in-out !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px 0 rgba(37, 99, 235, 0.5) !important;
}

.gr-panel {
    background: rgba(17, 24, 39, 0.7) !important;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
}

.tag-btn {
    font-size: 0.85rem !important;
    padding: 0.2rem 0.6rem !important;
}
"""

SAMPLE_PROMPTS = {
    "أخبار بالفصحى": "سَيِّدَاتِي وَآنِسَاتِي، نُحَيِّيكُمْ فِي هَذَا المَوْجَزِ الإِخْبَارِيِّ الصَّبَاحِيِّ مِنِ اسْتُودْيُو أَوْمْنِي فَوْيْس.",
    "محادثة مصرية": "ازيك عامل ايه النهاردة؟ عندي meeting مهم الساعة 3:30 في الشركة، ومحتاج أخلص التقارير بسرعة.",
    "سرد وثائقي": "فِي أَعْمَاقِ البِحَارِ، تَعِيشُ كَائِنَاتٌ مُدْهِشَةٌ تَشِعُّ ضَوْءاً طَبِيعِيّاً فِي ظَلَامٍ دَامِس.",
    "أرقام وعملات": "بلغت قيمة المبيعات 1500 دولار، مع زيادة بنسبة 25% مقارنة بالعام الماضي 2025.",
    "تعبير ونبرة": "مرحباً بكم جميعاً! [laughter] لا أستطيع أن أصف مدى سعادتي بلقائكم اليوم [sigh] أخيراً تحقق الحلم."
}


class WebStudioApp:
    """Encapsulates the Arabic OmniVoice Web Studio."""

    def __init__(self, engine: Optional[ArabicOmniVoiceEngine] = None):
        self.engine = engine or ArabicOmniVoiceEngine()
        self.normalizer = ArabicNormalizer()
        self.diacritizer = ArabicDiacritizer()
        self.voice_manager = self.engine.voice_manager

    def build_ui(self) -> gr.Blocks:
        voice_choices = [
            (f"{v.name_ar} ({v.dialect.upper()} - {v.gender})", v.id)
            for v in BUILTIN_VOICES
        ]

        with gr.Blocks(title="OmniVoice Arabic — استوديو الصوت العربي") as demo:
            # Header
            gr.HTML("""
            <div class="title-container">
                <h1>🎙️ استوديو أومني فويس العربي</h1>
                <p>نظام توليد الصوت العربي المتطور بالذكاء الاصطناعي — فصحى ومصري مع استنساخ الصوت والتشكيل الذكي</p>
                <div style="margin-top: 0.8rem;">
                    <span class="badge-tag">⚡ انتشار غير تعاقبي (Diffusion LM)</span>
                    <span class="badge-tag">🗣️ الفصحى والمصرية وخليجي وشامي</span>
                    <span class="badge-tag">🧬 استنساخ فوري بدون تدريب</span>
                    <span class="badge-tag">✨ تشكيل تلقائي (Harakat)</span>
                </div>
            </div>
            """)

            with gr.Tabs() as tabs:
                # Tab 1: Studio Synthesis
                with gr.TabItem("🎙️ التوليد والأصوات الجاهزة (Studio TTS)", id="tab_studio"):
                    with gr.Row():
                        with gr.Column(scale=5):
                            text_input = gr.Textbox(
                                label="النص العربي المراد نطقه (Arabic Text)",
                                placeholder="اكتب النص هنا أو اختر من الأمثلة أدناه...",
                                lines=5,
                                value=SAMPLE_PROMPTS["أخبار بالفصحى"],
                                elem_id="main_text_input"
                            )

                            # Quick sample buttons
                            with gr.Row():
                                gr.Markdown("**أمثلة سريعة:**")
                                btn_news = gr.Button("📰 أخبار فصحى", size="sm")
                                btn_egy = gr.Button("🇪🇬 مصري وكود سويتش", size="sm")
                                btn_doc = gr.Button("🌍 وثائقي", size="sm")
                                btn_num = gr.Button("🔢 أرقام وعملات", size="sm")
                                btn_exp = gr.Button("🎭 تعابير صوتية", size="sm")

                            with gr.Accordion("✨ معالجة النص والتشكيل (Tashkeel & Normalization)", open=False):
                                with gr.Row():
                                    btn_tashkeel = gr.Button("تشكيل النص تلقائياً (Add Harakat)", size="sm")
                                    btn_norm = gr.Button("توسيع الأرقام والتواريخ (Normalize)", size="sm")
                                preview_normalized = gr.Textbox(
                                    label="النص بعد المعالجة التلقائية (Normalized & Vocalized)",
                                    lines=3,
                                    interactive=False
                                )

                        with gr.Column(scale=4):
                            voice_dropdown = gr.Dropdown(
                                choices=voice_choices,
                                value=voice_choices[0][1],
                                label="اختر الصوت (Select Voice)",
                                elem_id="voice_select"
                            )

                            with gr.Row():
                                dialect_radio = gr.Radio(
                                    choices=[("الفصحى (MSA)", "msa"), ("المصرية (Egyptian)", "egyptian")],
                                    value="msa",
                                    label="اللهجة المعتمدة (Dialect Mode)"
                                )

                            with gr.Row():
                                speed_slider = gr.Slider(
                                    minimum=0.7, maximum=1.4, value=1.0, step=0.05,
                                    label="سرعة النطق (Speed)"
                                )
                                steps_slider = gr.Slider(
                                    minimum=12, maximum=48, value=24, step=4,
                                    label="خطوات الانتشار (Steps: 16 سريع، 32 فائق)"
                                )

                            synthesize_btn = gr.Button("🚀 توليد الصوت الآن (Generate Speech)", variant="primary", size="lg")

                    with gr.Row():
                        audio_output = gr.Audio(
                            label="الصوت المُولّد (Generated Audio)",
                            type="filepath",
                            interactive=False,
                            elem_id="main_audio_player"
                        )

                # Tab 2: Conversational Prosody & Organic Breathing
                with gr.TabItem("🌬️ المحادثة والتنفس الطبيعي (Conversational AI)", id="tab_conversational"):
                    gr.Markdown("""
                    ### 🌬️ هندسة المحادثة البشرية، التنفس الفسيولوجي والتلوين الصوتي (Conversational Prosody & Breathing)
                    تحويل روبوتات خدمة العملاء والرد الآلي (Telephony / IVR) إلى أصوات بشرية طبيعية تحاكي **الشهيق العضوي (180–5500 هرتز)** والسكتات الكلامية قبل أدوات الربط، مع تلوين ترددي يطابق الحالة الشعورية للمتحدث.
                    """)
                    with gr.Row():
                        with gr.Column(scale=5):
                            conv_text_input = gr.Textbox(
                                label="النص الحواري (مع وسوم المشاعر والتنفس الاختيارية)",
                                placeholder="اكتب النص الحواري هنا أو اختر من سيناريوهات الرد الآلي...",
                                lines=5,
                                value="[متعاطف] أهلاً بك يا فندم، نعتذر جداً عن التأخير، [تنفس] وبإذن الله هنحل المشكلة لحضرتك في أسرع وقت ممكن.",
                                elem_id="conv_text_input"
                            )

                            # Quick tag inserters
                            with gr.Row():
                                gr.Markdown("**إدراج وسوم النبرة والمشاعر:**")
                                btn_tag_excited = gr.Button("🚀 [متحمس]", size="sm")
                                btn_tag_symp = gr.Button("🤝 [متعاطف]", size="sm")
                                btn_tag_emot = gr.Button("💖 [مؤثر]", size="sm")
                                btn_tag_high = gr.Button("🔊 [نبرة_عالية]", size="sm")
                                btn_tag_low = gr.Button("🔉 [نبرة_منخفضة]", size="sm")
                                btn_tag_whisper = gr.Button("🤫 [همس]", size="sm")

                            with gr.Row():
                                gr.Markdown("**إدراج إيماءات التنفس والتعبير البشري:**")
                                btn_tag_laugh = gr.Button("😂 [ضحك]", size="sm")
                                btn_tag_sigh = gr.Button("😮‍💨 [تنهد]", size="sm")
                                btn_tag_breath = gr.Button("🌬️ [تنفس]", size="sm")
                                btn_tag_pause = gr.Button("⏸️ [وقفة]", size="sm")

                            # Quick call-center scenarios
                            with gr.Row():
                                gr.Markdown("**سيناريوهات محادثة جاهزة:**")
                                btn_scen_all = gr.Button("🌟 استعراض شامل لكل الإيماءات (All Gestures)", size="sm")
                                btn_scen_support = gr.Button("🎧 دعم عملاء واعتذار", size="sm")
                                btn_scen_booking = gr.Button("🎉 تأكيد حجز وبشرى", size="sm")
                                btn_scen_bank = gr.Button("🏛️ رد بنكي وامتنان", size="sm")
                                btn_scen_casual = gr.Button("☕ دردشة وتنهد", size="sm")

                            with gr.Accordion("📊 مصفوفة التنفس والتلوين الصوتي الترددي (EQ Matrix)", open=False):
                                gr.Markdown("""
                                | الحالة الشعورية | المدة (Duration) | الشدة (Gain) | فجوة ما قبل النطق | التلوين الصوتي الترددي (EQ Acoustic Coloration) |
                                | :--- | :---: | :---: | :---: | :--- |
                                | **طبيعي (Neutral)** | 140–160ms | -28 dBFS | 50ms | مسار صوتي متوازن وطبيعي |
                                | **متحمس (Excited)** | 80–110ms | -30 dBFS | 25ms | تعزيز الحضور الصوتي (+2.0 dB @ 4.2 kHz) وسرعة 0.98x |
                                | **متعاطف (Sympathetic)** | 180–220ms | -26 dBFS | 75ms | تدفئة الترددات المنخفضة-المتوسطة (+1.8 dB @ 280 Hz) وسرعة 0.86x |
                                | **مؤثر (Emotional)** | 150–180ms | -27 dBFS | 60ms | رنين صدري عميق (+1.5 dB @ 320 Hz) وسرعة 0.90x |
                                """)

                        with gr.Column(scale=4):
                            conv_voice_dropdown = gr.Dropdown(
                                choices=voice_choices,
                                value=voice_choices[1][1] if len(voice_choices) > 1 else voice_choices[0][1],
                                label="اختر صوت المتحدث (Select Voice)"
                            )
                            with gr.Row():
                                conv_dialect_radio = gr.Radio(
                                    choices=[("الفصحى (MSA)", "msa"), ("المصرية (Egyptian)", "egyptian")],
                                    value="egyptian",
                                    label="لهجة التوليد"
                                )
                            conv_emotion_dropdown = gr.Dropdown(
                                choices=[
                                    ("تلقائي (حسب الوسوم في النص)", "auto"),
                                    ("طبيعي متوازن (Neutral)", "neutral"),
                                    ("متحمس وبهيج (Excited)", "excited"),
                                    ("متعاطف وداعم (Sympathetic)", "sympathetic"),
                                    ("مؤثر وصادق (Emotional)", "emotional")
                                ],
                                value="auto",
                                label="تحديد النبرة الشعورية (Emotion Modulation)"
                            )
                            with gr.Row():
                                conv_breath_check = gr.Checkbox(
                                    value=True,
                                    label="تفعيل شهيق البداية والفسيولوجيا (Turn-Initial Breath)"
                                )
                                conv_clause_check = gr.Checkbox(
                                    value=True,
                                    label="سكتات وتنفس أدوات الربط (Clause Micro-Breaths)"
                                )
                            conv_steps_slider = gr.Slider(
                                minimum=12, maximum=40, value=20, step=4,
                                label="خطوات الانتشار (Diffusion Steps)"
                            )

                            conv_synthesize_btn = gr.Button("🎙️ توليد المحادثة الواقعية (Synthesize Human Speech)", variant="primary", size="lg")

                    with gr.Row():
                        conv_audio_output = gr.Audio(
                            label="الصوت البشري المُولّد مع التنفس والتعابير",
                            type="filepath",
                            interactive=False
                        )

                # Tab 3: Zero-Shot Voice Cloning
                with gr.TabItem("🧬 استنساخ الأصوات (Voice Cloning)", id="tab_clone"):
                    gr.Markdown("""
                    ### 🧬 استنساخ الصوت في ثوانٍ معدودة (Zero-Shot Voice Cloning)
                    ارفع ملفاً صوتياً (أو سجل بصوتك عبر الميكروفون) لمدة 3 إلى 10 ثوانٍ، وسيقوم النموذج بتقليد نبرة الصوت فورياً.
                    """)
                    with gr.Row():
                        with gr.Column(scale=5):
                            ref_audio_input = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="العينة الصوتية المرجعية (Reference Audio 3–10s)"
                            )
                            ref_text_input = gr.Textbox(
                                label="تفريغ العينة المرجعية (اختياري، يرفع دقة الاستنساخ)",
                                placeholder="اكتب الكلمات التي قيلت في العينة المرجعية..."
                            )
                            clone_target_text = gr.Textbox(
                                label="النص العربي المراد قراءته بالصوت المستنسخ",
                                lines=4,
                                placeholder="اكتب هنا النص الذي ترغب في أن ينطقه الصوت المستنسخ..."
                            )
                            clone_btn = gr.Button("🧬 بدء استنساخ الصوت والتوليد", variant="primary", size="lg")

                        with gr.Column(scale=4):
                            cloned_audio_output = gr.Audio(
                                label="الصوت المستنسخ المُولّد",
                                type="filepath",
                                interactive=False
                            )

                # Tab 3: Voice Design (Natural Language Prompts)
                with gr.TabItem("🎨 تصميم الأصوات بالأوامر (Voice Design)", id="tab_design"):
                    gr.Markdown("""
                    ### 🎨 تصميم نبرة الصوت بالوصف النصي
                    حدد مواصفات الصوت المرغوب (العمر، النبرة، السرعة، الطابع) وسيقوم النموذج بابتكار نبرة مطابقة.
                    """)
                    with gr.Row():
                        with gr.Column(scale=5):
                            design_prompt_input = gr.Textbox(
                                label="وصف مواصفات الصوت (Voice Description Prompt)",
                                lines=3,
                                value="صوت رجل وقور في الخمسين من عمره، نبرة إذاعية عميقة ولغة عربية فصيحة متقنة"
                            )
                            design_text_input = gr.Textbox(
                                label="النص المراد نطقه",
                                lines=4,
                                value="نرحب بكم جميعاً في هذا اللقاء الثقافي الاستثنائي."
                            )
                            design_btn = gr.Button("🎨 توليد صوت بالوصف", variant="primary", size="lg")

                        with gr.Column(scale=4):
                            design_audio_output = gr.Audio(
                                label="الصوت المُصمم",
                                type="filepath",
                                interactive=False
                            )

                # Tab 4: Long-Text Streaming
                with gr.TabItem("⚡ توليد النصوص الطويلة (Streaming)", id="tab_stream"):
                    gr.Markdown("""
                    ### ⚡ التوليد السريع للنصوص الطويلة
                    يقوم النظام بتقسيم الفقرات الطويلة آلياً ومعالجتها لتقليل زمن الانتظار حتى أول مقطع صوتي.
                    """)
                    stream_text_input = gr.Textbox(
                        label="نص الفقرة الطويلة (Long Article / Script)",
                        lines=7,
                        value="كان الذكاء الاصطناعي مجرد خيال علمي في منتصف القرن الماضي. واليوم أصبح جزءاً لا يتجزأ من حياتنا اليومية. من الهواتف الذكية إلى الأنظمة الطبية المتطورة، تتسارع الابتكارات لخدمة البشرية وتسهيل الأعمال."
                    )
                    stream_btn = gr.Button("⚡ توليد الفقرة المكتملة", variant="primary")
                    stream_audio_output = gr.Audio(label="الملف الصوتي الكامل", type="filepath")

                # Tab 5: Documentation & API Reference
                with gr.TabItem("📚 دليل الاستخدام والربط البرمجي (API)", id="tab_docs"):
                    gr.Markdown("""
                    ## 🔌 كيفية استخدام النظام في مشاريعك (Python & REST API)

                    ### 1. كود بايثون السريع (Python Quickstart)
                    ```python
                    from src.models.omnivoice_engine import ArabicOmniVoiceEngine

                    engine = ArabicOmniVoiceEngine()

                    # 1. توليد بصوت جاهز (فصحى أو مصري)
                    audio, sr = engine.synthesize(
                        text="مرحباً بكم في نظام أومني فويس العربي",
                        voice_id="msa_kareem",
                        output_path="output.wav"
                    )

                    # 2. استنساخ صوت فوري (Zero-shot Cloning)
                    audio, sr = engine.synthesize(
                        text="النهاردة الجو جميل جداً",
                        ref_audio="speaker_sample.wav",
                        output_path="cloned.wav"
                    )
                    ```

                    ### 2. استدعاء خادم الـ REST API
                    ```bash
                    # تشغيل الخادم
                    uvicorn src.app.api:app --host 0.0.0.0 --port 8000

                    # إرسال طلب توليد
                    curl -X POST http://localhost:8000/synthesize \\
                         -H "Content-Type: application/json" \\
                         -d '{"text": "السلام عليكم ورحمة الله", "voice_id": "msa_kareem"}' \\
                         --output speech.wav
                    ```

                    ### 3. وسوم التعبير الصوتي المدعومة (Expressive Tags)
                    يمكنك إدراج الوسوم التالية في النص:
                    - `[laughter]` : للضحك الطبيعي الخفيف
                    - `[sigh]` : للتنهد
                    - `[surprise-ah]` : لنبرة الدهشة والمفاجأة
                    """)

            # Event Handlers
            btn_news.click(lambda: SAMPLE_PROMPTS["أخبار بالفصحى"], outputs=text_input)
            btn_egy.click(lambda: SAMPLE_PROMPTS["محادثة مصرية"], outputs=text_input)
            btn_doc.click(lambda: SAMPLE_PROMPTS["سرد وثائقي"], outputs=text_input)
            btn_num.click(lambda: SAMPLE_PROMPTS["أرقام وعملات"], outputs=text_input)
            btn_exp.click(lambda: SAMPLE_PROMPTS["تعبير ونبرة"], outputs=text_input)

            def do_tashkeel(text):
                return self.diacritizer.diacritize(text)

            def do_norm(text, dialect):
                norm = self.normalizer if dialect == "msa" else ArabicNormalizer(dialect="egyptian")
                return norm.normalize(text)

            btn_tashkeel.click(do_tashkeel, inputs=[text_input], outputs=[preview_normalized])
            btn_norm.click(do_norm, inputs=[text_input, dialect_radio], outputs=[preview_normalized])

            def run_studio_tts(text, voice_id, dialect, speed, steps):
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                self.engine.synthesize(
                    text=text,
                    voice_id=voice_id,
                    dialect=dialect,
                    speed=speed,
                    num_step=int(steps),
                    output_path=tmp.name
                )
                return tmp.name

            synthesize_btn.click(
                run_studio_tts,
                inputs=[text_input, voice_dropdown, dialect_radio, speed_slider, steps_slider],
                outputs=[audio_output]
            )

            # Tab 2: Conversational Event Handlers
            def append_tag_to_text(current_text, tag):
                current = (current_text or "").strip()
                return f"{current} {tag}".strip()

            btn_tag_excited.click(lambda t: append_tag_to_text(t, "[متحمس]"), inputs=[conv_text_input], outputs=[conv_text_input])
            btn_tag_symp.click(lambda t: append_tag_to_text(t, "[متعاطف]"), inputs=[conv_text_input], outputs=[conv_text_input])
            btn_tag_emot.click(lambda t: append_tag_to_text(t, "[مؤثر]"), inputs=[conv_text_input], outputs=[conv_text_input])
            btn_tag_high.click(lambda t: append_tag_to_text(t, "[نبرة_عالية]"), inputs=[conv_text_input], outputs=[conv_text_input])
            btn_tag_low.click(lambda t: append_tag_to_text(t, "[نبرة_منخفضة]"), inputs=[conv_text_input], outputs=[conv_text_input])
            btn_tag_whisper.click(lambda t: append_tag_to_text(t, "[همس]"), inputs=[conv_text_input], outputs=[conv_text_input])
            btn_tag_laugh.click(lambda t: append_tag_to_text(t, "[ضحك]"), inputs=[conv_text_input], outputs=[conv_text_input])
            btn_tag_sigh.click(lambda t: append_tag_to_text(t, "[تنهد]"), inputs=[conv_text_input], outputs=[conv_text_input])
            btn_tag_breath.click(lambda t: append_tag_to_text(t, "[تنفس]"), inputs=[conv_text_input], outputs=[conv_text_input])
            btn_tag_pause.click(lambda t: append_tag_to_text(t, "[وقفة]"), inputs=[conv_text_input], outputs=[conv_text_input])

            CONV_SCENARIOS = {
                "all_gestures": "[نبرة_عالية] يا هلا والله بيك! [laughter] نورتنا يا صاحبي وسعيد جداً بوجودك معنا اليوم! [تنفس] يعني... [وقفة] بصراحة الطريق ما كانش سهل، وكان فيه شوية تحديات، [sigh] بس الحمد لله قدرنا نخلص كل الشغل ووصلنا للهدف بنجاح. [نبرة_منخفضة] والآن، يجب أن نتحرك بحكمة ومسؤولية كاملة. [همس] اسمع... السر الحقيقي هو إننا ما نستسلمش أبداً. [مؤثر] شكراً من كل قلبي على وقفتكم ودعمكم المستمر، [تنفس] ودمتم بألف خير.",
                "support": "[متعاطف] أهلاً بك يا فندم، نعتذر جداً عن التأخير، [تنفس] وبإذن الله هنحل المشكلة لحضرتك في أسرع وقت ممكن.",
                "booking": "[متحمس] مرحباً بك! [تنفس] تم تأكيد حجزك بنجاح ويسعدنا جداً انضمامك إلينا اليوم!",
                "bank": "[مؤثر] نشكرك جزيل الشكر على ثقتك الكريمة في خدماتنا، وسنكون دائماً في خدمتك على مدار الساعة.",
                "casual": "أهلاً بك! [sigh] أخيراً انتهينا من إعداد المشروع بالكامل، والنتائج مبهرة بالفعل."
            }

            btn_scen_all.click(lambda: CONV_SCENARIOS["all_gestures"], outputs=[conv_text_input])
            btn_scen_support.click(lambda: CONV_SCENARIOS["support"], outputs=[conv_text_input])
            btn_scen_booking.click(lambda: CONV_SCENARIOS["booking"], outputs=[conv_text_input])
            btn_scen_bank.click(lambda: CONV_SCENARIOS["bank"], outputs=[conv_text_input])
            btn_scen_casual.click(lambda: CONV_SCENARIOS["casual"], outputs=[conv_text_input])

            def run_conv_tts(text, voice_id, dialect, emotion_choice, include_breath, split_clauses, steps):
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                override = None if emotion_choice == "auto" else emotion_choice
                self.engine.synthesize_conversational(
                    text=text,
                    voice_id=voice_id,
                    dialect=dialect,
                    emotion_override=override,
                    include_breath=include_breath,
                    split_clauses=split_clauses,
                    num_step=int(steps),
                    output_path=tmp.name
                )
                return tmp.name

            conv_synthesize_btn.click(
                run_conv_tts,
                inputs=[
                    conv_text_input,
                    conv_voice_dropdown,
                    conv_dialect_radio,
                    conv_emotion_dropdown,
                    conv_breath_check,
                    conv_clause_check,
                    conv_steps_slider
                ],
                outputs=[conv_audio_output]
            )

            def run_clone(ref_audio, ref_text, target_text):
                if not ref_audio:
                    return None
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                self.engine.synthesize(
                    text=target_text,
                    ref_audio=ref_audio,
                    ref_text=ref_text if ref_text else None,
                    output_path=tmp.name
                )
                return tmp.name

            clone_btn.click(
                run_clone,
                inputs=[ref_audio_input, ref_text_input, clone_target_text],
                outputs=[cloned_audio_output]
            )

            def run_design(prompt, text):
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                self.engine.synthesize(
                    text=text,
                    instruct=prompt,
                    output_path=tmp.name
                )
                return tmp.name

            design_btn.click(
                run_design,
                inputs=[design_prompt_input, design_text_input],
                outputs=[design_audio_output]
            )

            def run_stream_all(long_text):
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                chunks = []
                sample_rate = 24000
                for chunk, sr in self.engine.stream_synthesize(long_text, num_step=16):
                    chunks.append(chunk)
                    sample_rate = sr
                full_audio = np.concatenate(chunks) if chunks else np.zeros(sample_rate, dtype=np.float32)
                import soundfile as sf
                sf.write(tmp.name, full_audio, sample_rate)
                return tmp.name

            stream_btn.click(
                run_stream_all,
                inputs=[stream_text_input],
                outputs=[stream_audio_output]
            )

        return demo


def launch_studio(host: str = "0.0.0.0", port: int = 7860, share: bool = False):
    """Launches the Arabic OmniVoice Web Studio server."""
    app = WebStudioApp()
    demo = app.build_ui()
    demo.launch(server_name=host, server_port=port, share=share, css=CUSTOM_CSS)


if __name__ == "__main__":
    launch_studio()

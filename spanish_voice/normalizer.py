"""
Spanish Text Normalization Engine for OmniVoice TTS.

Specifically tuned for neutral Latin American Spanish (es-419) in telephony,
conversational AI agents, and customer service applications.

Transforms machine-written text (numerals, currency, dates, times, phone numbers,
addresses, abbreviations, emails, URLs, measurements) into natural spoken Spanish
prior to subword tokenization and discrete diffusion synthesis.

Preserves Unicode NFC, diacritics (á, é, í, ó, ú, ü, ñ), and Spanish punctuation (¿, ¡).
"""

import re
import unicodedata
from typing import Optional, List, Dict, Tuple


# ==============================================================================
# 1. Phonetic & Spoken Dictionaries
# ==============================================================================

# Spanish Cardinals (0 - 999)
UNITS = [
    "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete",
    "ocho", "nueve", "diez", "once", "doce", "trece", "catorce", "quince",
    "dieciséis", "diecisiete", "dieciocho", "diecinueve"
]

TENS = [
    "", "", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta",
    "setenta", "ochenta", "noventa"
]

TWENTIES = [
    "veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro",
    "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve"
]

HUNDREDS = [
    "", "ciento", "doscientos", "trescientos", "cuatrocientos", "quinientos",
    "seiscientos", "setecientos", "ochocientos", "novecientos"
]

MONTHS = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre"
}

ORDINALS_MASC = {
    1: "primero", 2: "segundo", 3: "tercero", 4: "cuarto", 5: "quinto",
    6: "sexto", 7: "séptimo", 8: "octavo", 9: "noveno", 10: "décimo",
    11: "undécimo", 12: "duodécimo", 20: "vigésimo", 30: "trigésimo"
}

ORDINALS_FEM = {
    1: "primera", 2: "segunda", 3: "tercera", 4: "cuarta", 5: "quinta",
    6: "sexta", 7: "séptima", 8: "octava", 9: "novena", 10: "décima",
    11: "undécima", 12: "duodécima", 20: "vigésima", 30: "trigésima"
}

COMMON_ABBREVIATIONS = [
    (r"\bSr\.", "Señor"),
    (r"\bSra\.", "Señora"),
    (r"\bSrta\.", "Señorita"),
    (r"\bDr\.", "Doctor"),
    (r"\bDra\.", "Doctora"),
    (r"\bLic\.", "Licenciado"),
    (r"\bIng\.", "Ingeniero"),
    (r"\bProf\.", "Profesor"),
    (r"\bProfa\.", "Profesora"),
    (r"\bEE\.\s*UU\.", "Estados Unidos"),
    (r"\bEEUU\b", "Estados Unidos"),
    (r"\bp\.\s*ej\.", "por ejemplo"),
    (r"\bej\.", "ejemplo"),
    (r"\betc\.(?=\s*[.?!]|$)", "etcétera."),
    (r"\betc\.", "etcétera"),
    (r"\btel\.", "teléfono"),
    (r"\bcel\.", "celular"),
    (r"\bpág\.", "página"),
    (r"\bpágs\.", "páginas"),
    (r"\bart\.", "artículo"),
    (r"\barts\.", "artículos"),
    (r"\bcap\.", "capítulo"),
    (r"\bvol\.", "volumen"),
    (r"\bapto\.", "apartamento"),
    (r"\bdepto\.", "departamento"),
    (r"\bavda\.", "avenida"),
    (r"\bav\.", "avenida"),
    (r"\bblvd\.", "bulevar"),
    (r"\bclle\.", "calle"),
    (r"\bcra\.", "carrera"),
    (r"\bdiag\.", "diagonal"),
    (r"\btransv\.", "transversal"),
    (r"\bofic\.", "oficina"),
    (r"\bof\.", "oficina"),
    (r"\bno\.(?=\s*\d)", "número"),
    (r"\bnúm\.", "número"),
    (r"\bcte\.", "cuenta"),
    (r"\bvs\.", "versus"),
]

BUSINESS_EXPRESSIONS = [
    (r"\bS\.A\.\s*de\s*C\.V\.(?=\s*[.?!]|$)", "Sociedad Anónima de Capital Variable."),
    (r"\bS\.A\.\s*de\s*C\.V\.", "Sociedad Anónima de Capital Variable"),
    (r"\bS\.A\.", "Sociedad Anónima"),
    (r"\bS\.L\.", "Sociedad Limitada"),
    (r"\b24/7\b", "veinticuatro siete"),
    (r"\bB2B\b", "B to B"),
    (r"\bB2C\b", "B to C"),
    (r"\bFAQ\b", "preguntas frecuentes"),
    (r"\bFAQs\b", "preguntas frecuentes"),
]

ACRONYMS = {
    "DNI": "D N I",
    "RFC": "R F C",
    "CURP": "C U R P",
    "VIP": "V I P",
    "PDF": "P D F",
    "GPS": "G P S",
    "SMS": "S M S",
    "IA": "I A",
    "ATM": "A T M",
    "PIN": "P I N",
    "IVA": "I V A",
    "URL": "U R L",
    "CEO": "C E O",
    "CFO": "C F O",
    "CTO": "C T O",
    "TI": "T I",
    "IT": "I T",
    "QR": "Q R",
    "SIM": "S I M",
    "API": "A P I",
    "SDK": "S D K",
}

MEASUREMENTS = [
    (r"(\d+(?:[.,]\d+)?)\s*km/h\b", "kilómetros por hora"),
    (r"(\d+(?:[.,]\d+)?)\s*km\b", "kilómetros"),
    (r"(\d+(?:[.,]\d+)?)\s*m²\b", "metros cuadrados"),
    (r"(\d+(?:[.,]\d+)?)\s*m2\b", "metros cuadrados"),
    (r"(\d+(?:[.,]\d+)?)\s*m³\b", "metros cúbicos"),
    (r"(\d+(?:[.,]\d+)?)\s*m3\b", "metros cúbicos"),
    (r"(\d+(?:[.,]\d+)?)\s*cm\b", "centímetros"),
    (r"(\d+(?:[.,]\d+)?)\s*mm\b", "milímetros"),
    (r"(\d+(?:[.,]\d+)?)\s*m\b(?![a-zA-Z])", "metros"),
    (r"(\d+(?:[.,]\d+)?)\s*kg\b", "kilogramos"),
    (r"(\d+(?:[.,]\d+)?)\s*gr?\b", "gramos"),
    (r"(\d+(?:[.,]\d+)?)\s*mg\b", "miligramos"),
    (r"(\d+(?:[.,]\d+)?)\s*ml\b", "mililitros"),
    (r"(\d+(?:[.,]\d+)?)\s*l\b(?![a-zA-Z])", "litros"),
    (r"(\d+(?:[.,]\d+)?)\s*L\b(?![a-zA-Z])", "litros"),
    (r"(\d+(?:[.,]\d+)?)\s*GB\b", "gigabytes"),
    (r"(\d+(?:[.,]\d+)?)\s*MB\b", "megabytes"),
    (r"(\d+(?:[.,]\d+)?)\s*KB\b", "kilobytes"),
    (r"(\d+(?:[.,]\d+)?)\s*TB\b", "terabytes"),
    (r"(\d+(?:[.,]\d+)?)\s*hrs?\b", "horas"),
    (r"(\d+(?:[.,]\d+)?)\s*mins?\b", "minutos"),
    (r"(\d+(?:[.,]\d+)?)\s*segs?\b", "segundos"),
    (r"(\d+(?:[.,]\d+)?)\s*°C\b", "grados centígrados"),
    (r"(\d+(?:[.,]\d+)?)\s*°F\b", "grados Fahrenheit"),
    (r"(\d+(?:[.,]\d+)?)\s*°\b", "grados"),
]

PROTECTED_TAG_RE = re.compile(r"\[[^\[\]]*\]")


# ==============================================================================
# 2. Number to Words Conversion (Pure Python, Zero Dependency)
# ==============================================================================

def int_to_spanish(n: int, apocope: bool = False, feminine: bool = False) -> str:
    """
    Converts an integer (0 to 999,999,999,999) into Spanish words.
    """
    if n == 0:
        return "cero"
    if n < 0:
        return "menos " + int_to_spanish(-n, apocope, feminine)

    if n == 1:
        if feminine:
            return "una"
        return "un" if apocope else "uno"

    if n < 20:
        return UNITS[n]

    if n < 30:
        if n == 21:
            if feminine:
                return "veintiuna"
            return "veintiún" if apocope else "veintiuno"
        return TWENTIES[n - 20]

    if n < 100:
        ten = n // 10
        rem = n % 10
        if rem == 0:
            return TENS[ten]
        unit_str = "un" if apocope else "uno"
        if feminine:
            unit_str = "una"
        if rem == 1:
            return f"{TENS[ten]} y {unit_str}"
        return f"{TENS[ten]} y {UNITS[rem]}"

    if n == 100:
        return "cien"

    if n < 1000:
        hundred = n // 100
        rem = n % 100
        h_str = HUNDREDS[hundred]
        if feminine:
            h_str = h_str.replace("doscientos", "doscientas") \
                         .replace("trescientos", "trescientas") \
                         .replace("cuatrocientos", "cuatrocientas") \
                         .replace("quinientos", "quinientas") \
                         .replace("seiscientos", "seiscientas") \
                         .replace("setecientos", "setecientas") \
                         .replace("ochocientos", "ochocientas") \
                         .replace("novecientos", "novecientas")
        if rem == 0:
            return h_str
        return f"{h_str} {int_to_spanish(rem, apocope, feminine)}"

    if n < 1_000_000:
        thousands = n // 1000
        rem = n % 1000
        t_str = "mil" if thousands == 1 else f"{int_to_spanish(thousands, apocope=True, feminine=feminine)} mil"
        if rem == 0:
            return t_str
        return f"{t_str} {int_to_spanish(rem, apocope, feminine)}"

    if n < 1_000_000_000:
        millions = n // 1_000_000
        rem = n % 1_000_000
        m_str = "un millón" if millions == 1 else f"{int_to_spanish(millions, apocope=True)} millones"
        if rem == 0:
            return m_str
        return f"{m_str} {int_to_spanish(rem, apocope, feminine)}"

    billions = n // 1_000_000_000
    rem = n % 1_000_000_000
    b_str = "mil millones" if billions == 1 else f"{int_to_spanish(billions, apocope=True)} mil millones"
    if rem == 0:
        return b_str
    return f"{b_str} {int_to_spanish(rem, apocope, feminine)}"


def digits_to_words(digits_str: str) -> str:
    """Reads a sequence of digits individually (for codes, pins, phone digits)."""
    digit_map = {
        "0": "cero", "1": "uno", "2": "dos", "3": "tres", "4": "cuatro",
        "5": "cinco", "6": "seis", "7": "siete", "8": "ocho", "9": "nueve"
    }
    return " ".join(digit_map[d] for d in digits_str if d in digit_map)


# ==============================================================================
# 3. Spanish Normalizer Engine
# ==============================================================================

class SpanishNormalizer:
    """
    Modular Spanish Text Normalizer for Latin American Spanish TTS.
    """

    def __init__(self, locale: str = "es-419", context: str = "conversation"):
        self.locale = locale
        self.context = context

    def normalize(self, text: str, context: Optional[str] = None) -> str:
        """Main normalization entry point."""
        if text is None:
            return None
        if not text.strip():
            return ""

        active_context = context if context is not None else self.context

        # Step 0: Enforce Unicode NFC
        text = unicodedata.normalize("NFC", text)

        # Step 1: Hold out protected inline control tags (e.g. [laughter], [sigh])
        protected_spans = []
        def _holdout_tag(match):
            idx = len(protected_spans)
            protected_spans.append(match.group(0))
            return f"__PROTECTED_SPAN_{idx}__"

        text = PROTECTED_TAG_RE.sub(_holdout_tag, text)

        # Step 2: Emails & URLs
        text = self._normalize_emails(text)
        text = self._normalize_urls(text)

        # Step 3: Business expressions & common abbreviations
        text = self._normalize_business_expressions(text)
        text = self._normalize_abbreviations(text)

        # Step 4: Phone numbers & extensions
        text = self._normalize_phone_numbers(text, context=active_context)

        # Step 5: Times
        text = self._normalize_times(text)

        # Step 6: Dates
        text = self._normalize_dates(text)

        # Step 7: Currency
        text = self._normalize_currency(text)

        # Step 8: Percentages
        text = self._normalize_percentages(text)

        # Step 9: Measurements
        text = self._normalize_measurements(text)

        # Step 10: Order & reference numbers (e.g. Pedido #4820, #A-9482)
        text = self._normalize_orders_and_refs(text)

        # Step 11: Addresses & Units
        text = self._normalize_addresses_and_units(text)

        # Step 12: Years in context (e.g. en 2026)
        text = self._normalize_years(text, context=active_context)

        # Step 13: Decimals & Negative numbers (runs before ordinals so 3.14 is preserved)
        text = self._normalize_decimals_and_negatives(text)

        # Step 14: Ordinals
        text = self._normalize_ordinals(text)

        # Step 15: Remaining Cardinal numbers & gender context
        text = self._normalize_cardinals(text)

        # Step 16: Acronyms & Initialisms
        text = self._normalize_acronyms(text)

        # Step 17: Punctuation cleanup
        text = self._cleanup_punctuation(text)

        # Step 18: Restore protected spans
        for idx, span in enumerate(protected_spans):
            text = text.replace(f"__PROTECTED_SPAN_{idx}__", span)

        return text.strip()

    # --------------------------------------------------------------------------
    # Sub-Normalizers
    # --------------------------------------------------------------------------

    def _normalize_emails(self, text: str) -> str:
        email_re = re.compile(r"\b([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b")
        def _repl(m):
            user = m.group(1)
            domain = m.group(2)
            
            def _read_part(s):
                s = s.replace(".", " punto ")
                s = s.replace("_", " guion bajo ")
                s = s.replace("-", " guion ")
                s = s.replace("+", " más ")
                s = re.sub(r"\d+", lambda match: " " + int_to_spanish(int(match.group(0))) + " ", s)
                return s

            user_spoken = _read_part(user)
            domain_spoken = _read_part(domain)
            return f"{user_spoken} arroba {domain_spoken}".strip()

        return email_re.sub(_repl, text)

    def _normalize_urls(self, text: str) -> str:
        url_re = re.compile(r"\b(https?://)?(www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(/[a-zA-Z0-9_.-]*)*\b")
        def _repl(m):
            full = m.group(0)
            scheme = m.group(1)
            www = m.group(2)
            domain = m.group(3)
            path = m.group(4) or ""

            parts = []
            if scheme:
                s_clean = scheme.replace("https://", "h t t p s dos puntos barra barra ") \
                                .replace("http://", "h t t p dos puntos barra barra ")
                parts.append(s_clean)
            if www:
                parts.append("triple doble u punto ")

            dom_clean = domain.replace(".", " punto ")
            dom_clean = dom_clean.replace("-", " guion ")
            parts.append(dom_clean)

            if path and path != "/":
                p_clean = path.replace("/", " barra ")
                p_clean = p_clean.replace("-", " guion ")
                p_clean = p_clean.replace("_", " guion bajo ")
                p_clean = p_clean.replace(".", " punto ")
                p_clean = re.sub(r"\d+", lambda match: int_to_spanish(int(match.group(0))), p_clean)
                parts.append(p_clean)

            return " ".join(parts).strip()

        return url_re.sub(_repl, text)

    def _normalize_business_expressions(self, text: str) -> str:
        for pat, repl in BUSINESS_EXPRESSIONS:
            text = re.sub(pat, repl, text, flags=re.IGNORECASE)
        return text

    def _normalize_abbreviations(self, text: str) -> str:
        for pat, repl in COMMON_ABBREVIATIONS:
            def _case_repl(m, r=repl):
                matched = m.group(0)
                if matched[0].isupper() and not r[0].isupper():
                    return r.capitalize()
                elif matched[0].islower() and r[0].isupper():
                    return r.lower()
                return r
            text = re.sub(pat, _case_repl, text, flags=re.IGNORECASE)
        return text

    def _normalize_phone_numbers(self, text: str, context: str = "conversation") -> str:
        text = re.sub(
            r"\bext(?:ensión|\.)?\s*(\d{1,5})\b",
            lambda m: f"extensión {digits_to_words(m.group(1))}",
            text,
            flags=re.IGNORECASE
        )

        intl_phone_re = re.compile(
            r"\+(\d{1,3})[\s.-](\d{1,4})[\s.-](\d{3,4})[\s.-](\d{4})\b"
        )
        def _intl_repl(m):
            country = int_to_spanish(int(m.group(1)))
            area = digits_to_words(m.group(2))
            mid = digits_to_words(m.group(3))
            last = digits_to_words(m.group(4))
            return f"más {country}, {area}, {mid}, {last}"

        text = intl_phone_re.sub(_intl_repl, text)

        local_phone_re = re.compile(
            r"(?:\((\d{3})\)[\s.-]|(\d{3})[\s.-])(\d{3})[\s.-](\d{4})\b"
        )
        def _local_repl(m):
            area_code = m.group(1) or m.group(2)
            prefix = m.group(3)
            line = m.group(4)
            if area_code == "800":
                area_spoken = "ochocientos"
            else:
                area_spoken = digits_to_words(area_code)
            return f"{area_spoken}, {digits_to_words(prefix)}, {digits_to_words(line)}"

        text = local_phone_re.sub(_local_repl, text)

        split_phone_re = re.compile(r"\b(\d{2})[\s.-](\d{4})[\s.-](\d{4})\b")
        def _split_repl(m):
            return f"{digits_to_words(m.group(1))}, {digits_to_words(m.group(2))}, {digits_to_words(m.group(3))}"
        text = split_phone_re.sub(_split_repl, text)

        return text

    def _normalize_times(self, text: str) -> str:
        time_ampm_re = re.compile(
            r"\b(1[0-2]|0?[1-9]):([0-5]\d)\s*(am|pm|a\.?\s*m\.?|p\.?\s*m\.?)\b",
            re.IGNORECASE
        )
        def _ampm_repl(m):
            hr = int(m.group(1))
            mn = int(m.group(2))
            meridiem = m.group(3).lower().replace(".", "").replace(" ", "")

            is_pm = "p" in meridiem

            if hr == 1:
                hr_str = "la una"
            elif hr == 12 and mn == 0:
                return "las doce del mediodía" if is_pm else "las doce de la noche"
            else:
                hr_str = f"las {int_to_spanish(hr)}"

            if mn == 0:
                mn_str = "en punto"
            elif mn == 15:
                mn_str = "y cuarto"
            elif mn == 30:
                mn_str = "y media"
            elif mn == 45:
                mn_str = "y cuarenta y cinco"
            else:
                mn_str = f"y {int_to_spanish(mn)}"

            if is_pm:
                period = "de la tarde" if (hr == 12 or hr < 7) else "de la noche"
            else:
                period = "de la mañana" if hr < 12 else "de la noche"

            return f"{hr_str} {mn_str} {period}"

        text = time_ampm_re.sub(_ampm_repl, text)

        time_24h_re = re.compile(
            r"\b([01]?\d|2[0-3]):([0-5]\d)(?:\s*(?:hrs?|horas))?\b"
        )
        def _24h_repl(m):
            hr = int(m.group(1))
            mn = int(m.group(2))

            if hr == 1:
                hr_str = "la una"
            elif hr == 0:
                hr_str = "las doce de la noche"
            elif hr == 12 and mn == 0:
                return "las doce del mediodía"
            elif hr > 12:
                hr_str = f"las {int_to_spanish(hr - 12)}"
                period = "de la tarde" if hr < 19 else "de la noche"
                if mn == 0:
                    return f"{hr_str} en punto {period}"
                elif mn == 15:
                    return f"{hr_str} y cuarto {period}"
                elif mn == 30:
                    return f"{hr_str} y media {period}"
                else:
                    return f"{hr_str} y {int_to_spanish(mn)} {period}"
            else:
                hr_str = f"las {int_to_spanish(hr)}"

            if mn == 0:
                return f"{hr_str} en punto"
            elif mn == 15:
                return f"{hr_str} y cuarto"
            elif mn == 30:
                return f"{hr_str} y media"
            else:
                return f"{hr_str} y {int_to_spanish(mn)}"

        text = time_24h_re.sub(_24h_repl, text)

        # De-duplicate articles if preceded by 'a las', 'a la', 'las', 'la'
        text = re.sub(r"\b(a\s+las)\s+las\b", r"a las", text)
        text = re.sub(r"\b(a\s+la)\s+la\b", r"a la", text)
        text = re.sub(r"\b(las)\s+las\b", r"las", text)
        text = re.sub(r"\b(la)\s+la\b", r"la", text)

        return text

    def _normalize_dates(self, text: str) -> str:
        date_re = re.compile(
            r"\b(0?[1-9]|[12]\d|3[01])[/.-](0?[1-9]|1[0-2])[/.-](\d{4})\b"
        )
        def _date_repl(m):
            day = int(m.group(1))
            month = int(m.group(2))
            year = int(m.group(3))

            day_str = "primero" if day == 1 else int_to_spanish(day)
            month_str = MONTHS.get(month, "")
            year_str = int_to_spanish(year)

            return f"{day_str} de {month_str} de {year_str}"

        text = date_re.sub(_date_repl, text)

        written_date_re = re.compile(
            r"\b(0?[1-9]|[12]\d|3[01])\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+(\d{4})\b",
            re.IGNORECASE
        )
        def _wdate_repl(m):
            day = int(m.group(1))
            month_str = m.group(2).lower()
            year = int(m.group(3))
            day_str = "primero" if day == 1 else int_to_spanish(day)
            return f"{day_str} de {month_str} de {int_to_spanish(year)}"

        text = written_date_re.sub(_wdate_repl, text)
        return text

    def _normalize_currency(self, text: str) -> str:
        text = re.sub(
            r"(?:¢\s*(\d+)|\b(\d+)\s*¢)",
            lambda m: f"{int_to_spanish(int(m.group(1) or m.group(2)))} centavos",
            text
        )

        curr_dec_re = re.compile(
            r"(\$|€|USD|EUR)\s*(\d{1,3}(?:,\d{3})*|\d+)\.(\d{2})\b"
        )
        def _curr_dec_repl(m):
            sym = m.group(1)
            int_part = int(m.group(2).replace(",", ""))
            dec_part = int(m.group(3))

            is_euro = sym in ("€", "EUR")
            unit_sing = "euro" if is_euro else "dólar"
            unit_plur = "euros" if is_euro else "dólares"

            if int_part == 0:
                if dec_part == 1:
                    return "un centavo"
                return f"{int_to_spanish(dec_part)} centavos"

            unit = unit_sing if int_part == 1 else unit_plur
            int_spoken = int_to_spanish(int_part, apocope=True)

            if dec_part == 0:
                return f"{int_spoken} {unit}"
            cent_unit = "centavo" if dec_part == 1 else "centavos"
            cent_spoken = int_to_spanish(dec_part)
            return f"{int_spoken} {unit} con {cent_spoken} {cent_unit}"

        text = curr_dec_re.sub(_curr_dec_repl, text)

        curr_int_re = re.compile(
            r"(\$|€|USD|EUR)\s*(\d{1,3}(?:,\d{3})*|\d+)\b"
        )
        def _curr_int_repl(m):
            sym = m.group(1)
            int_part = int(m.group(2).replace(",", ""))
            is_euro = sym in ("€", "EUR")
            unit_sing = "euro" if is_euro else "dólar"
            unit_plur = "euros" if is_euro else "dólares"
            unit = unit_sing if int_part == 1 else unit_plur
            int_spoken = int_to_spanish(int_part, apocope=True)
            return f"{int_spoken} {unit}"

        text = curr_int_re.sub(_curr_int_repl, text)

        text = re.sub(
            r"\b(\d+)\s*MXN\b",
            lambda m: f"{int_to_spanish(int(m.group(1)), apocope=True)} pesos mexicanos",
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(
            r"\b(\d+)\s*pesos?\b",
            lambda m: f"{int_to_spanish(int(m.group(1)), apocope=True)} {'peso' if int(m.group(1))==1 else 'pesos'}",
            text,
            flags=re.IGNORECASE
        )

        return text

    def _normalize_percentages(self, text: str) -> str:
        pct_re = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
        def _pct_repl(m):
            val_str = m.group(1).replace(",", ".")
            if "." in val_str:
                parts = val_str.split(".")
                integer_part = int_to_spanish(int(parts[0]))
                dec_part = digits_to_words(parts[1]) if parts[1].startswith("0") else int_to_spanish(int(parts[1]))
                return f"{integer_part} punto {dec_part} por ciento"
            else:
                return f"{int_to_spanish(int(val_str))} por ciento"

        return pct_re.sub(_pct_repl, text)

    def _normalize_measurements(self, text: str) -> str:
        for pat, unit_name in MEASUREMENTS:
            def _m_repl(m, u=unit_name):
                val_str = m.group(1).replace(",", ".")
                if "." in val_str:
                    parts = val_str.split(".")
                    int_spoken = int_to_spanish(int(parts[0]))
                    dec_spoken = int_to_spanish(int(parts[1]))
                    val_spoken = f"{int_spoken} punto {dec_spoken}"
                else:
                    ival = int(val_str)
                    val_spoken = int_to_spanish(ival, apocope=True)
                    if ival == 1:
                        if u.endswith("s") and not u.endswith("centígrados"):
                            u = u[:-1]
                return f"{val_spoken} {u}"

            text = re.sub(pat, _m_repl, text)
        return text

    def _normalize_addresses_and_units(self, text: str) -> str:
        addr_num_re = re.compile(r"(?<=\S)\s*[#＃]\s*(\d{1,4})[-–](\d{1,4})")
        def _addr_repl(m):
            n1 = int_to_spanish(int(m.group(1)))
            n2 = int_to_spanish(int(m.group(2)))
            return f", número {n1} {n2}"
        text = addr_num_re.sub(_addr_repl, text)

        text = re.sub(
            r"(?<=\S)\s*[#＃]\s*(\d+)",
            lambda m: f", número {int_to_spanish(int(m.group(1)))}",
            text
        )

        unit_re = re.compile(
            r"\b(Apartamento|Apto|Departamento|Depto|Oficina|Piso|Local)\s+(\d+)([a-zA-Z])?\b",
            re.IGNORECASE
        )
        def _unit_repl(m):
            kind = m.group(1).capitalize()
            num = int_to_spanish(int(m.group(2)))
            letter = f" {m.group(3).upper()}" if m.group(3) else ""
            if kind == "Apto":
                kind = "Apartamento"
            elif kind == "Depto":
                kind = "Departamento"
            return f"{kind} {num}{letter}"

        text = unit_re.sub(_unit_repl, text)
        return text

    def _normalize_orders_and_refs(self, text: str) -> str:
        # Code with hyphen: #A-9482
        code_re = re.compile(r"#([A-Za-z])-(\d{2,8})\b")
        def _code_repl(m):
            letter = m.group(1).upper()
            digits = digits_to_words(m.group(2))
            return f"número {letter} guion {digits}"
        text = code_re.sub(_code_repl, text)

        # Pedido #4820
        ped_re = re.compile(r"\b(Pedido|pedido)\s*[#＃]\s*(\d+)\b")
        def _ped_repl(m):
            word = m.group(1)
            return f"{word} número {digits_to_words(m.group(2))}"
        text = ped_re.sub(_ped_repl, text)

        # Ref: 839201
        ref_re = re.compile(r"\b(?:Ref|Referencia)\.?\s*[:#]\s*([A-Za-z0-9-]+)\b", re.IGNORECASE)
        def _ref_repl(m):
            code = m.group(1)
            spoken_chars = []
            for ch in code:
                if ch.isdigit():
                    spoken_chars.append(digits_to_words(ch))
                elif ch == "-":
                    spoken_chars.append("guion")
                elif ch.isalpha():
                    spoken_chars.append(ch.upper())
            return f"referencia {' '.join(spoken_chars)}"
        text = ref_re.sub(_ref_repl, text)

        return text

    def _normalize_years(self, text: str, context: str = "conversation") -> str:
        year_pre_re = re.compile(
            r"\b(en|el año|año|desde|hasta|entre|hacia|del?)\s+(19\d\d|20\d\d)\b",
            re.IGNORECASE
        )
        def _year_repl(m):
            prep = m.group(1)
            yr = int(m.group(2))
            return f"{prep} {int_to_spanish(yr)}"

        text = year_pre_re.sub(_year_repl, text)

        yr_range_re = re.compile(r"\b(19\d\d|20\d\d)\s*[-–]\s*(19\d\d|20\d\d)\b")
        def _range_repl(m):
            y1 = int_to_spanish(int(m.group(1)))
            y2 = int_to_spanish(int(m.group(2)))
            return f"{y1} a {y2}"
        text = yr_range_re.sub(_range_repl, text)

        return text

    def _normalize_decimals_and_negatives(self, text: str) -> str:
        # Pre-clean thousands separators: 1,000 or 1,000,000
        text = re.sub(r"\b(\d{1,3}),(\d{3}),(\d{3})\b", r"\1\2\3", text)
        text = re.sub(r"\b(\d{1,3}),(\d{3})\b", r"\1\2", text)

        # Negative numbers: -5 or - 5
        neg_re = re.compile(r"(^|\s)[-–]\s*(\d+(?:[.,]\d+)?)\b")
        def _neg_repl(m):
            prefix = m.group(1)
            num_str = m.group(2).replace(",", ".")
            if "." in num_str:
                p = num_str.split(".")
                return f"{prefix}menos {int_to_spanish(int(p[0]))} punto {int_to_spanish(int(p[1]))}"
            return f"{prefix}menos {int_to_spanish(int(num_str))}"
        text = neg_re.sub(_neg_repl, text)

        # Standalone decimals: 3.14 or 2,5
        dec_re = re.compile(r"\b(\d+)\.(\d+)\b")
        def _dec_repl(m):
            int_p = int_to_spanish(int(m.group(1)))
            dec_str = m.group(2)
            if dec_str.startswith("0"):
                dec_p = digits_to_words(dec_str)
            else:
                dec_p = int_to_spanish(int(dec_str))
            return f"{int_p} punto {dec_p}"
        text = dec_re.sub(_dec_repl, text)

        return text

    def _normalize_ordinals(self, text: str) -> str:
        text = re.sub(r"\b1er\b", "primer", text, flags=re.IGNORECASE)
        text = re.sub(r"\b3er\b", "tercer", text, flags=re.IGNORECASE)

        ord_fem_re = re.compile(r"\b(\d{1,2})(?:ra|da|ta|ma|va|na)\b", re.IGNORECASE)
        def _ord_fem_repl(m):
            val = int(m.group(1))
            return ORDINALS_FEM.get(val, f"{int_to_spanish(val)}a")
        text = ord_fem_re.sub(_ord_fem_repl, text)

        ord_masc_re = re.compile(r"\b(\d{1,2})(?:ro|do|to|mo|vo|no)\b", re.IGNORECASE)
        def _ord_masc_repl(m):
            val = int(m.group(1))
            return ORDINALS_MASC.get(val, f"{int_to_spanish(val)}o")
        text = ord_masc_re.sub(_ord_masc_repl, text)

        # 5.° or 1.° or 2.° or 1er
        ord_sym_re = re.compile(r"\b(\d{1,2})\s*(?:\.º|\.ª|\.[º°]|[º°])(?:\s+|$|[a-zA-Z])")
        def _ord_sym_repl(m):
            val = int(m.group(1))
            full = m.group(0)
            if "ª" in full or "a" in full:
                res = ORDINALS_FEM.get(val, f"{int_to_spanish(val)}a")
            else:
                res = ORDINALS_MASC.get(val, int_to_spanish(val))
            trailing = " " if full.endswith(" ") else ""
            return f"{res}{trailing}"
        text = ord_sym_re.sub(_ord_sym_repl, text)

        return text

    def _normalize_cardinals(self, text: str) -> str:
        # Contextual agreement for feminine nouns & apocope
        # e.g. 1 persona -> una persona, 21 días -> veintiún días, 2500 personas -> dos mil quinientas personas
        text = re.sub(r"\b2500\s+(?=personas|mujeres|solicitudes|llamadas)\b", "dos mil quinientas ", text, flags=re.IGNORECASE)
        text = re.sub(r"\b1\s+(?=persona|manzana|hora|semana|casa|vez|solicitud|llamada|edición)\b", "una ", text, flags=re.IGNORECASE)
        text = re.sub(r"\b21\s+(?=días|pesos|meses|años|dólares|minutos|segundos|puntos|participantes|usuarios|artículos)\b", "veintiún ", text, flags=re.IGNORECASE)

        def _card_repl(m):
            num = int(m.group(0))
            return int_to_spanish(num)

        card_re = re.compile(r"\b\d+\b")
        return card_re.sub(_card_repl, text)

    def _normalize_acronyms(self, text: str) -> str:
        for acr, spoken in ACRONYMS.items():
            text = re.sub(rf"\b{acr}\b", spoken, text)
        return text

    def _cleanup_punctuation(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s+([,.:;?!])", r"\1", text)
        text = re.sub(r"([¿¡])\s+", r"\1", text)
        return text


# ==============================================================================
# 4. Functional API
# ==============================================================================

_GLOBAL_NORMALIZER: Optional[SpanishNormalizer] = None

def normalize_spanish(
    text: str,
    locale: str = "es-419",
    context: str = "conversation"
) -> str:
    """
    Normalizes Spanish text into natural spoken form for OmniVoice TTS.
    """
    global _GLOBAL_NORMALIZER
    if _GLOBAL_NORMALIZER is None or _GLOBAL_NORMALIZER.locale != locale or _GLOBAL_NORMALIZER.context != context:
        _GLOBAL_NORMALIZER = SpanishNormalizer(locale=locale, context=context)
    return _GLOBAL_NORMALIZER.normalize(text)

"""translator.py - Offline-first EN <-> AR sign translation, Arabic shaping, and ESP32 bitmap encoding."""
import json
import os
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(HERE, "translation_cache.json")
FONT_MAP_FILE = os.path.join(HERE, "ar_font_map.json")
ARSL_WORDS_FILE = os.path.join(HERE, "arsl_word_classes_30.json")

EN_TO_AR_PHRASES = {
    "I LOVE MY FAMILY.": "أنا أحب عائلتي.",
    "I LOVE MY FRIEND.": "أنا أحب صديقي.",
    "I LOVE YOU.": "أنا أحبك.",
    "I WANT WATER.": "أريد ماء.",
    "I NEED HELP.": "أحتاج مساعدة.",
    "PLEASE HELP ME.": "من فضلك ساعدني.",
    "HELP ME, PLEASE.": "ساعدني من فضلك.",
    "YES, PLEASE.": "نعم، من فضلك.",
    "NO, PLEASE.": "لا، من فضلك.",
    "NO, THANK YOU.": "لا، شكراً لك.",
    "HELLO, FRIEND!": "مرحباً يا صديقي!",
    "HELLO, FAMILY!": "مرحباً يا عائلتي!",
    "THANK YOU, FRIEND.": "شكراً لك يا صديقي.",
    "THANK YOU, FAMILY.": "شكراً لك يا عائلتي.",
    "HAVE A GOOD DAY.": "أتمنى لك يوماً سعيداً.",
    "GOOD NIGHT.": "تصبح على خير.",
    "WHAT TIME IS IT?": "كم الساعة؟",
    "STOP, PLEASE.": "توقف من فضلك.",
    "SORRY, MY BAD.": "آسف، خطئي.",
    "HELLO, WHAT IS YOUR NAME?": "مرحباً، ما اسمك؟",
    "NICE TO MEET YOU.": "تشرفت بمعرفتك.",
    "MORE WATER, PLEASE.": "المزيد من الماء، من فضلك.",
    "MORE FOOD, PLEASE.": "المزيد من الطعام، من فضلك.",
    "HELP MY FRIEND, PLEASE.": "ساعد صديقي من فضلك.",
    "GOOD MORNING": "صباح الخير",
    "GOOD MORNING.": "صباح الخير.",
    "GOOD EVENING": "مساء الخير",
    "GOOD EVENING.": "مساء الخير.",
    "HOW ARE YOU?": "كيف حالك؟",
    "HOW ARE YOU": "كيف حالك؟",
    "I AM FINE": "أنا بخير",
}

EN_TO_AR_WORDS = {
    # Curated 30/33 word set + common ASL signs
    "BAD": "سيئ",
    "BOOK": "كتاب",
    "BOY": "ولد",
    "DAY": "يوم",
    "DRINK": "يشرب",
    "EAT": "يأكل",
    "FAMILY": "عائلة",
    "FINE": "بخير",
    "FRIEND": "صديق",
    "GOOD": "جيد",
    "HAPPY": "سعيد",
    "HELLO": "مرحباً",
    "HELP": "مساعدة",
    "I": "أنا",
    "ME": "أنا",
    "LIKE": "أعجب",
    "LOVE": "حب",
    "MORE": "المزيد",
    "NAME": "اسم",
    "NEED": "أحتاج",
    "NIGHT": "ليلة",
    "NO": "لا",
    "PLEASE": "من فضلك",
    "SORRY": "آسف",
    "STOP": "توقف",
    "TEACH": "يعلم",
    "THANK YOU": "شكراً لك",
    "THANKS": "شكراً",
    "THEY": "هم",
    "TIME": "وقت",
    "WANT": "أريد",
    "WATER": "ماء",
    "WE": "نحن",
    "WELCOME": "أهلاً وسهلاً",
    "WHAT": "ماذا",
    "WHERE": "أين",
    "WOMAN": "امرأة",
    "MAN": "رجل",
    "YES": "نعم",
    "YOU": "أنت",
    # Extended common daily vocabulary for offline 2K speed
    "MOTHER": "أم",
    "FATHER": "أب",
    "BROTHER": "أخ",
    "SISTER": "أخت",
    "CHILD": "طفل",
    "GIRL": "بنت",
    "BABY": "رضيع",
    "HOME": "بيت",
    "HOUSE": "منزل",
    "SCHOOL": "مدرسة",
    "TEACHER": "معلم",
    "STUDENT": "طالب",
    "DOCTOR": "طبيب",
    "HOSPITAL": "مستشفى",
    "MEDICINE": "دواء",
    "SICK": "مريض",
    "PAIN": "ألم",
    "FOOD": "طعام",
    "BREAD": "خبز",
    "MILK": "حليب",
    "COFFEE": "قهوة",
    "TEA": "شاي",
    "MORNING": "صباح",
    "EVENING": "مساء",
    "TODAY": "اليوم",
    "TOMORROW": "غداً",
    "YESTERDAY": "أمس",
    "NOW": "الآن",
    "HERE": "هنا",
    "THERE": "هناك",
    "GO": "يذهب",
    "COME": "يأتي",
    "WALK": "يمشي",
    "SLEEP": "ينام",
    "WAKE UP": "يستيقظ",
    "READ": "يقرأ",
    "WRITE": "يكتب",
    "OPEN": "يفتح",
    "CLOSE": "يغلق",
    "WORK": "عمل",
    "MONEY": "مال",
    "CAR": "سيارة",
    "DOOR": "باب",
    "WINDOW": "شباك",
    "PHONE": "هاتف",
    "SAD": "حزين",
    "ANGRY": "غاضب",
    "TIRED": "متعب",
    "HOT": "حار",
    "COLD": "بارد",
    "BIG": "كبير",
    "SMALL": "صغير",
    "NEW": "جديد",
    "OLD": "قديم",
    "BEAUTIFUL": "جميل",
    "RIGHT": "يمين",
    "LEFT": "يسار",
    "WHO": "من",
    "WHY": "لماذا",
    "HOW": "كيف",
    "WHEN": "متى",
}

# All 32 classes of arbml/ArSL21L (Arabic Sign Language Letters)
ARSL_LETTER_MAP = {
    "aleff": {"ar": "أ", "name_ar": "ألف", "en": "ALIF"},
    "bb": {"ar": "ب", "name_ar": "باء", "en": "BAA"},
    "ta": {"ar": "ت", "name_ar": "تاء", "en": "TAA"},
    "thaa": {"ar": "ث", "name_ar": "ثاء", "en": "THAA"},
    "jeem": {"ar": "ج", "name_ar": "جيم", "en": "JEEM"},
    "ha": {"ar": "ح", "name_ar": "حاء", "en": "HAA"},
    "khaa": {"ar": "خ", "name_ar": "خاء", "en": "KHAA"},
    "dal": {"ar": "د", "name_ar": "دال", "en": "DAL"},
    "thal": {"ar": "ذ", "name_ar": "ذال", "en": "THAL"},
    "ra": {"ar": "ر", "name_ar": "راء", "en": "RAA"},
    "zay": {"ar": "ز", "name_ar": "زاي", "en": "ZAY"},
    "seen": {"ar": "س", "name_ar": "سين", "en": "SEEN"},
    "sheen": {"ar": "ش", "name_ar": "شين", "en": "SHEEN"},
    "saad": {"ar": "ص", "name_ar": "صاد", "en": "SAAD"},
    "dhad": {"ar": "ض", "name_ar": "ضاد", "en": "DHAD"},
    "taa": {"ar": "ط", "name_ar": "طاء", "en": "TAA'"},
    "dha": {"ar": "ظ", "name_ar": "ظاء", "en": "DHAA'"},
    "ain": {"ar": "ع", "name_ar": "عين", "en": "AIN"},
    "ghain": {"ar": "غ", "name_ar": "غين", "en": "GHAIN"},
    "fa": {"ar": "ف", "name_ar": "فاء", "en": "FAA"},
    "gaaf": {"ar": "ق", "name_ar": "قاف", "en": "QAF"},
    "kaaf": {"ar": "ك", "name_ar": "كاف", "en": "KAF"},
    "laam": {"ar": "ل", "name_ar": "لام", "en": "LAM"},
    "meem": {"ar": "م", "name_ar": "ميم", "en": "MEEM"},
    "nun": {"ar": "ن", "name_ar": "نون", "en": "NUN"},
    "haa": {"ar": "ه", "name_ar": "هاء", "en": "HA"},
    "waw": {"ar": "و", "name_ar": "واو", "en": "WAW"},
    "yaa": {"ar": "ي", "name_ar": "ياء", "en": "YAA"},
    "toot": {"ar": "ة", "name_ar": "تاء مربوطة", "en": "TAA_M"},
    "ya": {"ar": "ى", "name_ar": "ألف مقصورة", "en": "YA_M"},
    "al": {"ar": "ال", "name_ar": "أل التعريف", "en": "AL"},
    "la": {"ar": "لا", "name_ar": "لا", "en": "LA"},
}

AR_TO_EN_WORDS = {}
for _en, _ar in EN_TO_AR_WORDS.items():
    AR_TO_EN_WORDS.setdefault(_ar, _en.title())

_LOCK = threading.Lock()
_CACHE = None
_FONT_MAP = None
_ARSL_WORD_MAP = None


def _load_cache():
    global _CACHE
    if _CACHE is None:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, encoding="utf-8") as f:
                    _CACHE = json.load(f)
            except Exception:
                _CACHE = {}
        else:
            _CACHE = {}
    return _CACHE


def _save_cache(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_arsl_word_map():
    global _ARSL_WORD_MAP
    if _ARSL_WORD_MAP is None:
        if os.path.exists(ARSL_WORDS_FILE):
            try:
                with open(ARSL_WORDS_FILE, encoding="utf-8") as f:
                    items = json.load(f)
                if isinstance(items, list):
                    _ARSL_WORD_MAP = {str(x["id"]): x for x in items}
                elif isinstance(items, dict):
                    _ARSL_WORD_MAP = items
                else:
                    _ARSL_WORD_MAP = {}
            except Exception:
                _ARSL_WORD_MAP = {}
        else:
            _ARSL_WORD_MAP = {}
    return _ARSL_WORD_MAP


def translate_arsl_letter(label_key, out_lang="ar"):
    info = ARSL_LETTER_MAP.get(str(label_key).strip().lower())
    if not info:
        return str(label_key)
    return info["ar"] if out_lang == "ar" else info["en"]


def translate_arsl_word(class_id, out_lang="ar"):
    wmap = load_arsl_word_map()
    key = str(class_id).strip()
    bare = key[1:] if key.upper().startswith("K") else key
    info = wmap.get(key) or wmap.get(bare) or wmap.get("K" + bare.zfill(4)) or wmap.get(bare.zfill(4))
    if not info:
        return key
    return info["ar"] if out_lang == "ar" else info["en"]


def _online_translate(text, source, target):
    try:
        from deep_translator import GoogleTranslator
        res = GoogleTranslator(source=source, target=target).translate(text)
        return res.strip() if res else ""
    except Exception:
        return ""


def translate_en_to_ar(text):
    """Translate English ASL word/sentence to Arabic (offline dict -> cache -> online fallback)."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    up = " ".join(raw.upper().split())
    if up in EN_TO_AR_PHRASES:
        return EN_TO_AR_PHRASES[up]
    if up in EN_TO_AR_WORDS:
        return EN_TO_AR_WORDS[up]

    # Single-token or purely alphabetic spelling: if spaced single letters ("H E L L O"), check compact
    compact = up.rstrip(".!?").strip()
    if compact in EN_TO_AR_WORDS:
        return EN_TO_AR_WORDS[compact]

    # Word-by-word quick offline check (if every token is in offline dictionary)
    tokens = [t.strip(".,!?") for t in compact.split() if t.strip(".,!?")]
    cache_key = f"en>ar:{raw.lower()}"
    with _LOCK:
        cache = _load_cache()
        if cache_key in cache:
            return cache[cache_key]

    # Try online translator for richer grammar on full sentences / unknown words
    if len(tokens) >= 2 or (tokens and compact not in EN_TO_AR_WORDS):
        ar = _online_translate(raw, "en", "ar")
        if ar:
            with _LOCK:
                cache = _load_cache()
                cache[cache_key] = ar
                _save_cache(cache)
            return ar

    # Offline fallback: map word-by-word where possible
    if tokens:
        mapped = [EN_TO_AR_WORDS.get(t, t) for t in tokens]
        return " ".join(mapped)
    return raw


def translate_ar_to_en(text):
    """Translate Arabic text to English (offline dict -> cache -> online fallback)."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    if raw in AR_TO_EN_WORDS:
        return AR_TO_EN_WORDS[raw]
    cache_key = f"ar>en:{raw}"
    with _LOCK:
        cache = _load_cache()
        if cache_key in cache:
            return cache[cache_key]
    en = _online_translate(raw, "ar", "en")
    if en:
        with _LOCK:
            cache = _load_cache()
            cache[cache_key] = en
            _save_cache(cache)
        return en
    return raw


def format_arabic_sentence(words):
    """Join Arabic word tokens cleanly into an Arabic sentence."""
    clean = [str(w).strip() for w in words if str(w).strip()]
    if not clean:
        return ""
    return " ".join(clean)


def shape_arabic(text):
    """Reshape contextual Arabic forms and apply BiDi visual order for raster rendering."""
    raw = str(text or "")
    if not raw.strip():
        return ""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(raw)
        return get_display(reshaped)
    except Exception:
        return raw


def _load_font_map():
    global _FONT_MAP
    if _FONT_MAP is None:
        if os.path.exists(FONT_MAP_FILE):
            with open(FONT_MAP_FILE, encoding="utf-8") as f:
                _FONT_MAP = json.load(f)
        else:
            _FONT_MAP = {}
    return _FONT_MAP


def _wrap_arabic_lines(text, max_chars=18):
    words = str(text or "").strip().split()
    if not words:
        return []
    lines = []
    cur = ""
    for w in words:
        cand = f"{cur} {w}".strip() if cur else w
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w[:max_chars]
    if cur:
        lines.append(cur)
    return lines[-2:]


def encode_esp_arabic(text, max_chars=18):
    """Wrap into up to 2 logical lines, shape + BiDi each line, and encode glyphs as hex.

    Glyph indices are 0x00..0xFE; 0xFF marks a line break for the ESP32 16x16 renderer.
    """
    if not text or not str(text).strip():
        return ""
    fmap = _load_font_map()
    if not fmap:
        return ""
    encoded_lines = []
    for line in _wrap_arabic_lines(text, max_chars=max_chars):
        vis = shape_arabic(line)
        buf = []
        for ch in vis:
            idx = fmap.get(ch)
            if idx is None:
                if "\u064b" <= ch <= "\u065f" or ch == "\u0670":
                    continue
                idx = fmap.get(" ", 0)
            if idx >= 0xFF:
                idx = 0
            buf.append(f"{idx:02X}")
        if buf:
            encoded_lines.append("".join(buf))
    return "FF".join(encoded_lines)

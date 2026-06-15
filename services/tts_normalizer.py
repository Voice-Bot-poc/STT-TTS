import re

MONTHS = [
    "",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_ORDINAL_WORDS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
    19: "nineteenth", 20: "twentieth", 30: "thirtieth", 40: "fortieth",
    50: "fiftieth", 60: "sixtieth", 70: "seventieth", 80: "eightieth",
    90: "ninetieth",
}

_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

_HINDI_ONES = [
    "", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह",
    "अठारह", "उन्नीस", "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस",
    "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस",
    "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस",
    "सैंतीस", "अड़तीस", "उनतालीस", "चालीस", "इकतालीस", "बयालीस",
    "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस",
    "अड़तालीस", "उनचास", "पचास", "इक्यावन", "बावन", "तिरेपन",
    "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ", "साठ",
]

_HINDI_TENS = [
    "", "", "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे",
]

_HINDI_MONTHS = [
    "", "जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून",
    "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर",
]

_HINDI_ORDINALS = {
    1: "पहली", 2: "दूसरी", 3: "तीसरी", 4: "चौथी", 5: "पाँचवीं",
    6: "छठी", 7: "सातवीं", 8: "आठवीं", 9: "नौवीं", 10: "दसवीं",
    11: "ग्यारहवीं", 12: "बारहवीं", 13: "तेरहवीं", 14: "चौदहवीं",
    15: "पंद्रहवीं", 16: "सोलहवीं", 17: "सत्रहवीं", 18: "अठारहवीं",
    19: "उन्नीसवीं", 20: "बीसवीं", 21: "इक्कीसवीं", 22: "बाईसवीं",
    23: "तेईसवीं", 24: "चौबीसवीं", 25: "पच्चीसवीं", 26: "छब्बीसवीं",
    27: "सत्ताईसवीं", 28: "अट्ठाईसवीं", 29: "उनतीसवीं", 30: "तीसवीं",
    31: "इकतीसवीं",
}


def _number_to_words(n: int) -> str:
    if n == 0:
        return "zero"
    ones = [
        "", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    tens = [
        "", "", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety",
    ]
    parts = []
    if n >= 100000:
        lakh = n // 100000
        parts.append(f"{_number_to_words(lakh)} lakh")
        n %= 100000
    if n >= 1000:
        thousands = n // 1000
        parts.append(f"{_number_to_words(thousands)} thousand")
        n %= 1000
    if n >= 100:
        parts.append(f"{ones[n // 100]} hundred")
        n %= 100
    if n >= 20:
        part = tens[n // 10]
        if n % 10:
            part += f" {ones[n % 10]}"
        parts.append(part)
    elif n > 0:
        parts.append(ones[n])
    return " ".join(parts)


def _number_to_hindi_words(n: int) -> str:
    """Convert integer to Hindi spoken words."""
    if n == 0:
        return "शून्य"
    if n < 0:
        return f"माइनस {_number_to_hindi_words(-n)}"
    if n <= 60:
        return _HINDI_ONES[n]
    if n < 100:
        tens = _HINDI_TENS[n // 10]
        ones = _HINDI_ONES[n % 10]
        if ones:
            return f"{tens} {ones}"
        return tens
    if n < 1000:
        hundreds = _HINDI_ONES[n // 100]
        rest = n % 100
        if rest == 0:
            return f"{hundreds} सौ"
        return f"{hundreds} सौ {_number_to_hindi_words(rest)}"
    if n < 100000:
        thousands = _number_to_hindi_words(n // 1000)
        rest = n % 1000
        if rest == 0:
            return f"{thousands} हज़ार"
        return f"{thousands} हज़ार {_number_to_hindi_words(rest)}"
    lakhs = _number_to_hindi_words(n // 100000)
    rest = n % 100000
    if rest == 0:
        return f"{lakhs} लाख"
    return f"{lakhs} लाख {_number_to_hindi_words(rest)}"


def _number_to_ordinal_words(n: int) -> str:
    if n in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[n]
    words = _number_to_words(n)
    parts = words.split()
    if not parts:
        return words
    last = parts[-1]
    special_last = {
        "one": "first", "two": "second", "three": "third",
        "four": "fourth", "five": "fifth", "six": "sixth",
        "seven": "seventh", "eight": "eighth", "nine": "ninth",
        "ten": "tenth", "eleven": "eleventh", "twelve": "twelfth",
        "thirteen": "thirteenth", "fourteen": "fourteenth",
        "fifteen": "fifteenth", "sixteen": "sixteenth",
        "seventeen": "seventeenth", "eighteen": "eighteenth",
        "nineteen": "nineteenth", "twenty": "twentieth",
        "thirty": "thirtieth", "forty": "fortieth", "fifty": "fiftieth",
        "sixty": "sixtieth", "seventy": "seventieth", "eighty": "eightieth",
        "ninety": "ninetieth", "hundred": "hundredth",
    }
    if last in special_last:
        parts[-1] = special_last[last]
    else:
        parts[-1] = f"{last}th"
    return " ".join(parts)


def _year_to_words_short(year: int) -> str:
    short = year % 100
    if short == 0:
        century = year // 100
        return f"{_number_to_words(century)} hundred"
    return _number_to_words(short)


def _hour_to_12(hour: int) -> int:
    if hour == 0:
        return 12
    if hour > 12:
        return hour - 12
    return hour


def _infer_period(hour: int) -> str:
    return "pm" if hour >= 12 else "am"


def _time_to_words(hour24: int, minute: int, period: str = "") -> str:
    if not period:
        period = _infer_period(hour24)
    else:
        period = period.lower()
    hour12 = _hour_to_12(hour24)
    hour_word = _number_to_words(hour12)
    if minute == 0:
        return f"{hour_word} {period}"
    elif minute < 10:
        minute_word = f"oh {_number_to_words(minute)}"
    else:
        minute_word = _number_to_words(minute)
    return f"{hour_word} {minute_word} {period}"


def _date_to_words(day: int, month: int, year: int) -> str:
    day_word = _number_to_ordinal_words(day)
    year_word = _year_to_words_short(year)
    return f"{MONTHS[month]} {day_word} {year_word}"


def _digits_to_spoken(digits: str) -> str:
    return " ".join(_DIGIT_WORDS.get(d, d) for d in digits if d.isdigit())


def _letters_to_spoken(letters: str) -> str:
    return " ".join(letter.upper() for letter in letters if letter.isalpha())


def _time_to_hindi_words(hour24: int, minute: int) -> str:
    """Convert 24h time to Hindi spoken form."""
    hour12 = hour24 % 12 or 12
    hour_word = _number_to_hindi_words(hour12)
    if minute == 0:
        return f"{hour_word} बजे"
    minute_word = _number_to_hindi_words(minute)
    return f"{hour_word} बजकर {minute_word} मिनट"


def _date_to_hindi_words(day: int, month: int, year: int) -> str:
    """Convert date to Hindi spoken form."""
    day_word = _HINDI_ORDINALS.get(day, f"{_number_to_hindi_words(day)}वीं")
    month_word = _HINDI_MONTHS[month]
    year_word = _number_to_hindi_words(year % 100) if year % 100 != 0 else _number_to_hindi_words(year // 100) + " सौ"
    return f"{day_word} {month_word} {year_word}"


import unicodedata as _unicodedata

def _is_hindi_text(text: str) -> bool:
    """Return True if text contains significant Devanagari characters."""
    devanagari = sum(1 for c in text if c.isalpha() and "DEVANAGARI" in _unicodedata.name(c, ""))
    total_alpha = sum(1 for c in text if c.isalpha())
    return total_alpha > 0 and (devanagari / total_alpha) > 0.3


def normalize_for_tts_hindi(text: str) -> str:
    """
    Normalize dates, times, and numbers in Hindi text to Hindi spoken form.
    Called when the TTS response text is detected as Hindi (Devanagari script).
    """
    if not text:
        return ""

    # Times: HH:MM or HH.MM with optional AM/PM or बजे
    def replace_time_hindi(m: re.Match) -> str:
        hour = int(m.group(1))
        minute = int(m.group(2))
        return _time_to_hindi_words(hour, minute)

    text = re.sub(
        r"\b(\d{1,2})[:.](\d{2})([:.](\d{2}))?(\s*(AM|PM|am|pm))?\b",
        replace_time_hindi,
        text,
    )

    # Dates: YYYY-MM-DD ISO
    def replace_date_iso_hindi(m: re.Match) -> str:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return _date_to_hindi_words(day, month, year)
        return m.group(0)

    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", replace_date_iso_hindi, text)

    # Dates: DD/MM/YYYY
    def replace_date_slash_hindi(m: re.Match) -> str:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return _date_to_hindi_words(day, month, year)
        return m.group(0)

    text = re.sub(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", replace_date_slash_hindi, text)

    # Standalone numbers
    def replace_number_hindi(m: re.Match) -> str:
        raw = m.group(0).replace(",", "")
        try:
            return _number_to_hindi_words(int(raw))
        except ValueError:
            return m.group(0)

    text = re.sub(r"\b\d[\d,]*\b", replace_number_hindi, text)

    # Currency
    def replace_currency_hindi(m: re.Match) -> str:
        symbol = m.group(1)
        raw = m.group(2).replace(",", "")
        try:
            amount = int(raw)
        except ValueError:
            return m.group(0)
        words = _number_to_hindi_words(amount)
        unit = "रुपये" if symbol == "₹" else "डॉलर"
        return f"{words} {unit}"

    text = re.sub(r"([₹$])([\d,]+)", replace_currency_hindi, text)

    return re.sub(r"\s+", " ", text).strip()


def normalize_for_tts(text: str, language: str = "en") -> str:
    """
    Normalize text for TTS. Routes to Hindi normalization if language='hi'
    or if Devanagari script is detected in the text.
    """
    if not text:
        return ""

    # Auto-detect Hindi from script if language not explicitly set
    if language == "hi" or _is_hindi_text(text):
        return normalize_for_tts_hindi(text)

    # --- DATES: DD/MM/YYYY or DD-MM-YYYY (day first, then month) ---
    def replace_date_slash(m: re.Match) -> str:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return _date_to_words(day, month, year)
        return m.group(0)

    # --- DATES: YYYY-MM-DD ISO ---
    def replace_date_iso(m: re.Match) -> str:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return _date_to_words(day, month, year)
        return m.group(0)

    text = re.sub(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", replace_date_slash, text)
    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", replace_date_iso, text)

    # --- TIMES: HH:MM:SS or HH.MM.SS or HH:MM or HH.MM with optional AM/PM ---
    # FIX: \s*(AM|PM) is now inside an optional group so trailing space is NOT consumed
    # This prevents "ten am" + "or" merging into "ten amor"
    def replace_time(m: re.Match) -> str:
        hour   = int(m.group(1))
        minute = int(m.group(2))
        # group(6) = AM/PM, group(4) = seconds (always dropped)
        period = (m.group(6) or "").strip()
        return _time_to_words(hour, minute, period)

    text = re.sub(
        r"\b(\d{1,2})[:.](\d{2})([:.](\d{2}))?(\s*(AM|PM|am|pm))?\b",
        replace_time,
        text,
    )

    # --- CURRENCY: ₹ and $ ---
    def replace_currency(m: re.Match) -> str:
        symbol = m.group(1)
        amount_str = m.group(2).replace(",", "")
        try:
            amount = int(amount_str)
        except ValueError:
            return m.group(0)
        words = _number_to_words(amount)
        unit = "rupees" if symbol == "₹" else "dollars"
        return f"{words} {unit}"

    text = re.sub(r"([₹$])([\d,]+)", replace_currency, text)

    # --- REFERENCE NUMBERS: BK-2024-001 ---
    def replace_reference(m: re.Match) -> str:
        letters = _letters_to_spoken(m.group(1))
        digits = _digits_to_spoken(m.group(2) + m.group(3))
        return f"{letters} {digits}".strip()

    text = re.sub(r"\b([A-Za-z]{2,5})-(\d{2,})-(\d{2,})\b", replace_reference, text)

    # --- PHONE NUMBERS ---
    def replace_phone(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return _digits_to_spoken(digits)

    text = re.sub(r"\+?\d[\d\s-]{8,}\d", replace_phone, text)
    text = re.sub(r"\b(\d{10,13})\b", replace_phone, text)

    # --- ORDINALS ---
    def replace_ordinal(m: re.Match) -> str:
        value = int(m.group(1))
        return _number_to_ordinal_words(value)

    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", replace_ordinal, text, flags=re.IGNORECASE)

    # --- STANDALONE NUMBERS ---
    def replace_number(m: re.Match) -> str:
        raw = m.group(0).replace(",", "")
        try:
            value = int(raw)
        except ValueError:
            return m.group(0)
        return _number_to_words(value)

    text = re.sub(r"\b\d[\d,]*\b", replace_number, text)

    # --- TITLES & SYMBOLS ---
    text = re.sub(r"\bDr\.\s*", "Doctor ", text)
    text = re.sub(r"\bMr\.\s*", "Mister ", text)
    text = re.sub(r"\bMrs\.\s*", "Missus ", text)
    text = re.sub(r"\bMs\.\s*", "Miss ", text)
    text = re.sub(r"\bvs\b", "versus", text, flags=re.IGNORECASE)
    text = text.replace("&", "and")
    text = text.replace("%", " percent")

    return re.sub(r"\s+", " ", text).strip()


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("9:00 AM",                                               "nine am"),
        ("15:00",                                                 "three pm"),
        ("9.00.00",                                               "nine am"),
        ("14:30:00",                                              "two thirty pm"),
        ("2/05/2026",                                             "May second twenty six"),
        ("2026-05-28",                                            "May twenty eighth twenty six"),
        ("1st",                                                   "first"),
        ("15th",                                                  "fifteenth"),
        ("31st",                                                  "thirty first"),
        ("100",                                                   "one hundred"),
        ("Available slots are today at 10:00 or tomorrow at 9:00 AM",
                                                                  "Available slots are today at ten am or tomorrow at nine am"),
        ("Earliest slot is tomorrow at 9:00 AM",                  "Earliest slot is tomorrow at nine am"),
        ("Your appointment is on 2026-05-28 at 15:00",            "Your appointment is on May twenty eighth twenty six at three pm"),
        ("तुम्हारी अपॉइंटमेंट 12:00 बजे बुक हो गई है", "तुम्हारी अपॉइंटमेंट बारह बजे बुक हो गई है"),
    ]
    print(f"{'Input':<55} {'Expected':<45} {'Got':<45} Result")
    print("-" * 160)
    for inp, expected in tests:
        got = normalize_for_tts(inp)
        result = "PASS" if got == expected else "FAIL"
        print(f"{inp:<55} {expected:<45} {got:<45} {result}")
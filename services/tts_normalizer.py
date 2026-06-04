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


def normalize_for_tts(text: str) -> str:
    if not text:
        return ""

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
    ]
    print(f"{'Input':<55} {'Expected':<45} {'Got':<45} Result")
    print("-" * 160)
    for inp, expected in tests:
        got = normalize_for_tts(inp)
        result = "PASS" if got == expected else "FAIL"
        print(f"{inp:<55} {expected:<45} {got:<45} {result}")
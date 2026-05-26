import re

MONTHS = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

_ORDINAL_WORDS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
}

_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def _ordinal_suffix(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _date_to_words(day: int, month: int, year: int) -> str:
    suffix = _ordinal_suffix(day)
    return f"{day}{suffix} {MONTHS[month]} {year}"


def _hour_to_words(hour: int) -> str:
    if hour == 0:
        return "twelve"
    if hour > 12:
        hour -= 12
    words = [
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
    ]
    return words[hour]


def _time_to_words(hour: int, minute: int, period: str = "") -> str:
    if minute == 0:
        time_str = f"{_hour_to_words(hour)} o'clock"
    elif minute < 10:
        time_str = f"{_hour_to_words(hour)} oh {minute}"
    else:
        time_str = f"{_hour_to_words(hour)} {minute}"
    if period:
        time_str += f" {period.upper()}"
    return time_str


def _number_to_words(n: int) -> str:
    if n == 0:
        return "zero"
    if 1000 <= n < 2000:
        leading = n // 100
        remainder = n % 100
        base = f"{_number_to_words(leading)} hundred"
        if remainder:
            return f"{base} {_number_to_words(remainder)}"
        return base

    ones = [
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    tens = [
        "",
        "",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
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


def _digits_to_spoken(digits: str) -> str:
    return " ".join(_DIGIT_WORDS.get(d, d) for d in digits if d.isdigit())


def _letters_to_spoken(letters: str) -> str:
    return " ".join(letter.upper() for letter in letters if letter.isalpha())


def normalize_for_tts(text: str) -> str:
    if not text:
        return ""

    # Dates: DD/MM/YYYY or DD-MM-YYYY
    def replace_date_slash(m: re.Match) -> str:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return _date_to_words(day, month, year)
        return m.group(0)

    # Dates: YYYY-MM-DD
    def replace_date_iso(m: re.Match) -> str:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return _date_to_words(day, month, year)
        return m.group(0)

    text = re.sub(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", replace_date_slash, text)
    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", replace_date_iso, text)

    # Times: HH:MM AM/PM or HH:MM
    def replace_time(m: re.Match) -> str:
        hour, minute = int(m.group(1)), int(m.group(2))
        period = (m.group(3) or "").strip()
        if not period and hour >= 13:
            period = "PM"
            hour -= 12
        return _time_to_words(hour, minute, period)

    text = re.sub(r"\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\b", replace_time, text)

    # Currency: ₹ and $
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

    # Booking/reference numbers like BK-2024-001
    def replace_reference(m: re.Match) -> str:
        letters = _letters_to_spoken(m.group(1))
        digits = _digits_to_spoken(m.group(2) + m.group(3))
        return f"{letters} {digits}".strip()

    text = re.sub(r"\b([A-Za-z]{2,5})-(\d{2,})-(\d{2,})\b", replace_reference, text)

    # Phone numbers with optional + and separators
    def replace_phone(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return _digits_to_spoken(digits)

    text = re.sub(r"\+?\d[\d\s-]{8,}\d", replace_phone, text)
    text = re.sub(r"\b(\d{10,13})\b", replace_phone, text)

    # Standalone ordinals (not part of dates already normalized)
    def replace_ordinal(m: re.Match) -> str:
        value = int(m.group(1))
        word = _ORDINAL_WORDS.get(value)
        return word if word else m.group(0)

    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", replace_ordinal, text, flags=re.IGNORECASE)

    # Common symbols and titles
    text = re.sub(r"\bDr\.\s*", "Doctor ", text)
    text = re.sub(r"\bMr\.\s*", "Mister ", text)
    text = re.sub(r"\bMrs\.\s*", "Missus ", text)
    text = re.sub(r"\bMs\.\s*", "Miss ", text)
    text = re.sub(r"\bvs\b", "versus", text, flags=re.IGNORECASE)
    text = text.replace("&", "and")
    text = text.replace("%", " percent")

    return re.sub(r"\s+", " ", text).strip()

from datetime import datetime


def prepend_datetime(s: str, delimiter: str = "_") -> str:
    current_date = datetime.now().strftime("%Y-%m-%d")
    return f"{current_date}{delimiter}{s}"

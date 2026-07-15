import re

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def plain_output(text: str) -> str:
    """Strip Rich/terminal ANSI escape codes from CLI output for assertions."""
    return _ANSI_ESCAPE.sub("", text)
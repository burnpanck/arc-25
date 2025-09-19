try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum

try:
    from enum import property as enum_property
except ImportError:
    enum_property = property

try:
    from enum import nonmember as enum_nonmember
except ImportError:
    enum_nonmember = lambda x: x

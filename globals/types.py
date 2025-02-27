from dataclasses import dataclass

from numpy.ma import MaskedArray


@dataclass(slots=True)
class ParsedVariable:
    units: str
    values: MaskedArray

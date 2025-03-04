from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.ma import MaskedArray

type PrecipitationSeries = np.ndarray[tuple[datetime, float]]
type MetaData = dict[str, str | float]
type RecoveryData = dict[str, dict[str, PrecipitationSeries | MetaData]]


@dataclass(slots=True)
class ParsedVariable:
    units: str
    values: MaskedArray

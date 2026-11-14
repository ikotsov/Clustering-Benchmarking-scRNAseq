import os

from typing_extensions import Literal

Species = Literal["human", "mouse"]
NormMethod = Literal["pearson", "log_cpm"]
TuningObjective = Literal["external", "internal"]
PathLike = str | os.PathLike[str]

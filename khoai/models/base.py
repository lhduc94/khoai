import numpy as np
from typing import List, Dict, NamedTuple, Optional, Tuple


class ModelResult(NamedTuple):
    oof_preds: np.ndarray
    preds: Optional[np.ndarray]
    models: Dict[str, any]
    scores: Dict[str, float]
    folds: List[Tuple[np.ndarray, np.ndarray]]

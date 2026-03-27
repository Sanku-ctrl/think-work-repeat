from typing import Tuple
import numpy as np
from utils import TASK_TYPES, TIME_OF_DAY, encode_features

def generate_synthetic_dataset(num_samples: int = 220, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data for study-time prediction."""
    rng = np.random.default_rng(seed)

    task_base_minutes = {
        "reading": 30,
        "coding": 50,
        "revision": 25,
    }
    time_adjustment = {
        "morning": 0,
        "afternoon": 10,
        "night": 5,
    }

    features = []
    targets = []

    for _ in range(num_samples):
        task_type = rng.choice(TASK_TYPES)
        difficulty = int(rng.integers(1, 6))
        time_of_day = rng.choice(TIME_OF_DAY)

        estimated_minutes = (
            task_base_minutes[task_type]
            + difficulty * 12
            + time_adjustment[time_of_day]
            + rng.normal(0, 7)
        )

        estimated_minutes = float(np.clip(estimated_minutes, 20, 180))
        features.append(encode_features(task_type, difficulty, time_of_day))
        targets.append([estimated_minutes])

    x_data = np.array(features, dtype=np.float32)
    y_data = np.array(targets, dtype=np.float32)
    return x_data, y_data
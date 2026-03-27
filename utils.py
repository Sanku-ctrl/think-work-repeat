import numpy as np

TASK_TYPES = ["reading", "coding", "revision"]
TIME_OF_DAY = ["morning", "afternoon", "night"]

_TASK_TO_INDEX = {name: index for index, name in enumerate(TASK_TYPES)}
_TIME_TO_INDEX = {name: index for index, name in enumerate(TIME_OF_DAY)}

def _one_hot(index: int, size: int) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector

def encode_features(task_type: str, difficulty: int, time_of_day: str) -> np.ndarray:
    """Encode user inputs into a numeric feature vector for the model."""
    task_key = task_type.strip().lower()
    time_key = time_of_day.strip().lower()

    if task_key not in _TASK_TO_INDEX:
        raise ValueError(f"Unknown task type: {task_type}")
    if time_key not in _TIME_TO_INDEX:
        raise ValueError(f"Unknown time of day: {time_of_day}")

    difficulty_value = float(min(max(int(difficulty), 1), 5))

    task_vector = _one_hot(_TASK_TO_INDEX[task_key], len(TASK_TYPES))
    time_vector = _one_hot(_TIME_TO_INDEX[time_key], len(TIME_OF_DAY))
    difficulty_vector = np.array([difficulty_value / 5.0], dtype=np.float32)

    return np.concatenate([task_vector, difficulty_vector, time_vector]).astype(np.float32)
# Think, Work, Repeat - AI Study Session Assistant

Lightweight ML + productivity project built with PyTorch and Streamlit.

This app combines two focused features:
- Study time prediction with a small neural network
- A customizable Pomodoro timer for focus and break cycles

## Why This Project

The goal is to demonstrate a complete mini ML workflow in a practical app:
- Define input features and encode them
- Generate synthetic training data
- Train a PyTorch model end-to-end
- Serve predictions through an interactive UI
- Pair the prediction with a productivity loop (Pomodoro)

## Features

- Study duration prediction (in minutes)
- Input controls for task type, difficulty (1-5), and time of day
- Fast startup training on synthetic data (cached model)
- Pomodoro timer with:
   - Configurable focus duration (step: 5 minutes)
   - Configurable break duration (step: 1 minute)
   - Start/Pause, Reset Current, and manual session switch
   - Countdown display with progress bar

## Tech Stack

- Python
- PyTorch
- Streamlit
- NumPy
- Pandas (in dependencies for basic data handling)

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd think-work-repeat
```

2. Create and activate a virtual environment:

```bash
python -m venv twrVenv
# Windows
twrVenv\Scripts\activate
# macOS/Linux
source twrVenv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

## Project Structure

```text
think-work-repeat/
|-- app.py
|-- model.py
|-- data.py
|-- utils.py
|-- requirements.txt
`-- README.md
```

## How Prediction Works

### 1) Inputs and Feature Encoding

The model takes three user inputs:
- Task type: reading, coding, revision
- Difficulty: integer from 1 to 5
- Time of day: morning, afternoon, night

These are encoded into a 7-dimensional numeric vector:
- 3 values for task one-hot encoding
- 1 normalized difficulty value ($d / 5$)
- 3 values for time-of-day one-hot encoding

So each sample is:

$$
x \in \mathbb{R}^7
$$

Implemented in [utils.py](utils.py).

### 2) Synthetic Target Generation

Training labels are generated in [data.py](data.py) with this rule:

$$
y = \text{base(task)} + 12 \cdot \text{difficulty} + \text{timeAdjust(time)} + \epsilon
$$

Where:
- $\text{base(task)} \in \{25, 30, 50\}$ minutes depending on task
- $\text{timeAdjust(time)} \in \{0, 5, 10\}$ minutes
- $\epsilon \sim \mathcal{N}(0, 7)$ adds realistic noise
- Final $y$ is clipped to $[20, 180]$ minutes

This gives a simple but non-trivial regression dataset.

### 3) Neural Network Model (PyTorch)

Defined in [model.py](model.py) as a feedforward network:

$$
\hat{y} = W_3\,\sigma\left(W_2\,\sigma\left(W_1 x + b_1\right) + b_2\right) + b_3
$$

Where $\sigma$ is ReLU. Layer sizes:
- Input: 7
- Hidden 1: 16
- Hidden 2: 8
- Output: 1 (predicted minutes)

### 4) Loss Function and Optimization

The model is trained with Mean Squared Error:

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} \left(\hat{y}_i - y_i\right)^2
$$

Optimizer: Adam with learning rate 0.01, for 220 epochs.

### 5) Inference Flow in the App

In [app.py](app.py):
- Model and synthetic dataset are prepared once via Streamlit caching
- User input is encoded to feature vector
- PyTorch model predicts a scalar duration
- Output is rounded to integer minutes and shown in UI

## PyTorch Integration Details

- `StudyTimeModel(nn.Module)` encapsulates architecture
- Training uses tensor conversion from NumPy arrays
- Standard autograd loop:
   - Forward pass
   - MSE computation
   - `optimizer.zero_grad()`
   - `loss.backward()`
   - `optimizer.step()`
- Inference uses `@torch.no_grad()` and `model.eval()` for efficient prediction

## Pomodoro Logic Summary

Implemented in [app.py](app.py) using Streamlit session state:
- Maintains current mode (`focus` or `break`)
- Stores remaining seconds and session length
- Supports:
   - Start/Pause
   - Reset current mode
   - Manual switch between modes
- Progress bar reflects elapsed fraction of current session

## Limitations

- Dataset is synthetic (not real user history)
- Model is intentionally simple for fast training and demonstration
- No persistence of prediction or timer history across app restarts

## Future Improvements

- Train on real study logs
- Add model evaluation metrics (MAE/RMSE) in UI
- Save trained model weights
- Add personalized calibration per user
- Add analytics dashboard for completed focus sessions

## License

This project is provided as-is for educational purposes. Feel free to use, modify, and distribute.

## Author

**Sanket Motagi** | Created: March 2026  
GitHub: [@Sanku-ctrl](https://github.com/Sanku-ctrl)

## Contact & Contributing

Found a bug or have suggestions? Feel free to:
- Open an issue on GitHub
- Submit a pull request
- Contact via email

---

import time

import streamlit as st

from data import generate_synthetic_dataset
from model import predict_minutes, train_model
from utils import TASK_TYPES, TIME_OF_DAY, encode_features


def focus_seconds(minutes: int) -> int:
    return int(minutes) * 60


def break_seconds(minutes: int) -> int:
    return int(minutes) * 60


@st.cache_resource
def load_model():
    x_train, y_train = generate_synthetic_dataset(num_samples=220)
    return train_model(x_train, y_train)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        a[data-testid="stHeaderActionElements"],
        [data-testid="stHeaderActionElements"],
        [data-testid="stHeading"] a {
            display: none !important;
            visibility: hidden !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_seconds(total_seconds: int) -> str:
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def initialize_state() -> None:
    state = st.session_state

    if "focus_minutes" not in state:
        state.focus_minutes = 25
    if "break_minutes" not in state:
        state.break_minutes = 5

    if "timer_mode" not in state:
        state.timer_mode = "focus"
    if "timer_running" not in state:
        state.timer_running = False
    if "timer_seconds_left" not in state:
        state.timer_seconds_left = focus_seconds(state.focus_minutes)
    if "timer_end_time" not in state:
        state.timer_end_time = None
    if "timer_total_seconds" not in state:
        state.timer_total_seconds = focus_seconds(state.focus_minutes)

    if "predicted_minutes" not in state:
        state.predicted_minutes = None


def set_timer_mode(mode: str) -> None:
    st.session_state.timer_mode = mode
    st.session_state.timer_running = False
    if mode == "focus":
        seconds = focus_seconds(st.session_state.focus_minutes)
    else:
        seconds = break_seconds(st.session_state.break_minutes)

    st.session_state.timer_total_seconds = seconds
    st.session_state.timer_seconds_left = seconds
    st.session_state.timer_end_time = None


def render_prediction(model) -> None:
    st.subheader("Study Time Prediction")

    task_type = st.selectbox("Task type", TASK_TYPES, key="task_type_select")
    difficulty = st.slider("Difficulty", min_value=1, max_value=5, value=3, key="difficulty_slider")
    time_of_day = st.selectbox("Time of day", TIME_OF_DAY, key="time_of_day_select")

    if st.button("Predict Study Duration", key="predict_duration_btn"):
        features = encode_features(task_type, difficulty, time_of_day)
        st.session_state.predicted_minutes = predict_minutes(model, features)

    if st.session_state.predicted_minutes is not None:
        st.metric("Predicted study time", f"{st.session_state.predicted_minutes} minutes")


def render_pomodoro() -> None:
    st.subheader("Pomodoro Timer")

    current_mode = st.session_state.timer_mode
    is_running = st.session_state.timer_running

    settings_col1, settings_col2 = st.columns(2)
    with settings_col1:
        focus_value = st.number_input(
            "Focus session (minutes)",
            min_value=5,
            max_value=180,
            step=5,
            value=int(st.session_state.focus_minutes),
            disabled=is_running,
            key="focus_minutes_input",
        )
    with settings_col2:
        break_value = st.number_input(
            "Break session (minutes)",
            min_value=1,
            max_value=60,
            step=1,
            value=int(st.session_state.break_minutes),
            disabled=is_running,
            key="break_minutes_input",
        )

    if int(focus_value) != int(st.session_state.focus_minutes):
        st.session_state.focus_minutes = int(focus_value)
        if current_mode == "focus" and not is_running:
            set_timer_mode("focus")

    if int(break_value) != int(st.session_state.break_minutes):
        st.session_state.break_minutes = int(break_value)
        if current_mode == "break" and not is_running:
            set_timer_mode("break")

    timer_label = "Focus" if current_mode == "focus" else "Break"
    st.write(f"Current session: {timer_label}")

    timer_display = st.empty()
    progress_display = st.empty()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(
            "Start" if not is_running else "Pause",
            use_container_width=True,
            key="timer_start_pause_btn",
        ):
            if is_running:
                st.session_state.timer_running = False
                st.session_state.timer_end_time = None
            else:
                st.session_state.timer_running = True
                st.session_state.timer_end_time = time.time() + st.session_state.timer_seconds_left
            st.rerun()

    with col2:
        if st.button("Reset Current", use_container_width=True, key="timer_reset_btn"):
            set_timer_mode(current_mode)
            st.rerun()

    with col3:
        switch_label = "End Focus -> Break" if current_mode == "focus" else "End Break -> Focus"
        if st.button(switch_label, use_container_width=True, key="timer_switch_mode_btn"):
            next_mode = "break" if current_mode == "focus" else "focus"
            set_timer_mode(next_mode)
            st.rerun()

    if is_running and st.session_state.timer_end_time:
        while st.session_state.timer_running:
            remaining = max(0, int(st.session_state.timer_end_time - time.time()))
            st.session_state.timer_seconds_left = remaining

            with timer_display.container():
                st.metric("Time left", format_seconds(remaining))

            total_seconds = max(st.session_state.timer_total_seconds, 1)
            progress = 1 - (remaining / total_seconds)
            with progress_display.container():
                st.progress(min(max(progress, 0.0), 1.0))

            if remaining == 0:
                st.session_state.timer_running = False
                if st.session_state.timer_mode == "focus":
                    set_timer_mode("break")
                    st.success("Focus session complete. Break session is ready.")
                else:
                    set_timer_mode("focus")
                    st.success("Break session complete. Focus session is ready.")
                st.rerun()
                break

            time.sleep(1)
    else:
        with timer_display.container():
            st.metric("Time left", format_seconds(st.session_state.timer_seconds_left))

        total_seconds = max(st.session_state.timer_total_seconds, 1)
        progress = 1 - (st.session_state.timer_seconds_left / total_seconds)
        with progress_display.container():
            st.progress(min(max(progress, 0.0), 1.0))


def main() -> None:
    st.set_page_config(page_title="Think, Work, Repeat", layout="wide")
    inject_custom_css()
    st.title("Think, Work, Repeat - AI Study Session Assistant")

    initialize_state()
    model = load_model()

    render_prediction(model)

    st.divider()
    render_pomodoro()


if __name__ == "__main__":
    main()
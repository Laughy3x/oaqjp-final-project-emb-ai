"""Flask server for the emotion detection web application."""

from pathlib import Path
from typing import Any

from flask import Flask, render_template, request

from EmotionDetection import emotion_detector


BASE_DIR = Path(__file__).resolve().parent.parent
INVALID_TEXT_MESSAGE = "Invalid text! Please try again!"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)


def format_emotion_response(response: dict[str, Any]) -> str:
    """Build the response string required by the lab instructions."""
    return (
        "For the given statement, the system response is "
        f"'anger': {response['anger']}, "
        f"'disgust': {response['disgust']}, "
        f"'fear': {response['fear']}, "
        f"'joy': {response['joy']} and "
        f"'sadness': {response['sadness']}. "
        f"The dominant emotion is {response['dominant_emotion']}."
    )


@app.route("/")
def render_index_page() -> str:
    """Render the application's landing page."""
    return render_template("index.html")


@app.route("/emotionDetector")
def sent_analyzer() -> str:
    """Analyze the submitted text and return the formatted response string."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    response = emotion_detector(text_to_analyze)

    if "error" in response:
        return response["error"]
    if response["dominant_emotion"] is None:
        return INVALID_TEXT_MESSAGE

    return format_emotion_response(response)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)

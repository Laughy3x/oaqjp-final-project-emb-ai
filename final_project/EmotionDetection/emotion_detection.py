"""Emotion detection client for the packaged EmotionDetection module."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EMOTION_API_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
EMOTION_API_HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock",
    "Content-Type": "application/json",
}


def _empty_emotion_result() -> dict[str, Any]:
    """Return the blank-input response shape required by the lab."""
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }


def _build_payload(text_to_analyze: str) -> bytes:
    """Serialize the request payload expected by the Watson endpoint."""
    payload = {"raw_document": {"text": text_to_analyze}}
    return json.dumps(payload).encode("utf-8")


def _post_request(payload: bytes) -> str:
    """Send the request with the Python standard library."""
    request = Request(
        EMOTION_API_URL,
        data=payload,
        headers=EMOTION_API_HEADERS,
        method="POST",
    )
    with urlopen(request, timeout=8) as response:
        return response.read().decode("utf-8")


def _extract_emotions(response_text: str) -> dict[str, Any]:
    """Parse the Watson response and return the formatted emotion payload."""
    response_dict = json.loads(response_text)
    emotions = response_dict["emotionPredictions"][0]["emotion"]
    formatted_response = {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
    }
    formatted_response["dominant_emotion"] = max(
        formatted_response,
        key=formatted_response.get,
    )
    return formatted_response


def emotion_detector(text_to_analyze: str) -> dict[str, Any]:
    """Return the formatted emotion scores and the dominant emotion."""
    if not text_to_analyze or not text_to_analyze.strip():
        return _empty_emotion_result()

    payload = _build_payload(text_to_analyze)

    try:
        response_text = _post_request(payload)
        return _extract_emotions(response_text)
    except HTTPError as error:
        if error.code == 400:
            return _empty_emotion_result()
        return {"error": f"HTTP error: {error.code}"}
    except URLError as error:
        reason: Any = getattr(error, "reason", error)
        return {"error": f"URL error: {reason}"}
    except Exception as error:
        return {"error": f"Request failed: {error}"}

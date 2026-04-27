"""Unit tests for the EmotionDetection package."""

import json
import unittest
from unittest.mock import patch

from EmotionDetection import emotion_detector


TEST_CASES = [
    ("I am glad this happened", "joy"),
    ("I am really mad about this", "anger"),
    ("I feel disgusted just hearing about this", "disgust"),
    ("I am so sad about this", "sadness"),
    ("I am really afraid that this will happen", "fear"),
]


def _mock_response(dominant_emotion: str) -> str:
    """Build a fake Watson response with a chosen dominant emotion."""
    emotions = {
        "anger": 0.01,
        "disgust": 0.01,
        "fear": 0.01,
        "joy": 0.01,
        "sadness": 0.01,
    }
    emotions[dominant_emotion] = 0.91
    return json.dumps({"emotionPredictions": [{"emotion": emotions}]})


class EmotionDetectionTests(unittest.TestCase):
    """Validate dominant emotion detection for the required statements."""

    def test_required_statements(self) -> None:
        """Each required statement should map to the expected emotion."""
        for statement, expected_emotion in TEST_CASES:
            with self.subTest(statement=statement):
                with patch(
                    "EmotionDetection.emotion_detection._post_request",
                    return_value=_mock_response(expected_emotion),
                ):
                    response = emotion_detector(statement)
                self.assertEqual(response["dominant_emotion"], expected_emotion)


if __name__ == "__main__":
    unittest.main()

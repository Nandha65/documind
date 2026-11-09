import pytest
from google.genai import errors as genai_errors

from src.errors import friendly_error


class StatusError(Exception):
    """Stands in for OpenAI/Anthropic errors, which carry .status_code."""

    def __init__(self, status_code, message="error"):
        super().__init__(message)
        self.status_code = status_code


GEMINI_OVERLOADED = genai_errors.ServerError(503, {"error": {
    "code": 503, "status": "UNAVAILABLE",
    "message": "This model is currently experiencing high demand.",
}})


GEMINI_DAILY_QUOTA = genai_errors.ClientError(429, {"error": {
    "code": 429, "status": "RESOURCE_EXHAUSTED",
    "message": "You exceeded your current quota. Quota exceeded for metric: "
               "generate_content_free_tier_requests, limit: 20. Generate content requests "
               "per day limit exceeded on Free Tier.",
}})


@pytest.mark.parametrize("error, expected", [
    (GEMINI_OVERLOADED, "Gemini is overloaded right now"),
    (GEMINI_DAILY_QUOTA, "Gemini's daily quota is used up"),
    (StatusError(529, "Overloaded"), "Gemini is overloaded right now"),
    (StatusError(429, "Too many requests"), "rate limit reached"),
    (StatusError(401, "invalid x-api-key"), "rejected the API key"),
    (ConnectionError("connection reset by peer"), "Couldn't reach Gemini"),
    (StatusError(500, "internal error"), "server error (500)"),
    (ValueError("Something odd\nsecond line"), "Couldn't get an answer from Gemini: Something odd"),
])
def test_friendly_error(error, expected):
    assert expected in friendly_error("Gemini", error)

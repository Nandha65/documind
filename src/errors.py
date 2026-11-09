"""Turn provider API errors into short messages for the chat UI."""


def friendly_error(provider: str, error: Exception) -> str:
    # Gemini errors carry the HTTP status in .code; OpenAI and Anthropic in .status_code
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    text = str(error)
    name = type(error).__name__

    if status in (503, 529) or "overloaded" in text.lower() or "UNAVAILABLE" in text:
        return f"{provider} is overloaded right now. Please try again in a moment."
    if status == 429 or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower():
        return f"{provider} rate limit reached. Wait a minute and try again."
    if status in (401, 403):
        return f"{provider} rejected the API key. Check the key in your .env file."
    if "Connection" in name or "Timeout" in name:
        return f"Couldn't reach {provider}. Check your internet connection and try again."
    if isinstance(status, int) and status >= 500:
        return f"{provider} had a server error ({status}). Please try again."

    first_line = text.strip().splitlines()[0] if text.strip() else name
    return f"Couldn't get an answer from {provider}: {first_line[:200]}"

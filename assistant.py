"""
assistant.py
-------------
A lightweight, rule-based FAQ assistant for the CrimeVision-AI dashboard.

This works fully offline (no API key needed) and answers common questions
about the system's architecture, tech stack, and limitations — useful for
live demos, interviews, or viva, where someone can "ask the dashboard"
how it works instead of you explaining everything manually.

WANT A REAL LLM INSTEAD OF RULE-BASED MATCHING?
If you get an Anthropic or OpenAI API key later, swap `get_answer()`'s
body for an actual API call (see the commented example at the bottom of
this file). The rest of the dashboard code doesn't need to change at all
since it just calls `get_answer(question)`.
"""

import re

# ---- Knowledge base: (trigger keywords, answer) ----
# Matching is simple keyword-based, checked in order — first match wins.
KNOWLEDGE_BASE = [
    (
        ["how does this work", "how it works", "pipeline", "architecture", "steps"],
        "CrimeVision-AI works in stages: 1) A witness's spoken description is "
        "recorded and transcribed using OpenAI's Whisper (multilingual speech "
        "recognition). 2) If needed, the text is translated to English. "
        "3) The description is converted into a structured forensic-style prompt. "
        "4) Stable Diffusion generates a facial composite from that prompt. "
        "5) The composite's facial embedding is compared against a case database "
        "using cosine similarity to surface any potentially related records."
    ),
    (
        ["real database", "real police", "actual data", "connected to police", "real records"],
        "No — this system uses a fully simulated, synthetic case database built "
        "for demonstration purposes. It is not connected to any real "
        "law-enforcement records system. The architecture is designed so a real "
        "records API could be substituted with no change to the matching logic."
    ),
    (
        ["accurate", "accuracy", "how good", "reliable", "trust"],
        "Face-matching systems like this one have well-documented accuracy and "
        "fairness limitations — performance can vary across demographics, "
        "lighting conditions, and image quality. This is a proof-of-concept "
        "prototype for system design, not a production-ready identification "
        "tool, and its outputs should never be treated as proof of identity."
    ),
    (
        ["tech stack", "technology", "built with", "what did you use", "libraries"],
        "The stack: Whisper (speech-to-text), Deep Translator (multilingual "
        "translation), Stable Diffusion v1.5 via Hugging Face Diffusers "
        "(image generation), DeepFace with a RetinaFace detector and Facenet "
        "embeddings (face similarity matching), and Streamlit with Plotly for "
        "this dashboard."
    ),
    (
        ["multilingual", "languages", "hindi", "punjabi", "which language"],
        "Whisper supports multilingual speech recognition, and the system "
        "currently handles English, Hindi, and Punjabi, with automatic "
        "translation to English before the composite-generation stage."
    ),
    (
        ["why generative ai", "why stable diffusion", "why this model"],
        "Stable Diffusion was chosen because it can generate an image directly "
        "from a structured text description without needing a training dataset "
        "of labeled sketches — which doesn't exist for this use case. It runs "
        "locally, which matters for a system handling sensitive descriptions."
    ),
    (
        ["limitation", "problem", "weakness", "concern", "risk"],
        "Key limitations: (1) the case database is simulated, not real; "
        "(2) face-matching accuracy varies across demographics and image "
        "quality; (3) a generated composite is an AI-assisted approximation "
        "based on a witness's verbal description, not a verified photograph; "
        "(4) Stable Diffusion is computationally intensive on CPU, so this "
        "isn't advertised as real-time without benchmarking."
    ),
    (
        ["who built", "who made", "developer", "author"],
        "This system was independently designed and built by Sanchita Kumari, "
        "including the full pipeline integration: audio recording, Whisper "
        "transcription, multilingual translation, forensic prompt "
        "construction, Stable Diffusion image generation, the face-similarity "
        "case-correlation module, and this dashboard."
    ),
]

FALLBACK_ANSWER = (
    "I don't have a specific answer for that — try asking about how the "
    "pipeline works, the tech stack used, accuracy limitations, or whether "
    "this connects to a real police database."
)


def get_answer(question: str) -> str:
    """Return the best matching answer for a user's question."""
    question_lower = question.lower()

    for keywords, answer in KNOWLEDGE_BASE:
        if any(kw in question_lower for kw in keywords):
            return answer

    return FALLBACK_ANSWER


def get_suggested_questions():
    """Sample questions to show as clickable buttons in the UI."""
    return [
        "How does this system work?",
        "Is this connected to a real police database?",
        "What tech stack was used?",
        "What are the limitations of this system?",
        "Why did you use Stable Diffusion?",
    ]


# ---------------------------------------------------------------------------
# OPTIONAL UPGRADE: real LLM instead of rule-based matching
# ---------------------------------------------------------------------------
# If you get an API key, replace get_answer()'s body with something like:
#
# import anthropic
# client = anthropic.Anthropic(api_key="YOUR_KEY_HERE")
#
# PROJECT_CONTEXT = """
# You are an assistant embedded in the CrimeVision-AI dashboard, a forensic
# facial composite system. Answer questions about its architecture (Whisper
# speech-to-text, Deep Translator, Stable Diffusion image generation,
# DeepFace similarity matching), its use of a SIMULATED case database (not
# real police data), and its known limitations (demographic accuracy
# variation, composites are not verified photographs). Keep answers concise.
# """
#
# def get_answer(question: str) -> str:
#     response = client.messages.create(
#         model="claude-sonnet-4-5",
#         max_tokens=300,
#         system=PROJECT_CONTEXT,
#         messages=[{"role": "user", "content": question}],
#     )
#     return response.content[0].text
#
# Never commit a real API key to GitHub — use an environment variable
# (os.environ["ANTHROPIC_API_KEY"]) instead of hardcoding it.
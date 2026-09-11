def improve_communication(text):
    """
    Helps make a candidate's message clearer.
    It does NOT create new qualifications or experience.
    """

    if not text or not text.strip():
        return ""

    text = text.strip()

    # Simple communication improvements
    replacements = {
        "i am student": "I am a student.",
        "i know python": "I have knowledge of Python.",
        "i know java": "I have knowledge of Java.",
        "i did project": "I completed a project.",
        "i am interested ai": "I am interested in Artificial Intelligence.",
        "thank you": "Thank you for the opportunity."
    }

    lower_text = text.lower()

    if lower_text in replacements:
        return replacements[lower_text]

    # Basic sentence formatting
    if not text.endswith((".", "!", "?")):
        text += "."

    return text


def simplify_question(text):
    """
    Makes an interviewer's question easier to understand.
    """

    if not text or not text.strip():
        return ""

    text = text.strip()

    replacements = {
        "tell me about yourself":
            "Please introduce yourself and talk briefly about your education, skills, and interests.",

        "why should we hire you":
            "Why do you think you are suitable for this job?",

        "what are your strengths":
            "What are you good at?",

        "what are your weaknesses":
            "What is one area you are working to improve?",

        "tell me about your project":
            "Please explain your project, what you built, and what you learned."
    }

    lower_text = text.lower()

    return replacements.get(
        lower_text,
        text
    )
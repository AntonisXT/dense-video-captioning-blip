# utils/text_utils.py
import re


def clean_caption(text):
    """
    Clean and normalize a caption or summary string by:
    - Removing repeated words or phrases
    - Removing low-information patterns like "screenshot of"
    - Trimming noisy punctuation and formatting
    """
    if not text or not isinstance(text, str):
        return ""

    # Only keep the part before the first period
    text = text.split(".")[0].strip()

    # Remove repeated single words
    words = text.strip().split()
    cleaned_words = []
    prev_word = None
    for word in words:
        if word != prev_word:
            cleaned_words.append(word)
        prev_word = word
    caption = " ".join(cleaned_words)

    # Remove repeated bigrams
    caption = re.sub(r'\b(\w+ \w+)( \1\b)+', r'\1', caption)

    # Remove phrases like "a screenshot of"
    caption = re.sub(r'\ba\s+(video|screenshot|screenshots|screenshote|screen shot)\s+of\b', '', caption)

    # Remove individual blacklist words
    blacklist = ["screenshot", "screenshots", "screenshote", "screen shot"]
    for word in blacklist:
        caption = caption.replace(word, "")

    # Clean phrases around dashes
    caption = re.sub(r'^\b\w+\b\s*-\s*', '', caption)
    caption = re.sub(r'-\s*\b\w{1,2}\b$', '', caption)
    caption = re.sub(r'\b([\w\s]{2,20}?)\b(?:\s*-\s*\1\b)+', r'\1', caption)

    # Remove repeated words (e.g., "word word word")
    caption = re.sub(r'\b(\w+)( \1\b)+', r'\1', caption)

    # Normalize spacing and punctuation
    caption = re.sub(r'\s+', ' ', caption)
    caption = caption.strip(" .,-")

    return caption


def fix_overlapping_scenes(scenes):
    """
    Ensure that the output scenes have no time overlaps and
    that all scenes have a minimum duration.

    Parameters:
        scenes (list of dict): each scene must have 'start_time' and 'end_time'

    Returns:
        list of dict: fixed scene list
    """
    fixed = []
    prev_end = -1
    for scene in sorted(scenes, key=lambda x: x["start_time"]):
        if scene["start_time"] < prev_end:
            scene["start_time"] = prev_end
        if scene["end_time"] <= scene["start_time"]:
            scene["end_time"] = scene["start_time"] + 1
        fixed.append(scene)
        prev_end = scene["end_time"]
    return fixed
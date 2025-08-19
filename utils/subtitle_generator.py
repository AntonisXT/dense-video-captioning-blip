# utils/subtitle_generator.py

def seconds_to_timestamp(seconds):
    """
    Convert integer seconds to SRT-compliant timestamp string.

    Parameters:
        seconds (int): number of seconds

    Returns:
        str: formatted time in HH:MM:SS,000
    """
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02},000"


def generate_srt(scene_captions, output_path):
    """
    Generate an SRT subtitle file from timestamped scene captions.

    Parameters:
        scene_captions (list of dict): each with start_time, end_time, caption
        output_path (str): path to output .srt file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, scene in enumerate(scene_captions, 1):
            start = seconds_to_timestamp(scene["start_time"])
            end = seconds_to_timestamp(scene["end_time"])
            f.write(f"{idx}\n{start} --> {end}\n{scene['caption']}\n\n")
    print(f"✅ SRT subtitles saved to: {output_path}")

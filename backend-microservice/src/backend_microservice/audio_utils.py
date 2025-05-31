from pydub import AudioSegment


def get_audio_duration(file_path: str) -> float:
    """
    Get the duration of an audio file in seconds.

    :param file_path: Path to the audio file.
    :return: Duration of the audio file in seconds.
    """
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0  # Convert milliseconds to seconds/8.0

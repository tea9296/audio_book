import base64
import wave
import os
import ormsgpack
import pyaudio
import requests
from pydub import AudioSegment
from pydub.playback import play
from helper.commons import ServeReferenceAudio, ServeTTSRequest, load_file, split_text
from helper.file import audio_to_bytes, read_ref_text
import io


def generate_audio(
    url: str = "http://127.0.0.1:8080/v1/tts",
    text: str = "",
    reference_id: str = None,
    reference_audio: list[str] = None,
    reference_text: list[str] = None,
    output: str = "generated_audio",
    play_audio: bool = True,
    normalize: bool = True,
    audio_format: str = "wav",
    mp3_bitrate: int = 64,
    opus_bitrate: int = -1000,
    latency: str = "normal",
    max_new_tokens: int = 1024,
    chunk_length: int = 100,
    top_p: float = 0.7,
    repetition_penalty: float = 1.2,
    temperature: float = 0.7,
    speaker: str = None,
    emotion: str = None,
    streaming: bool = False,
    channels: int = 1,
    rate: int = 44100,
):
    idstr: str | None = reference_id
    # priority: ref_id > [{text, audio},...]
    if idstr is None:
        ref_audios = reference_audio
        ref_texts = reference_text
        if ref_audios is None:
            byte_audios = []
        else:
            byte_audios = [
                audio_to_bytes(ref_audio) for ref_audio in ref_audios
            ]
        if ref_texts is None:
            ref_texts = []
        else:
            ref_texts = [read_ref_text(ref_text) for ref_text in ref_texts]
    else:
        byte_audios = []
        ref_texts = []
        pass  # in api.py
    data = {
        "text":
        text,
        "references": [
            ServeReferenceAudio(audio=ref_audio, text=ref_text)
            for ref_text, ref_audio in zip(ref_texts, byte_audios)
        ],
        "reference_id":
        idstr,
        "normalize":
        normalize,
        "format":
        audio_format,
        "mp3_bitrate":
        mp3_bitrate,
        "opus_bitrate":
        opus_bitrate,
        "max_new_tokens":
        max_new_tokens,
        "chunk_length":
        chunk_length,
        "top_p":
        top_p,
        "repetition_penalty":
        repetition_penalty,
        "temperature":
        temperature,
        "speaker":
        speaker,
        "emotion":
        emotion,
        "streaming":
        streaming,
    }

    pydantic_data = ServeTTSRequest(**data)

    response = requests.post(
        url,
        data=ormsgpack.packb(pydantic_data,
                             option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        stream=streaming,
        headers={
            "authorization": "Bearer YOUR_API_KEY",
            "content-type": "application/msgpack",
        },
    )

    if response.status_code == 200:
        if streaming:
            p = pyaudio.PyAudio()
            audio_format = pyaudio.paInt16  # Assuming 16-bit PCM format
            stream = p.open(format=audio_format,
                            channels=channels,
                            rate=rate,
                            output=True)
            os.makedirs(os.path.dirname(output + ".wav"), exist_ok=True)
            wf = wave.open(f"{output}.wav", "wb")
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(audio_format))
            wf.setframerate(rate)

            stream_stopped_flag = False

            try:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        stream.write(chunk)
                        wf.writeframesraw(chunk)
                    else:
                        if not stream_stopped_flag:
                            stream.stop_stream()
                            stream_stopped_flag = True
            finally:
                stream.close()
                p.terminate()
                wf.close()
        else:
            audio_content = response.content
            audio_path = f"{output}.{audio_format}"
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)

            with open(audio_path, "wb") as audio_file:
                audio_file.write(audio_content)

            audio = AudioSegment.from_file(audio_path, format=audio_format)
            if play_audio:
                play(audio)
            print(f"Audio has been saved to '{audio_path}'.")
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.json())


# Example usage:
# synthesize_audio(text="Hello world", output="output_audio")


def generate_audio_segment(
    url: str,
    text: str,
    reference_id: str = None,
    reference_audio: list[str] = None,
    reference_text: list[str] = None,
    normalize: bool = True,
    audio_format: str = "wav",
    mp3_bitrate: int = 64,
    opus_bitrate: int = -1000,
    max_new_tokens: int = 1024,
    chunk_length: int = 100,
    top_p: float = 0.7,
    repetition_penalty: float = 1.2,
    temperature: float = 0.7,
    speaker: str = None,
    emotion: str = None,
    streaming: bool = False,
    channels: int = 1,
    rate: int = 44100,
) -> AudioSegment:
    idstr: str | None = reference_id
    if idstr is None:
        ref_audios = reference_audio
        ref_texts = reference_text
        if ref_audios is None:
            byte_audios = []
        else:
            byte_audios = [
                audio_to_bytes(ref_audio) for ref_audio in ref_audios
            ]
        if ref_texts is None:
            ref_texts = []
        else:
            ref_texts = [read_ref_text(ref_text) for ref_text in ref_texts]
    else:
        byte_audios = []
        ref_texts = []
        pass  # in api.py
    data = {
        "text":
        text,
        "references": [
            ServeReferenceAudio(audio=ref_audio, text=ref_text)
            for ref_text, ref_audio in zip(ref_texts, byte_audios)
        ],
        "reference_id":
        idstr,
        "normalize":
        normalize,
        "format":
        audio_format,
        "mp3_bitrate":
        mp3_bitrate,
        "opus_bitrate":
        opus_bitrate,
        "max_new_tokens":
        max_new_tokens,
        "chunk_length":
        chunk_length,
        "top_p":
        top_p,
        "repetition_penalty":
        repetition_penalty,
        "temperature":
        temperature,
        "speaker":
        speaker,
        "emotion":
        emotion,
        "streaming":
        streaming,
    }

    pydantic_data = ServeTTSRequest(**data)

    response = requests.post(
        url,
        data=ormsgpack.packb(pydantic_data,
                             option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        stream=streaming,
        headers={
            "authorization": "Bearer YOUR_API_KEY",
            "content-type": "application/msgpack",
        },
    )

    if response.status_code == 200:
        audio_content = response.content
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_content),
                                               format=audio_format)
        return audio_segment
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.json())
        return None


def generate_book(book_path: str,
                  output_path: str = "./book1.wav",
                  reference_audio: list[str] = None,
                  reference_text: list[str] = None,
                  temperature: float = 0.7,
                  url: str = "http://127.0.0.1:8080/v1/tts"):
    texts = load_file(book_path)
    text_list = split_text(texts)

    audio_segments = []
    for text in text_list:
        audio_segment = generate_audio_segment(
            url=url,
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            temperature=temperature,
        )
        if audio_segment:
            audio_segments.append(audio_segment)

    combined_audio = sum(audio_segments)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_audio.export(output_path, format="wav")
    print(f"Book audio has been saved to '{output_path}'.")

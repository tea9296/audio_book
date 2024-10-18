from typing import Annotated, Literal, Optional
import traceback, logging
from pydantic import BaseModel, Field, conint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "wav"
    mp3_bitrate: Literal[64, 128, 192] = 128
    # References audios for in-context learning
    references: list[ServeReferenceAudio] = []
    # Reference id
    # For example, if you want use https://fish.audio/m/7f92f8afb8ec43bf81429cc1c9199cb1/
    # Just pass 7f92f8afb8ec43bf81429cc1c9199cb1
    reference_id: str | None = None
    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True
    mp3_bitrate: Optional[int] = 64
    opus_bitrate: Optional[int] = -1000
    # Balance mode will reduce latency to 300ms, but may decrease stability
    latency: Literal["normal", "balanced"] = "normal"
    # not usually used below
    streaming: bool = False
    emotion: Optional[str] = None
    max_new_tokens: int = 1024
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7
    repetition_penalty: Annotated[float,
                                  Field(ge=0.9, le=2.0, strict=True)] = 1.2
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7


def load_file(file_path: str):
    """get the content and metadata of a text file (.pdf, .docx, .txt) and return a Document object

    Args:
        file_path (str): the path of the text file\n
        

    Raises:
        Exception: if the length of doc is 0, raise error

    Returns:
        list: list of Docuemnts
    """

    extention = file_path.split(".")[-1]

    try:
        if extention == "pdf" or extention == "PDF":
            docs = PyPDFLoader(file_path).load()
        elif extention == "docx" or extention == "DOCX":
            docs = Docx2txtLoader(file_path).load()
        else:
            docs = TextLoader(file_path, encoding="utf-8").load()
            for i in range(len(docs)):
                docs[i].metadata["page"] = i

        if len(docs) == 0:
            raise Exception

        ret = '\n'.join(doc.page_content for doc in docs)

        return ret
    except Exception as err:
        try:
            trace_text = traceback.format_exc()

            logging.warning("\nLoad " + file_path + " failed, ignored.\n" +
                            trace_text + "\n\n" + str(err))
        except:
            logging.warning("\nLoad file" + " failed, ignored.\n" +
                            trace_text + "\n\n")
        return ""


def split_text(
        text:
    str = "Just a weekly reminder for those who wish to contribute to the project I'm working on, love you!",
        lang: str = "en",
        max_len: int = 300) -> list[str]:
    """Split a text into multiple parts with a maximum length

    Args:
        text (str): the text to be splitted
        lang (str): the language of the text
        max_len (int, optional): the maximum length of each part. Defaults to 300.

    Returns:
        list[str]: the splitted text
    """

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", ",", ".", "。", "!", "?"],
        chunk_size=max_len,
        chunk_overlap=0,
    )

    ret = text_splitter.split_text(text)

    return ret

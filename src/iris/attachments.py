"""Files the user attaches to a typed message.

An attachment lives only for the conversation it was shared in: its text or
image is handed to the model for a few turns and never written to long-term
memory, which keeps only the file's name.
"""

import base64
import binascii
import io
from dataclasses import dataclass
from pathlib import PurePath
from typing import Optional

from PIL import Image, ImageOps, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_BYTES = 25 * 1024 * 1024
MAX_FILES = 10
# Longest side of an image as the model gets it; vision models scale down
# far below this anyway, so larger ones only cost time
MODEL_IMAGE_SIDE = 1568
# Longest side of the copy shown in the conversation
PREVIEW_SIDE = 640


class AttachmentError(ValueError):
    """An attachment that cannot be read; the message says why."""


@dataclass
class Attachment:
    name: str
    kind: str  # "image" or "document"
    mime: str
    # What the model gets: a document's text, or a base64 JPEG
    text: Optional[str] = None
    image: Optional[str] = None
    # A small JPEG data URL for the conversation view (images only)
    preview: Optional[str] = None

    def describe(self) -> dict:
        """The attachment as the UI shows it, without its contents."""
        info = {"name": self.name, "kind": self.kind, "mime": self.mime}
        if self.preview:
            info["preview"] = self.preview
        return info


def _jpeg(image: Image.Image, side: int, quality: int) -> bytes:
    copy = image.copy()
    copy.thumbnail((side, side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _read_image(name: str, mime: str, data: bytes) -> Attachment:
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
    except (UnidentifiedImageError, OSError) as e:
        raise AttachmentError(f"{name} is not an image iris can open") from e
    # Transparent parts become white rather than black
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        backdrop = Image.new("RGB", image.size, (255, 255, 255))
        backdrop.paste(image, mask=image.getchannel("A"))
        image = backdrop
    else:
        image = image.convert("RGB")
    preview = base64.b64encode(_jpeg(image, PREVIEW_SIDE, 82)).decode("ascii")
    return Attachment(
        name=name,
        kind="image",
        mime=mime,
        image=base64.b64encode(_jpeg(image, MODEL_IMAGE_SIDE, 90)).decode("ascii"),
        preview=f"data:image/jpeg;base64,{preview}",
    )


def _read_pdf(name: str, data: bytes) -> Attachment:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise AttachmentError(f"{name} is password protected")
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as e:
        raise AttachmentError(f"{name} could not be read as a PDF") from e
    text = "\n\n".join(
        f"[Page {number}]\n{page.strip()}" for number, page in enumerate(pages, 1) if page.strip()
    )
    if not text:
        # A scan: pages of pictures with no text layer
        text = "(This PDF has no text that can be extracted; it may be a scan.)"
    return Attachment(name=name, kind="document", mime="application/pdf", text=text)


def _as_text(data: bytes) -> Optional[str]:
    """The bytes as text, or None when they look binary."""
    if b"\x00" in data[:8192]:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def prepare(raw: dict) -> Attachment:
    """Turn an upload ({name, mime, data: base64}) into an Attachment."""
    name = PurePath(str(raw.get("name") or "file")).name
    mime = str(raw.get("mime") or "").lower()
    try:
        data = base64.b64decode(raw.get("data") or "", validate=True)
    except (binascii.Error, ValueError) as e:
        raise AttachmentError(f"{name} arrived damaged") from e
    if not data:
        raise AttachmentError(f"{name} is empty")
    if len(data) > MAX_BYTES:
        raise AttachmentError(f"{name} is larger than {MAX_BYTES // (1024 * 1024)} MB")

    suffix = PurePath(name).suffix.lower()
    if mime.startswith("image/"):
        return _read_image(name, mime, data)
    if mime == "application/pdf" or suffix == ".pdf":
        return _read_pdf(name, data)
    # Anything else is read as text when it is text: code, notes, CSV, JSON
    text = _as_text(data)
    if text is not None:
        return Attachment(name=name, kind="document", mime=mime or "text/plain", text=text)
    raise AttachmentError(f"iris cannot read {name}: only images, PDFs and text files")


def prepare_all(raws: list) -> list[Attachment]:
    if len(raws) > MAX_FILES:
        raise AttachmentError(f"At most {MAX_FILES} files can go with one message")
    return [prepare(raw) for raw in raws]


def recorded_text(prompt: str, attachments: list[Attachment]) -> str:
    """The message as it is remembered: the typed text and the file names."""
    if not attachments:
        return prompt
    names = ", ".join(a.name for a in attachments)
    return f"{prompt}\n\n(Attached: {names})" if prompt else f"(Attached: {names})"

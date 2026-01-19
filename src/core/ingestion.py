import os
import re
from typing import List, Dict, Optional

import fitz  # PyMuPDF
from tqdm import tqdm


class SimpleTextSplitter:
    """
    A simple recursive-like splitter based on progressively smaller separators.
    Note: chunk_size/chunk_overlap are character-based (not token-based).
    """

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 250,
        separators: Optional[List[str]] = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using progressively smaller separators."""
        if not text:
            return []

        pieces = [text]
        for sep in self.separators:
            new_pieces = []

            for piece in pieces:
                if len(piece) <= self.chunk_size:
                    new_pieces.append(piece)
                    continue

                # Split this piece
                if sep == "":
                    # Character split with overlap
                    step = max(1, self.chunk_size - self.chunk_overlap)
                    for i in range(0, len(piece), step):
                        new_pieces.append(piece[i : i + self.chunk_size])
                else:
                    sub_pieces = piece.split(sep)
                    current_chunk = ""

                    for sub in sub_pieces:
                        candidate_len = (
                            len(current_chunk)
                            + (len(sep) if current_chunk else 0)
                            + len(sub)
                        )
                        if candidate_len < self.chunk_size:
                            current_chunk += (sep if current_chunk else "") + sub
                        else:
                            if current_chunk:
                                new_pieces.append(current_chunk)
                            current_chunk = sub

                    if current_chunk:
                        new_pieces.append(current_chunk)

            pieces = new_pieces

        # Clean up (remove empty/whitespace-only)
        return [p.strip() for p in pieces if p and p.strip()]


def normalize_pdf_text(text: str) -> str:
    """
    Normalize typical PDF-extracted text:
    - Fix hyphenation across line breaks: "exam-\\nple" -> "example"
    - Convert single newlines to spaces, preserve paragraph breaks (double newlines)
    - Collapse excessive blank lines/spaces
    """
    if not text:
        return ""

    # Fix hyphenation across line breaks
    text = re.sub(r"-\n(?=\w)", "", text)

    # Convert single newlines to spaces, keep paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Collapse 3+ newlines to double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# Sentence end (no dot). Splits after ! ? Armenian full stop (։)
# Also allows closing quotes/brackets right after punctuation, then whitespace OR end of text.
SENT_END_RE = re.compile(
    r"""
    [!?։]
    (?:["”’')\]]*)     # optional closing quotes/brackets right after
    (?:\s+|$)          # whitespace OR end of text
    """,
    re.VERBOSE,
)


def split_sentences_no_dot(text: str) -> List[str]:
    """
    Split text into sentences based on ! ? । (Armenian full stop '։'),
    without treating '.' as a sentence boundary.
    """
    if not text:
        return []

    out: List[str] = []
    start = 0

    for m in SENT_END_RE.finditer(text):
        end = m.end()

        # trim trailing whitespace from the slice end
        end_no_ws = end
        while end_no_ws > start and text[end_no_ws - 1].isspace():
            end_no_ws -= 1

        s = text[start:end_no_ws].strip()
        if s:
            out.append(s)

        start = end

    tail = text[start:].strip()
    if tail:
        out.append(tail)

    return out


def semantic_chunk_sentences(
    sentences: List[str],
    chunk_size: int = 3000,
    chunk_overlap: int = 500,
) -> List[str]:
    """
    Create chunks by concatenating whole sentences up to chunk_size characters.
    Applies overlap by carrying over the last sentences whose combined length
    is up to chunk_overlap characters.

    Note: This is character-based overlap (better than none), but token-based
    chunking is ideal if you can add a tokenizer.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0  # approx in characters including spaces between sentences

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return

        chunk = " ".join(cur).strip()
        if chunk:
            chunks.append(chunk)

        if chunk_overlap > 0:
            overlap_sents: List[str] = []
            overlap_len = 0
            # accumulate from the end until we hit overlap budget
            for s in reversed(cur):
                add_len = len(s) + (1 if overlap_sents else 0)
                if overlap_len + add_len > chunk_overlap:
                    break
                overlap_sents.append(s)
                overlap_len += add_len
            overlap_sents.reverse()

            cur = overlap_sents
            cur_len = sum(len(s) for s in cur) + max(0, len(cur) - 1)
        else:
            cur = []
            cur_len = 0

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        add_len = len(s) + (1 if cur else 0)
        if cur_len + add_len <= chunk_size:
            cur.append(s)
            cur_len += add_len
            continue

        # flush current chunk, then handle this sentence
        flush()

        # If a single sentence is too long, hard-split it with overlap
        if len(s) > chunk_size:
            step = max(1, chunk_size - chunk_overlap)
            for i in range(0, len(s), step):
                part = s[i : i + chunk_size].strip()
                if part:
                    chunks.append(part)
            cur = []
            cur_len = 0
        else:
            cur = [s]
            cur_len = len(s)

    flush()
    return chunks


# --- NEW: Article-based splitting on "հոդված" (case-insensitive) ---

# Matches "հոդված" as a whole word (unicode-aware), case-insensitive.
# We keep the delimiter by splitting with a capturing group.
ARTICLE_SPLIT_RE = re.compile(r"(?i)(\bհոդված\b)")


def split_by_article_word(text: str) -> List[str]:
    """
    Split a normalized text into sections starting at each occurrence of the word "հոդված"
    (case-insensitive). The returned chunks include the "հոդված" token at the start of each
    section (except possibly a preamble chunk before the first article).

    Example:
      "Նախաբան ... Հոդված 1 ... Հոդված 2 ..." ->
      ["Նախաբան ...", "Հոդված 1 ...", "Հոդված 2 ..."]
    """
    if not text:
        return []

    parts = ARTICLE_SPLIT_RE.split(text)
    if not parts:
        return []

    chunks: List[str] = []

    # parts pattern: [preamble, 'հոդված', after1, 'հոդված', after2, ...]
    preamble = (parts[0] or "").strip()
    if preamble:
        chunks.append(preamble)

    i = 1
    while i < len(parts):
        token = parts[i]  # 'հոդված' (original casing as found)
        after = parts[i + 1] if i + 1 < len(parts) else ""
        section = f"{token}{after}".strip()
        if section:
            chunks.append(section)
        i += 2

    return chunks


def split_long_sections_with_overlap(
    sections: List[str], chunk_size: int, chunk_overlap: int
) -> List[str]:
    """
    If an article section is larger than chunk_size, split it using SimpleTextSplitter
    (character-based) while keeping overlap. Otherwise keep as-is.

    This ensures article boundaries are respected first, then size constraints apply.
    """
    splitter = SimpleTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out: List[str] = []

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= chunk_size:
            out.append(sec)
        else:
            # Fall back to recursive splitting inside the article text
            out.extend(splitter.split_text(sec))

    return out


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 300):
        self.text_splitter = SimpleTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file using PyMuPDF."""
        parts: List[str] = []
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    # keep page boundaries as paragraphs
                    parts.append(page.get_text("text"))
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""

        return "\n\n".join(parts)

    def semantic_chunk_text(self, text: str) -> List[str]:
        """
        Sentence-based chunking (no '.'), with overlap.
        Uses ! ? Armenian '։' as sentence boundaries.
        """
        text = normalize_pdf_text(text)
        sentences = split_sentences_no_dot(text)
        return semantic_chunk_sentences(
            sentences,
            chunk_size=self.text_splitter.chunk_size,
            chunk_overlap=self.text_splitter.chunk_overlap,
        )

    def article_chunk_text(self, text: str) -> List[str]:
        """
        Article-based chunking: split on the word "հոդված" (case-insensitive),
        then enforce chunk_size by splitting overly-long sections.
        """
        text = normalize_pdf_text(text)
        sections = split_by_article_word(text)
        return split_long_sections_with_overlap(
            sections,
            chunk_size=self.text_splitter.chunk_size,
            chunk_overlap=self.text_splitter.chunk_overlap,
        )

    def process_directory(
        self, data_dir: str, use_semantic: bool = False, use_article: bool = True
    ) -> List[Dict]:
        """
        Process all PDFs in a directory and return chunks with metadata.

        Modes:
          - use_article=True  -> split by "հոդված" boundaries (recommended for legal codes)
          - else use_semantic -> sentence-based chunking
          - else             -> recursive splitting
        """
        all_chunks: List[Dict] = []
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            return []

        pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]

        if use_article:
            mode = "Article(հոդված)"
        else:
            mode = "Semantic" if use_semantic else "Recursive"

        print(f"Processing {len(pdf_files)} PDF files (Mode: {mode})...")

        for filename in tqdm(pdf_files):
            file_path = os.path.join(data_dir, filename)
            text = self.extract_text_from_pdf(file_path)

            if not text.strip():
                continue

            if use_article:
                chunks = self.article_chunk_text(text)
            elif use_semantic:
                chunks = self.semantic_chunk_text(text)
            else:
                chunks = self.text_splitter.split_text(normalize_pdf_text(text))

            for i, chunk in enumerate(chunks):
                all_chunks.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "source": filename,
                            "chunk_id": i,
                            "path": file_path,
                            "chunking": mode,
                        },
                    }
                )

        return all_chunks


if __name__ == "__main__":
    pass

"""
Arabic Clause Streamer & Conversational Punctuation Pauses (Pillar 6)
Implements natural punctuation micro-pauses (120ms comma, 280ms period/question, 150ms dash/colon)
and coordinating conjunction clause splitting.
"""

import re
from typing import List, Generator, Tuple


class ArabicClauseStreamer:
    """
    Streaming text buffer that detects clause boundaries, punctuation micro-pauses,
    and coordinating conjunctions to stream real-time synthesized speech chunks.
    """

    # Clause boundary punctuation with pause specifications
    PUNCTUATION_PAUSES_MS = {
        "،": 120,
        ",": 120,
        ".": 280,
        "!": 280,
        "؟": 280,
        "?": 280,
        ":": 150,
        "-": 150,
        "—": 150,
        ";": 150,
        "؛": 150
    }

    # Major sentence terminal marks
    SENTENCE_TERMINALS = re.compile(r"([.!?؟]+)")

    # Minor clause marks
    CLAUSE_BOUNDARIES = re.compile(r"([،,;؛:—\-]+)")

    # Coordinating conjunctions that trigger natural respiratory pauses in long clauses
    CONJUNCTIONS = re.compile(r"\s+(ولكن|لكن|و|فـ|ف|يعني|طبعاً|بالتالي)\s+", re.UNICODE)

    def __init__(self, min_clause_words: int = 2, max_clause_words: int = 14):
        self.min_clause_words = min_clause_words
        self.max_clause_words = max_clause_words
        self.buffer = ""
        self.is_interrupted = False

    def reset(self):
        """Resets the streamer state."""
        self.buffer = ""
        self.is_interrupted = False

    def interrupt(self):
        """Immediately interrupts streaming and drops pending buffer."""
        self.is_interrupted = True
        self.buffer = ""

    def feed_token(self, token: str) -> List[Tuple[str, int]]:
        """
        Feeds a text token (or word) into the streaming buffer.
        Returns a list of complete ready clauses with their pause durations in ms:
        [(clause_text, pause_after_ms), ...]
        """
        if self.is_interrupted:
            return []

        self.buffer += token
        return self._extract_clauses(flush=False)

    def flush(self) -> List[Tuple[str, int]]:
        """
        Flushes all remaining buffered text at end of stream.
        """
        if self.is_interrupted or not self.buffer.strip():
            self.buffer = ""
            return []

        clauses = self._extract_clauses(flush=True)
        self.buffer = ""
        return clauses

    def _extract_clauses(self, flush: bool = False) -> List[Tuple[str, int]]:
        ready_clauses: List[Tuple[str, int]] = []

        while True:
            # Check for terminal sentence marks first (. ! ? ؟)
            m = self.SENTENCE_TERMINALS.search(self.buffer)
            if m:
                end_idx = m.end()
                clause = self.buffer[:end_idx].strip()
                punct = m.group(1)[0]
                pause_ms = self.PUNCTUATION_PAUSES_MS.get(punct, 280)

                self.buffer = self.buffer[end_idx:].lstrip()
                if clause:
                    ready_clauses.append((clause, pause_ms))
                continue

            # Check for intermediate clause marks (، , : - ; ؛)
            m_clause = self.CLAUSE_BOUNDARIES.search(self.buffer)
            if m_clause:
                words_before = self.buffer[:m_clause.start()].strip().split()
                if len(words_before) >= self.min_clause_words or flush:
                    end_idx = m_clause.end()
                    clause = self.buffer[:end_idx].strip()
                    punct = m_clause.group(1)[0]
                    pause_ms = self.PUNCTUATION_PAUSES_MS.get(punct, 120)

                    self.buffer = self.buffer[end_idx:].lstrip()
                    if clause:
                        ready_clauses.append((clause, pause_ms))
                    continue

            # Check if buffer has grown too long without punctuation (> max_clause_words)
            words = self.buffer.strip().split()
            if len(words) >= self.max_clause_words:
                # Look for coordinating conjunction to split gracefully
                conj_match = self.CONJUNCTIONS.search(self.buffer)
                if conj_match and len(self.buffer[:conj_match.start()].split()) >= self.min_clause_words:
                    split_idx = conj_match.start()
                    clause = self.buffer[:split_idx].strip()
                    self.buffer = self.buffer[split_idx:].lstrip()
                    if clause:
                        ready_clauses.append((clause, 140))
                    continue
                else:
                    # Split at max words
                    clause_words = words[:self.max_clause_words]
                    clause = " ".join(clause_words)
                    self.buffer = " ".join(words[self.max_clause_words:]).strip()
                    ready_clauses.append((clause, 120))
                    continue

            # If flushing, emit whatever remains
            if flush and self.buffer.strip():
                clause = self.buffer.strip()
                self.buffer = ""
                ready_clauses.append((clause, 280))
                break

            break

        return ready_clauses

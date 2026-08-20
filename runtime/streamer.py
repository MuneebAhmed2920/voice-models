"""
Clause Streamer & Text Token Buffer
Buffers incoming LLM streaming tokens, performs clause boundary detection,
and orchestrates ultra-low-latency real-time TTS synthesis for conversational bots.
"""

import re
import time
from typing import List, Generator, Optional, Tuple


class ClauseStreamer:
    """
    Splits continuous token streams into conversational clauses on punctuation/sentence boundaries.
    Optimizes Time-To-First-Audio (TTFA) for LLM-driven voice bots.
    """

    # Clause boundary delimiters: sentence ends, comma pauses, colons, semicolons, newlines
    CLAUSE_SPLIT_PATTERN = re.compile(r"([.!?,;:\n\r]+)")
    
    # Minimum characters/words for a standalone clause to avoid awkward fragments
    MIN_CLAUSE_CHARS = 4
    MAX_CLAUSE_WORDS = 14  # If no punctuation after 14 words, split naturally

    def __init__(self, min_clause_chars: int = 4, max_clause_words: int = 14):
        self.min_clause_chars = min_clause_chars
        self.max_clause_words = max_clause_words
        self.buffer = ""
        self.clause_index = 0
        self.is_interrupted = False

    def reset(self):
        """Resets the streamer state."""
        self.buffer = ""
        self.clause_index = 0
        self.is_interrupted = False

    def interrupt(self):
        """Signals barge-in / cancellation and purges buffered text."""
        self.is_interrupted = True
        self.buffer = ""

    def feed_token(self, token: str) -> List[str]:
        """
        Feeds a text chunk/token from the LLM.
        Returns a list of complete clauses ready for immediate TTS generation.
        """
        if self.is_interrupted:
            return []

        self.buffer += token
        return self._extract_clauses(flush=False)

    def flush(self) -> List[str]:
        """
        Flushes any remaining text in the buffer at the end of LLM generation.
        """
        if self.is_interrupted or not self.buffer.strip():
            self.buffer = ""
            return []

        return self._extract_clauses(flush=True)

    def _extract_clauses(self, flush: bool = False) -> List[str]:
        """
        Extracts ready clauses from the current text buffer.
        """
        ready_clauses: List[str] = []

        while self.buffer:
            # Check for punctuation splits
            parts = self.CLAUSE_SPLIT_PATTERN.split(self.buffer)
            
            if len(parts) >= 3:
                # parts[0] is text, parts[1] is delimiter, parts[2:] is remainder
                clause_text = (parts[0] + parts[1]).strip()
                remainder = "".join(parts[2:])

                # Check if clause is too short, e.g. "Dr." or "Mr." or single letter
                if len(clause_text) < self.min_clause_chars and not flush:
                    # Keep accumulating in buffer
                    break

                if clause_text:
                    ready_clauses.append(clause_text)
                    self.clause_index += 1

                self.buffer = remainder
            else:
                # No punctuation delimiter found yet
                # Check if word count exceeded max_clause_words
                words = self.buffer.split()
                if len(words) >= self.max_clause_words and not flush:
                    # Split at a word boundary
                    split_idx = self.max_clause_words
                    clause_text = " ".join(words[:split_idx]).strip()
                    remainder = " ".join(words[split_idx:]).strip()
                    if clause_text:
                        ready_clauses.append(clause_text)
                        self.clause_index += 1
                    self.buffer = remainder
                elif flush:
                    remaining_text = self.buffer.strip()
                    if remaining_text:
                        ready_clauses.append(remaining_text)
                        self.clause_index += 1
                    self.buffer = ""
                else:
                    break

        return ready_clauses

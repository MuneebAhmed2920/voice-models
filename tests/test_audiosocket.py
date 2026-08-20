"""
Unit tests for Asterisk AudioSocket framing and protocol constants.
"""

import struct
import uuid
import unittest
from server.audiosocket_server import (
    TYPE_UUID,
    TYPE_AUDIO,
    TYPE_SILENCE,
    TYPE_ERROR,
    HEADER_FORMAT,
    HEADER_SIZE
)


class TestAudioSocketProtocol(unittest.TestCase):

    def test_header_packing(self):
        msg_type = TYPE_AUDIO
        payload_len = 320
        header = struct.pack(HEADER_FORMAT, msg_type, payload_len)
        self.assertEqual(len(header), 3)

        parsed_type, parsed_len = struct.unpack(HEADER_FORMAT, header)
        self.assertEqual(parsed_type, TYPE_AUDIO)
        self.assertEqual(parsed_len, 320)

    def test_uuid_framing(self):
        call_uuid = uuid.uuid4()
        payload = call_uuid.bytes
        self.assertEqual(len(payload), 16)

        header = struct.pack(HEADER_FORMAT, TYPE_UUID, len(payload))
        full_msg = header + payload
        self.assertEqual(len(full_msg), 19)

        parsed_type, parsed_len = struct.unpack(HEADER_FORMAT, full_msg[:HEADER_SIZE])
        self.assertEqual(parsed_type, TYPE_UUID)
        self.assertEqual(parsed_len, 16)
        extracted_uuid = uuid.UUID(bytes=full_msg[HEADER_SIZE:])
        self.assertEqual(extracted_uuid, call_uuid)


if __name__ == "__main__":
    unittest.main()

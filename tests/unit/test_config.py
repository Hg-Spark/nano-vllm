import tempfile
import unittest

from nanovllm.config import Config


class ConfigTest(unittest.TestCase):

    def test_kvcache_block_size_must_be_positive(self):
        with tempfile.TemporaryDirectory() as model_dir:
            for block_size in (0, -256):
                with self.subTest(block_size=block_size):
                    with self.assertRaisesRegex(
                        ValueError,
                        "positive multiple of 256",
                    ):
                        Config(
                            model_dir,
                            kvcache_block_size=block_size,
                        )


if __name__ == "__main__":
    unittest.main()

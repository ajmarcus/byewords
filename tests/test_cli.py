import unittest
from io import StringIO
from contextlib import redirect_stdout

from byewords.cli import main


class TestCli(unittest.TestCase):

    def test_cli_prints_expected_message(self):
        buf = StringIO()

        with redirect_stdout(buf):
            main()

        self.assertEqual(buf.getvalue().strip(), "Hello from byewords!")


if __name__ == "__main__":
    unittest.main()

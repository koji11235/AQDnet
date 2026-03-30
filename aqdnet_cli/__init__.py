"""
AQDnet CLI: Command-line interface for feature extraction, training, and prediction
"""

__version__ = "0.1.0"
__author__ = "AQDnet Authors"


def main():
    from .__main__ import main as cli_main

    return cli_main()

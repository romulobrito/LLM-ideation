#!/usr/bin/env python3
"""
Compat: delega para embedding_models_eval.cli (instale com pip install -e . ou PYTHONPATH=src).
"""

import sys

from embedding_models_eval.cli import main

if __name__ == "__main__":
    sys.exit(main())

"""Allow ``python -m spinflow`` from the repository (after ``pip install -e .`` or with ``PYTHONPATH=src``)."""
from .main import main

if __name__ == "__main__":
    main()

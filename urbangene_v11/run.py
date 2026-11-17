"""Local entry point; does not modify the installed V0 package."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path[:0] = [str(Path(__file__).parent / "src"),
                str(Path(__file__).parent.parent / "urbangene/src")]
from urbangene_v11.cli import main

if __name__ == "__main__":
    main()

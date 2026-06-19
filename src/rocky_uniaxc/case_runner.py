"""CLI entry point for running a single uniaxial compression case.

Reads a ``settings.json`` file, constructs a
:class:`~rocky_uniaxc.pyrocky.uniax.UniaxialCompressionSimulation`, and
executes it.
"""

import json
import sys
from pathlib import Path
from rocky_uniaxc.pyrocky.uniax import Settings, UniaxialCompressionSimulation


def main():
    """Parse command-line arguments and run a single simulation case.

    Expects a single argument: the path to a ``settings.json`` file.

    Example::

        python -m rocky_uniaxc.case_runner path/to/settings.json
    """
    if len(sys.argv) < 2:
        print("Usage: python -m rocky_uniaxc.case_runner path/to/settings.json")
        sys.exit(1)

    settings_path = Path(sys.argv[1]).resolve()
    project_dir = settings_path.parent

    with open(settings_path) as f:
        data = json.load(f)
    data["project_dir"] = str(project_dir)
    settings = Settings.from_dict(data)

    sim = UniaxialCompressionSimulation(settings)
    sim.execute()


if __name__ == "__main__":
    main()

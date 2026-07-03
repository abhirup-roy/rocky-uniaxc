import os
import subprocess
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm


class cd:
    """Context manager for temporarily changing the current working directory.

    Usage::

        with cd("/tmp"):
            # working directory is now /tmp
            ...
        # original directory is restored
    """

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


# Default command run inside each job to drive a single uniaxial case.
DEFAULT_RUN_COMMAND = 'Rocky --script "script_uniax.py" --headless >> rocky.log'

# Module-load / setup blocks reused by the cluster presets.
_BEAR_COMMANDS = (
    "set -e\n\n"
    "module purge; module load bluebear\n"
    "module load bear-apps/2024a\n"
    "module load ANSYS_Rocky/2026R1\n"
)
_AZ_COMMANDS = "set -e\n\nml rocky/26.1.0\n"
_DEFAULT_ACCOUNT = "windowcr-astrazeneca-muhammad"


class RockyScheduler:
    """Generate and submit SLURM jobs for Rocky uniaxial compression cases.

    Each ``#SBATCH`` directive is held declaratively as an attribute; directives
    left as ``None`` are omitted from the generated script. Arbitrary extra
    directives can be passed as keyword arguments (e.g. ``cpus_per_gpu=1`` becomes
    ``#SBATCH --cpus-per-gpu 1``). Based loosely off the `coexist` module's scheduler interface.

    Rather than constructing this directly, most users _may_ reach for one of the
    cluster presets — :meth:`bb_cpu`, :meth:`bb_gpu`, :meth:`az_gpu` or
    :meth:`custom`.

    Parameters
    ----------
    job_name : str, default "uniaxc"
        The ``#SBATCH --job-name`` value.
    time : str, default "12-0"
        The ``#SBATCH --time`` value (e.g. ``"12-0"`` for 12 days).
    ntasks : int or None, default 1
        The ``#SBATCH --ntasks`` value.
    gres : str or None
        The ``#SBATCH --gres`` value, e.g. ``"gpu:1"``.
    cpus_per_task : int or None
        The ``#SBATCH --cpus-per-task`` value.
    nodes : int or None
        The ``#SBATCH --nodes`` value.
    qos : str or None
        The ``#SBATCH --qos`` value.
    account : str or None
        The ``#SBATCH --account`` value.
    partition : str or None
        The ``#SBATCH --partition`` value.
    mail_type : str or None
        The ``#SBATCH --mail-type`` value.
    shebang : str, default "#!/bin/bash"
        First line of the generated script.
    commands : str
        Setup commands (module loads, ``set -e``, etc.) inserted before the
        Rocky launch command.
    run_command : str
        The command that runs the simulation. Defaults to
        :data:`DEFAULT_RUN_COMMAND`.
    script_name : str, default "runRocky.sh"
        File name of the generated submission script inside each case directory.
    raw_script : str or None
        If provided, :meth:`generate` writes this content verbatim and ignores
        all other directive attributes. Used by :meth:`custom`.
    **sbatch : Any
        Additional ``#SBATCH`` directives; keys have underscores converted to
        dashes (``cpus_per_gpu="1"`` -> ``#SBATCH --cpus-per-gpu 1``).

    Examples
    --------
    >>> from rocky_uniaxc.schedulers import RockyScheduler
    >>> scheduler = RockyScheduler.bb_cpu(ncpus=20, run_days=5)
    >>> scheduler.generate("case_0")          # writes case_0/runRocky.sh
    >>> scheduler.launch_all(["case_0", "case_1"])   # submits both via sbatch
    """

    def __init__(
        self,
        job_name="uniaxc",
        time="10-0",
        ntasks=1,
        gres=None,
        cpus_per_task=None,
        nodes=None,
        qos=None,
        account=None,
        partition=None,
        mail_type=None,
        shebang="#!/bin/bash",
        commands="set -e\n",
        run_command=DEFAULT_RUN_COMMAND,
        script_name="runRocky.sh",
        raw_script=None,
        **sbatch,
    ):
        self.job_name = job_name
        self.time = str(time) if time is not None else None
        self.ntasks = ntasks
        self.gres = gres
        self.cpus_per_task = cpus_per_task
        self.nodes = nodes
        self.qos = qos
        self.account = account
        self.partition = partition
        self.mail_type = mail_type
        self.shebang = shebang
        self.commands = commands
        self.run_command = run_command
        self.script_name = script_name
        self.raw_script = raw_script
        self.sbatch = sbatch

    @classmethod
    def bb_cpu(cls, ncpus=20, run_days=12, account=None, **kwargs):
        """Preset for the University of Birmingham BlueBear CPU partition."""
        return cls(
            time=f"{run_days}-0",
            ntasks=ncpus,
            cpus_per_task=1,
            nodes=1,
            qos="bbdefault",
            account=account,
            mail_type="ALL",
            commands=_BEAR_COMMANDS,
            **kwargs,
        )

    @classmethod
    def bb_gpu(cls, ngpus=1, run_days=12, account=None, **kwargs):
        """Preset for the University of Birmingham BlueBear GPU partition."""
        return cls(
            time=f"{run_days}-0",
            ntasks=1,
            gres=f"gpu:{ngpus}",
            qos="bbgpu",
            account=account,
            commands=_BEAR_COMMANDS,
            **kwargs,
        )

    @classmethod
    def az_gpu(cls, ngpus=1, run_days=12, **kwargs):
        """Preset for SCP GPU partition."""
        return cls(
            shebang="#!/bin/sh",
            time=f"{run_days}-0",
            ntasks=1,
            gres=f"gpu:{ngpus}",
            partition="gpu",
            commands=_AZ_COMMANDS,
            cpus_per_gpu=1,
            **kwargs,
        )

    @classmethod
    def custom(cls, script_text, script_name="runRocky.sh"):
        """Preset that writes a fully user-supplied submission script.

        Parameters
        ----------
        script_text : str
            Complete submission script. Must start with a shebang
            (``#!``) line.
        script_name : str, default "runRocky.sh"
            File name of the generated script.

        Raises
        ------
        ValueError
            If ``script_text`` does not start with a shebang.
        """
        if not isinstance(script_text, str) or not script_text.startswith("#!"):
            raise ValueError(
                "Custom script must be a string starting with a shebang (e.g. "
                "'#!/bin/bash')."
            )
        return cls(raw_script=script_text, script_name=script_name)

    def _sbatch_directives(self):
        """Return an ordered mapping of ``#SBATCH`` directive name -> value.

        ``None`` values are skipped. Keys use the SLURM dash form.
        """
        directives = OrderedDict()
        directives["job-name"] = self.job_name
        directives["ntasks"] = self.ntasks
        directives["cpus-per-task"] = self.cpus_per_task
        directives["nodes"] = self.nodes
        directives["time"] = self.time
        directives["gres"] = self.gres
        directives["qos"] = self.qos
        directives["partition"] = self.partition
        directives["account"] = self.account
        directives["mail-type"] = self.mail_type

        for key, val in self.sbatch.items():
            directives[key.replace("_", "-")] = val

        return OrderedDict((k, v) for k, v in directives.items() if v is not None)

    def render(self):
        """Return the submission script as a string."""
        if self.raw_script is not None:
            return self.raw_script

        lines = [self.shebang]
        for key, val in self._sbatch_directives().items():
            lines.append(f"#SBATCH --{key}={val}")

        script = "\n".join(lines) + "\n\n"
        commands = self.commands
        if not commands.endswith("\n"):
            commands += "\n"
        script += commands
        script += f"\n{self.run_command}\n"
        return script

    def generate(self, case_dir):
        """Write the submission script into ``case_dir``.

        Parameters
        ----------
        case_dir : str or pathlib.Path
            Directory to write ``self.script_name`` into.

        Returns
        -------
        pathlib.Path
            Path to the written script.
        """
        script_path = (Path(case_dir) / self.script_name).resolve()
        script_path.write_text(self.render())
        return script_path

    def submit(self, case_dir):
        """Submit a single case via ``sbatch`` from within ``case_dir``.

        Parameters
        ----------
        case_dir : str or pathlib.Path
            Case directory containing the generated script.

        Returns
        -------
        subprocess.CompletedProcess
            The completed ``sbatch`` process.
        """
        with cd(str(case_dir)):
            return subprocess.run(
                ["sbatch", self.script_name],
                check=True,
                capture_output=True,
                text=True,
            )

    def launch_all(self, case_dirs, desc="Submitting Jobs"):
        """Submit a batch of cases via ``sbatch`` with a progress bar.

        Parameters
        ----------
        case_dirs : iterable of str or pathlib.Path
            Case directories to submit.
        desc : str, default "Submitting Jobs"
            Progress-bar description.

        Returns
        -------
        list[int]
            Indices of cases that failed to launch.
        """
        case_dirs = list(case_dirs)
        total_cases = len(case_dirs)
        failed_jobs = []

        for i, case_dir in tqdm(
            enumerate(case_dirs), total=total_cases, desc=desc, unit="case"
        ):
            try:
                self.submit(case_dir)
            except subprocess.CalledProcessError as e:
                err_msg = e.stderr.strip() if e.stderr else "Unknown error"
                tqdm.write(f"Failed to launch case {i}: {err_msg}")
                failed_jobs.append(i)
            except Exception as e:
                tqdm.write(f"Unexpected error in case {i}: {e}")
                failed_jobs.append(i)

        if failed_jobs:
            print(
                f"\nCompleted with errors. {len(failed_jobs)} jobs failed to "
                f"launch (Case indices: {failed_jobs})."
            )
        else:
            print("\nSuccess: All jobs launched.")

        print(f"All {total_cases} cases prepared and processed.")
        return failed_jobs

    def __repr__(self):
        docs = []
        for attr in dir(self):
            if not attr.startswith("_"):
                memb = getattr(self, attr)
                if not callable(memb):
                    docs.append(f"{attr} = {memb}")

        name = self.__class__.__name__
        underline = "-" * len(name)
        return f"{name}\n{underline}\n" + "\n".join(docs)

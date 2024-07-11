import argparse
from datetime import timedelta
import subprocess
from pathlib import Path

JOBNAME = "multigrad_benchmark"
ACCOUNT = "galsampler"
PARTITION = "bdwall"


def generate_sbatch(
    *, jobnum,
    numtasks,
    nhalos,
    nsteps,
    timelim,
        tasks_on_separate_nodes=True):

    return f"""#!/bin/bash

#SBATCH --job-name={JOBNAME}_{jobnum}
#SBATCH --output={JOBNAME}_{jobnum}_job.out
#SBATCH --error={JOBNAME}_{jobnum}_error.out
#SBATCH --account={ACCOUNT}
#SBATCH --partition={PARTITION}
#SBATCH --time={str(timelim)}
#SBATCH --nodes={numtasks if tasks_on_separate_nodes else 1}
#SBATCH --ntasks-per-node={1 if tasks_on_separate_nodes else numtasks}


srun --cpu-bind=cores --ntasks {numtasks} python benchmark.py --num-halos {nhalos} --save bench.txt --num-steps {nsteps}
"""


parser = argparse.ArgumentParser(
    __file__,
    description="Submit scaling test jobs for multigrad"
)
parser.add_argument("--num-halos", type=int, default=100_000)
parser.add_argument("--num-steps", type=int, default=100)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--jobname", type=str, default=JOBNAME)
parser.add_argument("--account", type=str, default=ACCOUNT)
parser.add_argument("--partition", type=str, default=PARTITION)
parser.add_argument("--tasks-on-single-node", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    JOBNAME = args.jobname
    ACCOUNT = args.account
    PARTITION = args.partition

    # Generate batch scripts
    sbatches = []
    for num_nodes in range(1, 11):
        sbatches.append(generate_sbatch(
            jobnum=num_nodes,
            numtasks=num_nodes,
            nhalos=args.num_halos,
            nsteps=args.num_steps,
            timelim=timedelta(minutes=10),
            tasks_on_separate_nodes=not args.tasks_on_single_node,
        ))

    # Submit jobs in parallel
    for i, sbatch in enumerate(sbatches):
        filename = Path(f"temp_sbatch{i}.sh")
        with open(filename, "w") as f:
            f.write(sbatch)
        cmd = f"sbatch {filename}".split()
        subprocess.Popen(cmd)

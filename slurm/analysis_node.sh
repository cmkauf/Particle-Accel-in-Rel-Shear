#!/bin/bash
# ---------------------------------------------------------------------------------------
# Single-node shearpic analysis job for Stampede3, using a process pool on one node.
#
#   sbatch slurm/analysis_node.sh                 # edit RUN and the python line below first
#   sbatch --export=ALL,RUN=423 slurm/analysis_node.sh
#
# Partitions and memory per core are listed in README.md, section 6.  On spr nodes pass
# --workers N or --mem-per-task, since one worker per core can run out of memory.
# On a login node shearpic uses a single worker.
# ---------------------------------------------------------------------------------------
#SBATCH -J shearpic-analysis
#SBATCH -o %x-%j.out                  # stdout+stderr file: <job name>-<job id>.out
#SBATCH -p skx                        # partition (skx-dev for < 2 h)
#SBATCH -N 1                          # one node: parallel_map uses a process pool on it
#SBATCH --ntasks-per-node=48          # all 48 skx cores (sets SLURM_CPUS_ON_NODE -> worker count)
#SBATCH -t 02:00:00
#SBATCH -A <ALLOCATION>               # your TACC allocation/project, e.g. TG-PHY250146 (see `/usr/local/etc/taccinfo`)

set -euo pipefail

RUN=${RUN:-423}                                        # run number, run name or directory
REPO=${REPO:-$HOME/Particle-Accel-in-Rel-Shear}        # git checkout of this repository

module load python
source "$WORK/venvs/shearpic/bin/activate"             # created once, see README "Installation"

# One BLAS/OpenMP thread per worker; shearpic's pool sets this too, the exports cover code outside it.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
# shearpic reads .athdf files without file locking; yt or h5py code can otherwise fail on Lustre.
export HDF5_USE_FILE_LOCKING=FALSE

# Runs and products live on $SCRATCH, which purges files not accessed for 10 days,
# so the products are copied to $WORK at the end.
export PARTICLE_ACCEL_DATA=${PARTICLE_ACCEL_DATA:-$SCRATCH/particle-accel/runs}
export PARTICLE_ACCEL_OUTPUT=${PARTICLE_ACCEL_OUTPUT:-$SCRATCH/particle-accel/outputs}
mkdir -p "$PARTICLE_ACCEL_OUTPUT"

cd "$SCRATCH"
shearpic env                                           # log machine, CPUs, memory, paths and registry

python "$REPO/examples/parallel_snapshots.py" --run "$RUN" --workers auto --mem-per-task 2GB

# Keep the small products: figures, tables and reduced arrays.
mkdir -p "$WORK/particle-accel/outputs"
rsync -a --prune-empty-dirs \
      --include='*/' --include='*.png' --include='*.pdf' --include='*.csv' --include='*.npz' \
      --include='*.mp4' --include='*.gif' --exclude='*' \
      "$PARTICLE_ACCEL_OUTPUT/" "$WORK/particle-accel/outputs/"
echo "products copied to $WORK/particle-accel/outputs"

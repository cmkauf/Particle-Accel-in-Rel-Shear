#!/bin/bash
# ---------------------------------------------------------------------------------------
# Multi-node shearpic analysis job for Stampede3, with one Python process per MPI rank.
#
#   sbatch slurm/analysis_mpi.sh
#   sbatch --export=ALL,RUN=423 slurm/analysis_mpi.sh
#
# ibrun starts nodes x ntasks-per-node copies of the script.  With --backend auto,
# parallel_map gives each rank items[rank::size] to process serially; rank 0 gathers the
# results and writes the products, and raises if any item failed, so the job does not hang.
# Nested parallel_map calls run serially.  Use analysis_node.sh unless one node is not enough.
#
# Each rank holds one snapshot at a time; on spr nodes use fewer ranks per node.
#
# Build mpi4py once against the TACC MPI library on a login node; PyPI wheels may not work with ibrun:
#   module load python impi
#   source $WORK/venvs/shearpic/bin/activate
#   MPICC=mpicc pip install --no-cache-dir --no-binary=mpi4py mpi4py
# ---------------------------------------------------------------------------------------
#SBATCH -J shearpic-mpi
#SBATCH -o %x-%j.out                  # stdout+stderr file
#SBATCH -p skx                        # partition (skx-dev: max 2 h, for tests)
#SBATCH -N 2
#SBATCH --ntasks-per-node=48          # MPI ranks per node (one per skx core)
#SBATCH -t 02:00:00
#SBATCH -A <ALLOCATION>               # your TACC allocation/project, e.g. TG-PHY250146

set -euo pipefail

RUN=${RUN:-423}
REPO=${REPO:-$HOME/Particle-Accel-in-Rel-Shear}

module load python impi
source "$WORK/venvs/shearpic/bin/activate"

# Without these, every rank would start one thread per core.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export HDF5_USE_FILE_LOCKING=FALSE   # for yt or h5py code; shearpic reads .athdf files without locking

export PARTICLE_ACCEL_DATA=${PARTICLE_ACCEL_DATA:-$SCRATCH/particle-accel/runs}
export PARTICLE_ACCEL_OUTPUT=${PARTICLE_ACCEL_OUTPUT:-$SCRATCH/particle-accel/outputs}
mkdir -p "$PARTICLE_ACCEL_OUTPUT"

cd "$SCRATCH"
python -c "import mpi4py; print('mpi4py', mpi4py.__version__)"   # fail early if mpi4py is missing

# --backend auto picks MPI because ibrun starts more than one rank.
ibrun python "$REPO/examples/parallel_snapshots.py" --run "$RUN" --backend auto

mkdir -p "$WORK/particle-accel/outputs"
rsync -a --prune-empty-dirs \
      --include='*/' --include='*.png' --include='*.pdf' --include='*.csv' --include='*.npz' \
      --include='*.mp4' --include='*.gif' --exclude='*' \
      "$PARTICLE_ACCEL_OUTPUT/" "$WORK/particle-accel/outputs/"
echo "products copied to $WORK/particle-accel/outputs"

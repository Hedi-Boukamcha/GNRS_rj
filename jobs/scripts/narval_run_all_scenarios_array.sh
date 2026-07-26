#!/bin/bash
# ============================================================
# Job SLURM ARRAY (Narval) : les 6 scenarios different_costs_* de
# data/controlled_orders/test, un job par scenario, en parallele.
# Chaque tache lance : run_controlled_acceptation.py --mode all
# --scenario_filter <scenario> --deltas 0.2 0.5 0.8
#
# Pre-requis :
#   1) avoir clone/copie le repo GNRS_rj sur Narval
#   2) avoir lance une fois jobs/scripts/narval_setup_env.sh
#   3) creer le dossier de logs : mkdir -p logs
#
# Soumission (depuis la racine du repo) :
#   sbatch jobs/scripts/narval_run_all_scenarios_array.sh
#
# Suivi :
#   squeue -u $USER
#   sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode
#   tail -f logs/gnrs-scenarios-<JOBID>_<TASKID>.out
# ============================================================
#SBATCH --account=def-adhaj          # <-- A REMPLACER par ton compte RAC (def-xxx ou rrg-xxx)
#SBATCH --job-name=gnrs-scenarios
#SBATCH --array=0-5
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# #SBATCH --mail-user=hedi.boukamcha.1@ulaval.ca
# #SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ---------- Un scenario par indice de l'array (0 a 5) ----------
SCENARIOS=(
    "different_costs_100_25"
    "different_costs_100_50"
    "different_costs_25_100"
    "different_costs_25_50"
    "different_costs_50_100"
    "different_costs_50_25"
)
SCENARIO="${SCENARIOS[$SLURM_ARRAY_TASK_ID]}"

DELTAS="0.2 0.5 0.8"
AGENT_PATH="data/training_costs/"
DEVICE="cuda"
GENERATE_GANTTS="true"
# ------------------------------------------------------------------

module purge
module load StdEnv/2023 python/3.11 cuda/12.2

source "$HOME/envs/gnrs_rj/bin/activate"

cd "$SLURM_SUBMIT_DIR"

echo "== Tache array $SLURM_ARRAY_TASK_ID (job $SLURM_ARRAY_JOB_ID) sur $(hostname) =="
echo "Scenario : $SCENARIO"
echo "Deltas   : $DELTAS"
nvidia-smi || true

python run_controlled_acceptation.py \
    --mode all \
    --root_dir "data/controlled_orders/test" \
    --scenario_filter "$SCENARIO" \
    --deltas $DELTAS \
    --device "$DEVICE" \
    --agent_path "$AGENT_PATH" \
    --generate_gantts "$GENERATE_GANTTS"

echo "== Scenario $SCENARIO termine (tache $SLURM_ARRAY_TASK_ID) =="

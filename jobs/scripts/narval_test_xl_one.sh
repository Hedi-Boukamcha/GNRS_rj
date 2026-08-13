#!/bin/bash
# ============================================================
# Job SLURM (Narval) : test rapide sur UNE SEULE instance "xl"
# et UN SEUL delta, pour verifier que le pipeline tourne bien sur
# Calcul Canada (env, GPU, acces aux donnees/agent, ecriture des
# resultats) AVANT de soumettre le gros array de 9 taches
# (jobs/scripts/narval_run_xl_array.sh, 48h/tache).
#
# Pre-requis :
#   1) avoir clone/copie le repo GNRS_rj sur Narval
#   2) avoir lance une fois jobs/scripts/narval_setup_env.sh
#   3) creer le dossier de logs : mkdir -p logs
#
# Soumission (depuis la racine du repo) :
#   sbatch jobs/scripts/narval_test_xl_one.sh
#
# Suivi :
#   squeue -u $USER
#   tail -f logs/gnrs-xl-test-<JOBID>.out
#
# Si ca reussit : results/controlled_orders/test/xl/delta_0_1/same_costs/instance_1_early/
# et analysis/xl/delta_0_1/same_costs/instance_1_early/ doivent contenir des fichiers.
# Tu peux alors soumettre le vrai array (narval_run_xl_array.sh).
# ============================================================
#SBATCH --account=def-adhaj          # <-- A REMPLACER par ton compte RAC (def-xxx ou rrg-xxx)
#SBATCH --job-name=gnrs-xl-test
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---------- Une seule instance, un seul delta ----------
INPUT="data/controlled_orders_ub/test/xl/same_costs/instance_1_early.json"
SCENARIO="xl"
INST="same_costs"
DELTA_RATIO=0.1
OUTPUT_ROOT="results/controlled_orders/test"
AGENT_PATH="data/training_costs_ub/"
DEVICE="cuda"
GENERATE_GANTTS="true"
# --------------------------------------------------------

module purge
module load StdEnv/2023 python/3.11 cuda/12.2

source /home/hedibk/envs/gnrs_rj/bin/activate

cd "$SLURM_SUBMIT_DIR"

echo "== Job $SLURM_JOB_ID demarre sur $(hostname) =="
echo "Input    : $INPUT"
echo "Delta    : $DELTA_RATIO"
echo "Device   : $DEVICE"
nvidia-smi || true

python acceptation_solver.py \
    --mode one \
    --input "$INPUT" \
    --output_root "$OUTPUT_ROOT" \
    --scenario "$SCENARIO" \
    --inst "$INST" \
    --delta_ratio "$DELTA_RATIO" \
    --device "$DEVICE" \
    --agent_path "$AGENT_PATH" \
    --generate_gantts "$GENERATE_GANTTS"

echo "== Job $SLURM_JOB_ID termine =="

#!/bin/bash
# ============================================================
# Job SLURM (Nibi) : methode d'acceptation sur UNE instance
# (run_controlled_acceptation.py --mode one)
#
# Pre-requis :
#   1) avoir clone/copie le repo GNRS_rj sur Nibi (independant de Narval)
#   2) avoir lance une fois jobs/scripts/nibi_setup_env.sh
#   3) creer le dossier de logs : mkdir -p logs
#
# Soumission (depuis la racine du repo, sur Nibi) :
#   sbatch jobs/scripts/nibi_run_acceptation_one.sh
#
# NOTE : verifie que --account est valide sur Nibi (les allocations
# par defaut def-xxx/rrg-xxx sont normalement valables sur tous les
# clusters nationaux, mais confirme avec `sacctmgr show associations
# user=$USER` une fois connecte). Verifie aussi le type de GPU
# disponible avec `sinfo -o "%N %G"` si tu veux cibler un modele
# precis (ex: --gpus-per-node=h100:1).
# ============================================================
#SBATCH --account=def-adhaj
#SBATCH --job-name=gnrs-acceptation
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# #SBATCH --mail-user=hedi.boukamcha.1@ulaval.ca
# #SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ---------- Parametres de l'instance a lancer ----------
INPUT="data/controlled_orders/test/different_costs_50_25/inst1/early_1_to_cmax_over_3.json"
SCENARIO="different_costs_50_25"
INST="inst1"
DELTA_RATIO=0.2
AGENT_PATH="data/training_costs/"
DEVICE="cuda"
GENERATE_GANTTS="false"
# --------------------------------------------------------

module purge
module load StdEnv/2023 python/3.11 cuda/12.2

source "$HOME/envs/gnrs_rj/bin/activate"

cd "$SLURM_SUBMIT_DIR"

echo "== Job $SLURM_JOB_ID demarre sur $(hostname) =="
echo "Input    : $INPUT"
echo "Scenario : $SCENARIO"
echo "Inst     : $INST"
echo "Delta    : $DELTA_RATIO"
echo "Device   : $DEVICE"
nvidia-smi || true

python run_controlled_acceptation.py \
    --mode one \
    --input "$INPUT" \
    --scenario "$SCENARIO" \
    --inst "$INST" \
    --delta_ratio "$DELTA_RATIO" \
    --device "$DEVICE" \
    --agent_path "$AGENT_PATH" \
    --generate_gantts "$GENERATE_GANTTS"

echo "== Job $SLURM_JOB_ID termine =="

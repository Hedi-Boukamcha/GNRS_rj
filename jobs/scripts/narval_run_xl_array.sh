#!/bin/bash
# ============================================================
# Job SLURM ARRAY (Narval) : methode d'acceptation sur les
# instances de TEST "xl" (data/controlled_orders_ub/test/xl).
#
# 3 scenarios de cout x 3 deltas = 9 taches, une par combinaison,
# chacune traitant les 60 instances du scenario pour son delta.
# Les instances xl sont ~10x plus grosses (en taille de fichier)
# que les "s", d'ou le split fin (une tache par delta) plutot
# qu'une tache par scenario seul : ca reduit le risque de depasser
# le --time alloue et permet un meilleur remplissage de la queue.
#
# IMPORTANT sur les arguments d'acceptation_solver.py :
#   Avec root_dir pointant directement sur le dossier de taille
#   (ex: data/controlled_orders_ub/test/xl), le code interprete :
#     scenario = nom du dossier root_dir (ici "xl", constant)
#     inst     = nom du sous-dossier (same_costs / portion_of_3_7 / portion_of_7_3)
#   => pour filtrer par scenario de cout il faut donc utiliser
#     --inst_filter, PAS --scenario_filter (qui ne matcherait jamais
#     puisque `scenario` vaut toujours "xl" dans ce mode).
#
# Pre-requis :
#   1) avoir clone/copie le repo GNRS_rj sur Narval
#   2) avoir lance une fois jobs/scripts/narval_setup_env.sh
#   3) creer le dossier de logs : mkdir -p logs
#
# Soumission (depuis la racine du repo) :
#   sbatch jobs/scripts/narval_run_xl_array.sh
#
# Suivi :
#   squeue -u $USER
#   sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode
#   tail -f logs/gnrs-xl-<JOBID>_<TASKID>.out
# ============================================================
#SBATCH --account=def-adhaj          # <-- A REMPLACER par ton compte RAC (def-xxx ou rrg-xxx)
#SBATCH --job-name=gnrs-xl
#SBATCH --array=0-8
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# #SBATCH --mail-user=hedi.boukamcha.1@ulaval.ca
# #SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ---------- Grille scenario x delta (3 x 3 = 9 taches) ----------
SCENARIOS=("same_costs" "portion_of_3_7" "portion_of_7_3")
DELTAS=(0.1 0.2 0.5)

SCENARIO_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
DELTA_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

SCENARIO="${SCENARIOS[$SCENARIO_IDX]}"
DELTA="${DELTAS[$DELTA_IDX]}"

ROOT_DIR="data/controlled_orders_ub/test/xl"
OUTPUT_ROOT="results/controlled_orders/test"
AGENT_PATH="data/training_costs_ub/"
DEVICE="cuda"
GENERATE_GANTTS="true"
# ------------------------------------------------------------------

module purge
module load StdEnv/2023 python/3.11 cuda/12.2

source "$HOME/envs/gnrs_rj/bin/activate"

cd "$SLURM_SUBMIT_DIR"

echo "== Tache array $SLURM_ARRAY_TASK_ID (job $SLURM_ARRAY_JOB_ID) sur $(hostname) =="
echo "Root_dir : $ROOT_DIR"
echo "Scenario : $SCENARIO (inst_filter)"
echo "Delta    : $DELTA"
echo "Device   : $DEVICE"
nvidia-smi || true

python acceptation_solver.py \
    --mode all \
    --root_dir "$ROOT_DIR" \
    --output_root "$OUTPUT_ROOT" \
    --inst_filter "$SCENARIO" \
    --deltas "$DELTA" \
    --device "$DEVICE" \
    --agent_path "$AGENT_PATH" \
    --generate_gantts "$GENERATE_GANTTS"

echo "== Scenario $SCENARIO / delta $DELTA termine (tache $SLURM_ARRAY_TASK_ID) =="

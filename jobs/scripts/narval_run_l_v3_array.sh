#!/bin/bash
# ============================================================
# Job SLURM ARRAY (Narval) : instances L de paired_v3, agent v3 no_cmax.
# 24 taches = 6 scenarios x 2 criteres E (total, indiv) x 2 beta (0.1, 0.2) ;
# chaque tache lance les 3 deltas (0.1 0.2 0.5) sur les 60 instances de son scenario.
#
# CPU seulement : le beam search developpe les candidats dans des workers Ray en CPU ;
# avec un agent sur GPU (cuda / mps), les appels du GNN dans ces workers echouent
# silencieusement (candidats perdus, branches "infaisables"). Pas de GPU demande.
#
# Avec --root_dir .../test/l, le "scenario" interne vaut "l" et le dossier
# diff_costs_X_Y est vu comme "inst" -> on filtre avec --inst_filter.
#
# Soumission (depuis la racine du repo) :
#   mkdir -p logs
#   sbatch --array=0 --time=03:00:00 jobs/scripts/narval_run_l_v3_array.sh   # test : 1 tache
#   sbatch jobs/scripts/narval_run_l_v3_array.sh                             # tout
# Suivi :
#   squeue -u $USER
#   sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
#   tail -f logs/gnrs-l-v3-<JOBID>_<TASKID>.out
# ============================================================
#SBATCH --account=def-adhaj
#SBATCH --job-name=gnrs-l-v3
#SBATCH --array=0-23
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --mail-user=hedi.boukamcha.1@ulaval.ca
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
# Pour un mail par tache (24 x debut/fin) au lieu d'un seul pour tout l'array : ajouter ARRAY_TASKS

set -euo pipefail

SCENARIOS=(same_costs_5_5 diff_costs_5_5 same_costs_3_7 diff_costs_3_7 same_costs_7_3 diff_costs_7_3)
CRITERIA=(total indiv)
BETAS=(0.1 0.2)

ID=$SLURM_ARRAY_TASK_ID
SCENARIO=${SCENARIOS[$(( ID % 6 ))]}
CRITERION=${CRITERIA[$(( (ID / 6) % 2 ))]}
BETA=${BETAS[$(( ID / 12 ))]}

module purge
module load StdEnv/2023 python/3.11

source "$HOME/envs/gnrs_rj/bin/activate"
cd "$SLURM_SUBMIT_DIR"

# Ray : un dossier temporaire par tache (evite les collisions entre taches sur le meme noeud).
# Chemin COURT : Ray y cree des sockets AF_UNIX limitees a 107 caracteres
# (OLD : $SLURM_TMPDIR/ray_<job>_<id> -> /localscratch/hedibk.<job>.<id>/ray_... trop long).
export RAY_TMPDIR="/tmp/r_${SLURM_ARRAY_JOB_ID}_${ID}"
mkdir -p "$RAY_TMPDIR"
trap 'rm -rf "$RAY_TMPDIR"' EXIT

echo "== Tache $ID (job $SLURM_ARRAY_JOB_ID) sur $(hostname) =="
echo "Scenario=$SCENARIO | Critere=$CRITERION | Beta=$BETA | commit=$(git rev-parse --short HEAD)"

python acceptation_solver_complete_agent.py --mode all \
    --root_dir "data/controlled_orders_paired_v3/test/l" \
    --inst_filter "$SCENARIO" \
    --output_root "results/paired_v3_no_cmax/test" \
    --analysis_root "analysis_paired_v3_no_cmax" \
    --agent_path "data/training_paired_v3_no_cmax/" \
    --deltas 0.1 0.2 0.5 \
    --betas "$BETA" \
    --criteria "$CRITERION" \
    --device cpu \
    --generate_gantts false \
    --num_cpus "$SLURM_CPUS_PER_TASK"

echo "== Tache $ID terminee =="

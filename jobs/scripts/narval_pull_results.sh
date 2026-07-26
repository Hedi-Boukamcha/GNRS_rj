#!/bin/bash
# ============================================================
# A executer LOCALEMENT sur ton Mac (pas sur Narval, pas via sbatch),
# depuis la racine du repo GNRS_rj, une fois le(s) job(s) termine(s) :
#
#   bash jobs/scripts/narval_pull_results.sh
#
# Rapatrie depuis Narval :
#   - analysis/                     (suivi par git)
#   - results/controlled_orders/    (ignore par git, mais utile en local)
#   - logs/                         (ignore par git, logs SLURM .out/.err)
#
# Ensuite, pour committer/pousser sur GitHub :
#   git status
#   git add analysis/
#   git commit -m "Resultats different_costs_* depuis Narval"
#   git push
# ============================================================
set -euo pipefail

# ---------- A adapter ----------
NARVAL_USER="<ton_identifiant_narval>"     # <-- ton identifiant Alliance (ex: hboukam)
REMOTE_DIR="~/GNRS_rj"                     # <-- chemin du repo sur Narval
# --------------------------------

REMOTE="${NARVAL_USER}@narval.alliancecan.ca"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "== Rapatriement depuis $REMOTE:$REMOTE_DIR =="

rsync -avz --progress "$REMOTE:$REMOTE_DIR/analysis/" "$LOCAL_DIR/analysis/"
rsync -avz --progress "$REMOTE:$REMOTE_DIR/results/controlled_orders/" "$LOCAL_DIR/results/controlled_orders/"
rsync -avz --progress "$REMOTE:$REMOTE_DIR/logs/" "$LOCAL_DIR/logs/"

echo ""
echo "== Termine =="
echo "Verifie avec : git status"
echo "Puis : git add analysis/ && git commit -m '...' && git push"

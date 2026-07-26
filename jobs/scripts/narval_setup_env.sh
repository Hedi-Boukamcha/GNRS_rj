#!/bin/bash
# ============================================================
# Setup de l'environnement Python pour GNRS_rj sur Narval
# (Alliance de recherche numerique du Canada / Compute Canada)
#
# A executer UNE SEULE FOIS sur un noeud de connexion (login node),
# pas via sbatch :
#
#   cd ~/GNRS_rj        # ou l'endroit ou tu as clone/copie le repo
#   bash jobs/scripts/narval_setup_env.sh
#
# Cree un venv persistant dans $HOME/envs/gnrs_rj, installe unique-
# ment via les wheels precompilees de l'Alliance (--no-index), donc
# aucun acces internet requis (les noeuds de calcul n'en ont pas).
# ============================================================
set -euo pipefail

module purge
module load StdEnv/2023 python/3.11 cuda/12.2

ENV_DIR="$HOME/envs/gnrs_rj"

echo "== Verification des wheels disponibles (informatif) =="
avail_wheels torch torch_geometric ray pandas matplotlib numpy || true

echo "== Creation du venv dans $ENV_DIR =="
python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --no-index --upgrade pip

echo "== Installation des dependances (depuis le wheelhouse Alliance) =="
pip install --no-index numpy pandas matplotlib
pip install --no-index torch torchvision torchaudio
pip install --no-index torch_geometric
pip install --no-index ray

deactivate

echo ""
echo "Environnement pret : $ENV_DIR"
echo "Dans le script de job, active-le avec :"
echo "  source $ENV_DIR/bin/activate"

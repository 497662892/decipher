#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time 00-04:00:00
#SBATCH --partition=gpu4_dev
#SBATCH --job-name jupyter-notebook
#SBATCH --output logs/jupyter-notebook-%J.log

# Thank you, Yale Center for Research Computing, for this script  #SBATCH --gres=gpu:1                 # 指定 GPU 数量
# modified for use on BigPurple in NYU Langone Medical Center by Paul Glick

# get tunneling info
XDG_RUNTIME_DIR=""
port=8890
node=$(hostname -s)
user=$(whoami)
#cluster=$(hostname -f | awk -F"." '{print $2}')
case "$SLURM_SUBMIT_HOST" in
bigpurple-ln1)
 login_node=bigpurple1.nyumc.org
 ;;
bigpurple-ln2)
 login_node=bigpurple2.nyumc.org
 ;;
bigpurple-ln3)
 login_node=bigpurple3.nyumc.org
 ;;
bigpurple-ln4)
 login_node=bigpurple4.nyumc.org
 ;;
esac
# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@${login_node}

For more info and how to connect from windows,
   see research.computing.yale.edu/jupyter-nb
Here is the MobaXterm info:

Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ${login_node}
SSH login: $user
SSH port: 22
Use a Browser on your local machine to go to:
http://localhost:${port}  (prefix w/ https:// if using password)
Use the token string from the URL printed below and add it to the URL above
"

if [[ -n "${HOME:-}" && -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

if [[ -n "${CONDA_ENV:-}" ]]; then
  source activate "$CONDA_ENV"
fi

WORKDIR="${WORKDIR:-$PWD}"
cd "$WORKDIR"

if [[ -n "${JUPYTER_MODULES:-}" ]]; then
  # shellcheck disable=SC2086
  module load $JUPYTER_MODULES
fi

jupyter-notebook --no-browser --port=${port} --ip=${node}

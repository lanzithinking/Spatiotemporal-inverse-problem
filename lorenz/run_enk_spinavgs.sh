#!/bin/bash
 
#SBATCH -N 1                        # number of compute nodes
#SBATCH -c 5                        # number of "tasks" (cores)
#SBATCH --mem=64G                   # GigaBytes of memory required (per node)

#SBATCH -p parallel                 # partition 
#SBATCH -q normal                   # QOS

#SBATCH -t 2-12:00                  # wall time (D-HH:MM)
##SBATCH -A slan7                   # Account hours will be pulled from (commented out with double # in front)
#SBATCH -o %x.log                   # STDOUT (%j = JobId)
#SBATCH -e %x.err                   # STDERR (%j = JobId)
#SBATCH --mail-type=END,FAIL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=slan@asu.edu    # send-to address

# load environment
# module load python/3.7.1
module load anaconda3/2020.2

# go to working directory
cd ~/Projects/ST-inverse/code/Lorenz

# run python script
 if [ $# -eq 0 ]; then
	alg_NO=0
	mdl_NO=0
	ensbl_size=100
elif [ $# -eq 1 ]; then
	alg_NO="$1"
	mdl_NO=0
	ensbl_size=100
elif [ $# -eq 2 ]; then
	alg_NO="$1"
	mdl_NO="$2"
	ensbl_size=100
elif [ $# -eq 3 ]; then
	alg_NO="$1"
	mdl_NO="$2"
	ensbl_size="$3"
fi

if [[ ${alg_NO}==0 ]]; then
	alg_name='EKI'
elif [[ ${alg_NO}==1 ]]; then
	alg_name='EKS'
else
	echo "Wrong args!"
	exit 0
fi

if [[ ${mdl_NO}==0 ]]; then
	mdl_name='simple'
elif [[ ${alg_NO}==1 ]]; then
	mdl_name='STlik'
else
	echo "Wrong args!"
	exit 0
fi

python -u run_lorenz_EnK_spinavgs.py ${alg_NO} ${mdl_NO} ${ensbl_size} #> ${alg_name}_${mdl_name}_J${ensbl_size}.log
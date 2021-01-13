yecho 'Path where raisimlib is installed, beginning from "/home". Example: /home/USER/raisimlib:'

read raisim_path

RSG_PATH=$raisim_path'/raisimGymTorch'

echo '# raisimGymTorch Path' >> ~/.bashrc
echo export RSG_PATH=$raisim_path'/raisimGymTorch' >> ~/.bashrc

source ~/.bashrc

cd ..
this_path=`pwd`

mkdir $RSG_PATH/raisimGymTorch/env/envs/rsg_quadcopter
ln -s $this_path/rsg_quadcopter/* $RSG_PATH/raisimGymTorch/env/envs/rsg_quadcopter/

echo RSG_PATH=$raisim_path'/raisimGymTorch' added to ~/.bashrc and rsg_quacopter files linked to raisimGymTorch. Run compile.sh once or after any changes in rsg_quadcopter files before running a run_*.sh script.

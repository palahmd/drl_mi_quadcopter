### Set up Path to raisimGymTorch in ~/.bashrc for linking and compiling


if grep -Fxq "export RSG_PATH=$RSG_PATH" ~/.bashrc
then 
	echo "raisimGymTorch Path is already added to ~/.bashrc"
	RSG_PATH=$RSG_PATH
else
	echo 'Enter path where raisimLib is installed, starting from "/home/..."; Example: /home/USER/raisimLib:'
	read raisim_path
	RSG_PATH=$raisim_path'/raisimGymTorch'
	echo "Adding raisimGymTorch Path to ~/.bashrc"
	echo '## raisimGymTorch Path' >> ~/.bashrc
	echo "export RSG_PATH=$RSG_PATH" >> ~/.bashrc
	source ~/.bashrc
fi


### Link all the algorithm, environments and helper files to raisimGymTorch
cd ..
home_path=`pwd`
echo "Linking algorithm, environments and helper files to raisimGymTorch"

# algorithms
for dir in ./algorithms/*/
do
	dir=${dir%/}
	dir=${dir##*/}
	
	mkdir -p $RSG_PATH/raisimGymTorch/algo/$dir
	ln -s $home_path/algorithms/$dir/* $RSG_PATH/raisimGymTorch/algo/$dir
done

# environments
for dir in ./environments/*/
do
	dir=${dir%/}
	dir=${dir##*/}
	
	mkdir -p $RSG_PATH/raisimGymTorch/env/envs/$dir
	ln -s $home_path/environments/$dir/* $RSG_PATH/raisimGymTorch/env/envs/$dir
done

# helper
for dir in ./helper/*/
do
	dir=${dir%/}
	dir=${dir##*/}
	
	mkdir -p $RSG_PATH/raisimGymTorch/helper/$dir
	ln -s $home_path/helper/$dir/* $RSG_PATH/raisimGymTorch/helper/$dir
done

# Link raisimUnity for linux to current folder
ln -s $RSG_PATH/../raisimUnity/linux $home_path/raisimUnity

echo "Done - ready to compile and run"

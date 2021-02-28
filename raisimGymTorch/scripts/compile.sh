this_path=`pwd`

if grep -Fxq "export RSG_PATH=$RSG_PATH" ~/.bashrc
then
	cd $RSG_PATH
	python3 setup.py develop
	cd $this_path
else
	echo "please initialize first by running init.sh"

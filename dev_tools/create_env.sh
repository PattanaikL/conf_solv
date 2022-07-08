# This script does the following tasks:
# 	- creates the conda
# 	- prompts user for desired CUDA version
# 	- installs required packages with correct cuda versions


# get OS type
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=MacOS;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac
echo "Running ${machine}..."


# request user to select one of the supported CUDA versions
# source: https://pytorch.org/get-started/locally/
PS3='Please enter 1, 2, 3, 4, or 5 to specify the desired CUDA version from the options above: '
options=("9.2" "10.1" "10.2" "11.0" "cpu" "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "9.2")
            CUDA="cudatoolkit=9.2"
            CUDA_VERSION="cu92"
            break
            ;;
        "10.1")
			CUDA="cudatoolkit=10.1"
            CUDA_VERSION="cu101"
            break
            ;;
        "10.2")
			CUDA="cudatoolkit=10.2"
            CUDA_VERSION="cu102"
            break
            ;;
        "11.0")
            CUDA="cudatoolkit=11.0"
            CUDA_VERSION="cu110"
            break
            ;;
        "cpu")
			# "cpuonly" works for Linux and Windows
			CUDA="cpuonly"
			# Mac does not use "cpuonly"
			if [ $machine == "Mac" ]
			then
				CUDA=" "
			fi
            CUDA_VERSION="cpu"
            break
            ;;
        "Quit")
            exit
            ;;
        *) echo "invalid option $REPLY";;
    esac
done

echo "Creating conda environment..."
echo "Running: create -n conf_solv python=3.7.13"
conda create -n conf_solv python=3.7.13 -y

# activate the environment to install pytorch
source activate conf_solv

echo "Checking which python..."
which python

echo "Installing PyTorch with requested CUDA version..."
echo "Running: conda install pytorch $CUDA -c pytorch"
conda install pytorch $CUDA -c pytorch -y

echo "Running: conda install -c pyg pyg==2.0.4"
conda install -c pyg pyg==2.0.4 -y

echo "Running: conda install -c conda-forge pytorch-lightning==1.6.1"
conda install -c conda-forge pytorch-lightning==1.6.1 -y

echo "Running: conda env update -f environment.yml"
conda env update -f environment.yml


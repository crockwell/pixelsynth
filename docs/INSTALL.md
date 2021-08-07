### Installation

General requirements are outlined below. I also shared code I used to install in May 2021 below for reference.

#### Requirements for updated code release

This code environment uses more current libraries, and closely matches results. Code versions used to create paper results is also shared at the bottom. This should be used when using the demo code to match video results [on this webpage](https://crockwell.github.io/pixelsynth).

- Python 3.7
- Pytorch version 1.7.1
- [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md) version 0.4.0
- cv2, matplotlib, mock, tensorboardX
- [Habitat-lab](https://github.com/facebookresearch/habitat-lab) (for Matterport only).
Additionally, this requires some special code be copy-pasted in (linked at the bottom of this page)
- [Habitat-sim](https://github.com/facebookresearch/habitat-sim) (for Matterport only).

##### Linux setup with conda
```
conda create --name pixelsynth python=3.7
conda activate pixelsynth

# installing Pytorch 1.7.1 and Pytorch3D 0.4.0
conda install -c pytorch pytorch=1.7.1 torchvision
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e . && cd ..

# installing remaining requirements
conda install -c conda-forge matplotlib
conda install mock
conda install opencv
conda install -c conda-forge tensorboardx
conda install -c anaconda ipython
pip install pytorch_fid

# Matterport only:
# installing habitat-lab and habitat-sim
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab && pip install -r requirements.txt 
python setup.py develop --all & cd ..
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim && pip install -r requirements.txt
conda install habitat-sim headless -c conda-forge -c aihabitat
```

##### Additional Step (Matterport Training Only)
Loading Matterport during training relies on a custom vector environment function in habitat. 

Replace `habitat-lab/habitat/core/vector_env.py` with [this code](https://github.com/crockwell/pixelsynth/blob/master/utils/custom_habitat_vector_env.py). 
For details, see [here](https://github.com/facebookresearch/synsin/issues/2).

#### Requirements used by paper code

This version of the environment uses some packages from ~May 2020 and is used in the paper. This environment should be used when using the demo code to approximately match video results [on this webpage](https://crockwell.github.io/pixelsynth).

- Python 3.7.6
- Pytorch version 1.4.0
- [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md) version 0.2.0
- cv2 version 4.2.0.34, matplotlib, mock, tensorboardX
- [Habitat-lab](https://github.com/facebookresearch/habitat-lab) (for Matterport only) - version 0.1.5.
Additionally, this requires some special code be copy-pasted in (linked at the bottom of this page)
- [Habitat-sim](https://github.com/facebookresearch/habitat-sim) (for Matterport only) - version 0.1.5.

```
conda create --name pixelsynth_replicate python=3.7.6
conda activate pixelsynth_replicate

# installing Pytorch 1.4.0 and Pytorch3D 0.2.0
conda install -c pytorch pytorch=1.4.0 torchvision
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && git checkout e3819a49dfa855de1a7c99c0583fb69f9bdad75b
pip install -e . && cd ..

# installing remaining requirements
conda install -c conda-forge matplotlib
conda install mock
conda install opencv
conda install -c conda-forge tensorboardx
conda install -c anaconda ipython
pip install pytorch_fid
pip install opencv-python==4.2.0.34

# Matterport only:
# installing habitat-lab and habitat-sim
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab && git checkout 96243e6fdae1ba3ee8aae30edce9c18998515773
pip install -r requirements.txt 
python setup.py develop --all & cd ..
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim && git checkout 020041d75eaf3c70378a9ed0774b5c67b9d3ce99
pip install -r requirements.txt
conda install habitat-sim headless -c conda-forge -c aihabitat
```

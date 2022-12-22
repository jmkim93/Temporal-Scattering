## Unidirectional scattering with spatial homogeneity using photonic time disorder

This repository contains the code and data for preprint [arXiv:2208.11884](https://arxiv.org/abs/2208.11884), which is to be published in XXX.


### Original Environment
* Ubuntu 20.04 LTS
* AMD Ryzen 3950X 16-core processor with 128 GB RAM 

### Prerequisite
* Python 3.9.6
* Numpy 1.20.3  
* Matplotlib 3.4.2
* Spyder 5.0.5
* Pandas 1.3.1

---

### Usage

TMM.py gives a preliminary class definition for time-varying systems and related wave-matter interaction. 

Every .py file for each figure includes:
* definition of time-varying system with the corresponding electromagnetic scattering calculation,
* data acquisition for ensembles (if needed),
* and plotting figure,  

which are separated by a few of cells. Running cells successively will finally result in the output figure in pdf format.

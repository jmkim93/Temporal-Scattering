## Unidirectional scattering with spatial homogeneity using corelated photonic time disorder

This repository contains codes for the numerical analysis and the data visualization of the results in [J. Kim et al., Nat. Phys. (2023)](https://doi.org/10.1038/s41567-023-01962-3).


### Original Environment
* Ubuntu 20.04 LTS
* AMD Ryzen 3950X 16-core processor with 128 GB RAM 

### Dependencies
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

which are separated by a few of cells. Running cells successively will finally result in the output figure and its source data in pdf, xlsx formats, respectively.


### Ensemble Data
can be downloaded [Google Drive](https://drive.google.com/drive/folders/1xI2q6jY6WhD9Nc9Y8Xahh1r1poYFHJnH?usp=sharing).

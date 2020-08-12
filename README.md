# Beam

The beam package was originally built using anaconda. If you don't have it, download the python 3 version from https://www.anaconda.com/distribution/ and install it. You can use miniconda if you don't want to download the entire anaconda distribution, available here https://docs.conda.io/en/latest/miniconda.html.

Once you have everything installed, go to the directory you want to place the code in and run:
```
git clone https://github.com/CU-PWFA/Beam.git
```

Navigate to the top level directory of the newly downloaded repository and run:
```
conda env create -f beam.yml
```
This wil create a virtual enviornment named beam.

Activate the new environment using:
```
conda activate beam
```

Run the compile script to build the Python directories:
```
python calc/setup.py build_ext --inplace
```

If all goes well everything will complete successfully. 

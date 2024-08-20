# Using MC-ML #

MOCAT-MC and MOCAT-ML Integrated Approach to Predict Space Object Density Distributions

To run both mocat-mc and mocat-ml in one script, run combined_script.py

Before doing so, make edits to all paths in combined_script.py, mocat_mc_wrapper.m, and setup_MCconfig.m(Will be streamlined soon so this will be less tedious)

In the main function of combined_script.py, you must designate the path for the model you would like to run, the initial conditions for MC, and the size you would like to reshape to for the model (Ex. (1, 128, 32, 32))

All dependencies for both mocat-ml and mocat-mc are required

Windows is currently experiencing some issues do to posixpath and fastai so Linux is the best way to run it

Any edits to the variables for mocat_mc will be done like normal in the setup_MCconfig.m file


# Required module #

Python, numpy, fastai, tsai, h5py, torch, torchvision

GPU-access/CUDA

# Related works #

 [MOCAT-MC](https://github.com/ARCLab-MIT/MOCAT-MC?tab=readme-ov-file#charles-rinos-orbit-computation-code) 
 
 [MOCAT-SSEM](https://github.com/ARCLab-MIT/MOCAT-SSEM)


# MIT License #
Copyright (c) 2023 ARCLab

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
OR OTHER DEALINGS IN THE SOFTWARE.



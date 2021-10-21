# BIAS: Bias In Algorithms, Structural
## A toolbox for detecting structural bias in continuous optimization heuristics

## Setup
This package requires an R-installation to be present, with the following packages installed:
- PoweR
- AutoSEARCH
- nortest

Note: The code for generating the RF used to predict the type of bias is included, but the full RF is not. These can be found on zenodo: https://doi.org/10.6084/m9.figshare.16546041. Note that in order to generate these RFs, the raw data of rejections (also on the same zenodo page) is needed, and the paths to the data should be modified accordingly.
# BIAS: Bias In Algorithms, Structural
## A toolbox for detecting structural bias in continuous optimization heuristics.

## Setup

This package requires an R-installation to be present, with the following packages installed:
- PoweR
- AutoSEARCH
- nortest

### Detailed setup

1. Download and install R from https://cran.r-project.org/
2. Install R packages

    ```R
    install.packages("PoweR")
    install.packages("AutoSEARCH")
    install.packages("nortest")
    ```
3. Download this repository (clone or as zip)
4. Create a python virtual env `python -m venv env`
5. Activate the env (in powershell for example: `env/Scripts/Activate.ps1 `)
6. Install dependencies `pip install -r requirements.txt`
7. Checkout the `example.py` to start using the BIAS toolbox.


### Additional files

Note: The code for generating the RF used to predict the type of bias is included, but the full RF is not. These can be found on zenodo: https://doi.org/10.6084/m9.figshare.16546041. Note that in order to generate these RFs, the raw data of rejections (also on the same zenodo page) is needed, and the paths to the data should be modified accordingly.
# BIAS: Bias In Algorithms, Structural
## A toolbox for detecting structural bias in continuous optimization heuristics.

## Setup

This package requires an R-installation to be present, with the following packages installed:
- PoweR
- AutoSEARCH
- nortest
- data.table
- goftest
- ddst

### Detailed setup

1. Download and install R from https://cran.r-project.org/
2. Install R packages

    ```R
    install.packages("PoweR")
    install.packages("AutoSEARCH")
    install.packages("nortest")
    install.packages("data.table")
    install.packages("goftest")
    install.packages("ddst")
    ```
3. Download this repository (clone or as zip)
4. Create a python virtual env `python -m venv env`
5. Activate the env (in powershell for example: `env/Scripts/Activate.ps1 `)
6. Install dependencies `pip install -r requirements.txt`
7. Checkout the `example.py` to start using the BIAS toolbox.


## Example

```py
#example of using the BIAS toolbox to test a DE algorithm

from scipy.optimize import differential_evolution
import numpy as np
from BIAS import BIAS, f0

bounds = [(0,1), (0, 1), (0, 1), (0, 1), (0, 1)]

#do 30 independent runs (5 dimensions)
samples = []
print("Performing optimization method 30 times of f0.")
for i in np.arange(30):
    result = differential_evolution(f0, bounds, maxiter=100)
    samples.append(result.x)

samples = np.array(samples)

test = BIAS()
print(test.predict(samples, show_figure=True))
```

## Additional files

Note: The code for generating the RF used to predict the type of bias is included, but the full RF is not. These can be found on zenodo: https://doi.org/10.6084/m9.figshare.16546041. Note that in order to generate these RFs, the raw data of rejections (also on the same zenodo page) is needed, and the paths to the data should be modified accordingly.
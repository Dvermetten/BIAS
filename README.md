# Deep-BIAS: Bias In Algorithms, Structural
## A toolbox for detecting structural bias in continuous optimization heuristics.

With a deep-learning extension to better evaluate the type of bias and gain insights using explainable AI



## Using the BIAS-Toolbox with Docker (Recommended)

The BIAS-Toolbox can be used inside a Docker container, eliminating the need to manually install all dependencies and packages. Follow the steps below to run the Docker image, and to start working with the toolbox in a Jupyter notebook environment. We provide the following prebuild container: `ghcr.io/nikivanstein/bias:master`

### Prerequisites

Make sure you have Docker installed on your system. You can install Docker by following the instructions [here](https://docs.docker.com/get-docker/).

### Steps to Run the Docker Image

1. **Pull the Prebuild Image**
   The following command will pull the prebuild image to your system.

   ```bash
   docker pull ghcr.io/nikivanstein/bias:master
   ```

2. **Run the Prebuild Docker Container**:
   The following command will start the container and expose the Jupyter notebook interface on port `8888`:
   
   ```bash
   docker run -p 8888:8888 ghcr.io/nikivanstein/bias:master
   ```

3. **Access the Jupyter Notebook**:
   After starting the container, you should see a message with instructions to access the Jupyter notebook. It will look something like this:
   
   ```
   To access the notebook, open this file in a browser:
       http://127.0.0.1:8888/?token=<token>
   ```
   
   Open the provided URL in your web browser to start using the BIAS-Toolbox within Jupyter.


### Steps to Build the Dockerfile yourself

1. **Clone the Repository**:
   If you haven't already cloned the BIAS repository, do so with the following command:
   
   ```bash
   git clone https://github.com/nikivanstein/BIAS.git
   cd BIAS
   ```

2. **Build the Docker Image**:
   The `Dockerfile` included in this repository will install all necessary dependencies (both Python and R), download required data and model files, and set up the environment.

   To build the Docker image, run the following command from the root of the repository (where the `Dockerfile` is located):
   
   ```bash
   docker build -t bias-toolbox .
   ```

   This will create a Docker image named `bias-toolbox`.

3. **Run the Docker Container**:
   Once the image is built, you can run the container. The following command will start the container and expose the Jupyter notebook interface on port `8888`:
   
   ```bash
   docker run -p 8888:8888 bias-toolbox
   ```

4. **Access the Jupyter Notebook**:
   After starting the container, you should see a message with instructions to access the Jupyter notebook. It will look something like this:
   
   ```
   To access the notebook, open this file in a browser:
       http://127.0.0.1:8888/?token=<token>
   ```
   
   Open the provided URL in your web browser to start using the BIAS-Toolbox within Jupyter.


### Stopping the Container

To stop the running Docker container, press `CTRL+C` in the terminal where the container is running, or find the container's ID with the command:

```bash
docker ps
```

Then stop the container with:

```bash
docker stop <container_id>
```

### Additional Notes

- The image is configured to use Jupyter Notebook with R and Python integrations.
- R version `4.1.2` is installed and configured along with the necessary R packages as specified in the `install.r` script.
- Python dependencies are handled via the `requirements.txt` file.

By using Docker, you can avoid issues related to dependency installation and system setup, providing a consistent environment for running the BIAS-Toolbox.


## Setup using Pip

Another way of using the BIAS-Toolbox is by installing the pip package.

This package requires an R-installation to be present.

The package is tested with R 4.1.2 (install from source https://cran.r-project.org/src/base/R-4/R-4.1.2.tar.gz)

The R packages will be installed automatically upon first importing BIAS.

Install the BIAS toolbox using pip:

    pip install struct-bias

This installs the following R packages:

- PoweR
- AutoSEARCH
- nortest
- data.table
- goftest
- ddst


### Detailed setup using virtual env

1. Download and install R from https://cran.r-project.org/ use version 4.1.2  
   Example for Ubuntu based system:
    ```sh
    sudo wget https://cran.rstudio.com/src/base/R-4/R-4.1.2.tar.gz  
    tar zxvf R-4.1.2.tar.gz  
    cd R-4.1.2  
    ./configure --enable-R-shlib --with-blas --with-lapack
    make  
    sudo make install  
    ```
    
2. Download this repository (clone or as zip)
3. Create a python virtual env `python -m venv env`
4. Activate the env (in powershell for example: `env/Scripts/Activate.ps1 `)
5. Install dependencies `pip install -r requirements.txt`
6. Checkout the `example.py` to start using the BIAS toolbox.


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

y, preds = test.predict_deep(samples)
test.explain(samples, preds, filename="explanation.png")
```

## Additional files

Note: The code for generating the RF used to predict the type of bias is included, but the full RF is not. These can be found on zenodo: https://doi.org/10.6084/m9.figshare.16546041.
The RF models will be downloaded automatically the first time the predict function requires them.

### Citation

If you use the BIAS toolbox in a scientific publication, we would appreciate using the following citations:

```
@ARTICLE{9828803,
  author={Vermetten, Diederick and van Stein, Bas and Caraffini, Fabio and Minku, Leandro L. and Kononova, Anna V.},
  journal={IEEE Transactions on Evolutionary Computation}, 
  title={BIAS: A Toolbox for Benchmarking Structural Bias in the Continuous Domain}, 
  year={2022},
  volume={26},
  number={6},
  pages={1380-1393},
  doi={10.1109/TEVC.2022.3189848}
}

@software{niki_van_stein_2023_7803623,
  author       = {Niki van Stein and
                  Diederick Vermetten},
  title        = {Basvanstein/BIAS: v1.1 Deep-BIAS Toolbox},
  month        = apr,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v1.1},
  doi          = {10.5281/zenodo.7803623},
  url          = {https://doi.org/10.5281/zenodo.7803623}
}
```

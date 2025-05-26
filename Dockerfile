# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building R and other tools, including X11 libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libreadline-dev \
    wget \
    curl \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libpcre2-dev \
    libpcre3-dev \
    gfortran \
    libx11-dev \
    libxt-dev \
    x11proto-core-dev \
    libcairo2-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Download and install R 4.1.2
RUN wget https://cran.rstudio.com/src/base/R-4/R-4.1.2.tar.gz && \
    tar zxvf R-4.1.2.tar.gz && \
    cd R-4.1.2 && \
    ./configure --enable-R-shlib --with-blas --with-lapack && \
    make && \
    make install && \
    cd .. && \
    rm -rf R-4.1.2 R-4.1.2.tar.gz


# Copy the current directory contents into the container at /app
COPY ./BIAS /app/BIAS

# Install R packages (add any necessary R packages here)
RUN Rscript /app/BIAS/install.r

ENV R_PACKAGES_INSTALLED=Yes

# Copy example files
COPY example.py /app/example.py
COPY requirements.txt /app/requirements.txt
COPY setup.py /app/setup.py
COPY README.md /app/README.md

# Install Python dependencies specified in requirements.txt
RUN pip install --upgrade pip
#RUN python setup.py install
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y zip unzip

# Download reference value files
# Download and unzip the files from figshare
RUN wget https://figshare.com/ndownloader/files/30591411 -O bias_data.zip && \
    unzip bias_data.zip -d /app/BIAS/data/ && \
    rm bias_data.zip

RUN wget https://figshare.com/ndownloader/files/43106839 -O bias_models.zip && \
    mkdir -p /app/BIAS/models/ && \
    unzip bias_models.zip -d /app/BIAS/models/ && \
    rm bias_models.zip

# Install any additional dependencies for Jupyter notebooks
RUN pip install jupyter

# Set environment variables for R libraries
ENV R_HOME=/usr/local/lib/R
ENV LD_LIBRARY_PATH=/usr/local/lib/R/lib:/usr/local/lib/R/modules:$LD_LIBRARY_PATH

# Copy tutorial file (last such that we can update it easily)
COPY Tutorial.ipynb /app/Tutorial.ipynb

# Expose the port that Jupyter will run on
EXPOSE 8888

# Add a script to start Jupyter automatically when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# Optional: Add a health check
#HEALTHCHECK --interval=30s CMD curl --fail http://localhost:8888 || exit 1

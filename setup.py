import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = "1.3.2"
gh_ref = os.environ.get("GITHUB_REF")
if gh_ref:
    *_, tag = gh_ref.split("/")
    __version__ = tag.replace("v", "")

setuptools.setup(
    name='struct-bias',
    version=__version__,
    author="Diederick Vermetten, Niki van Stein",
    author_email="d.l.vermetten@liacs.leidenuniv.nl",
    description="BIAS toolbox: Structural bias detection for continuous optimization algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={
        'BIAS': ['install.r', 'models/*'],
    },
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'tensorflow',
        'shap',
        'rpy2',
        'scipy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'statsmodels',
        'regex',
        'autokeras'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

import os
import rpy2.robjects as robjects


# get the value of the environment variable HOME
R_installed = os.getenv("R_PACKAGES_INSTALLED")

if R_installed != "Yes":
    dirname = os.path.dirname(__file__)
    robjects.r.source(f"{dirname}/install.r", encoding="utf-8")
    os.environ["R_PACKAGES_INSTALLED"] = "Yes"

from .SB_Toolbox import BIAS, f0

__all__ = ("BIAS", "f0")

#!/usr/bin/env python
# -------------------------
# Installer
# -------------------------
import subprocess, sys, os, argparse

def install_requirements(req_file):
    if not os.path.isfile(req_file):
        print(f"Requirements file not found: {req_file}", file=sys.stderr)
        return False
    command = [sys.executable, '-m', 'pip', 'install', '-r', req_file]
    print("Running:", " ".join(command))
    try:
        subprocess.check_call(command)
        return True
    except subprocess.CalledProcessError as e:
        print("Installation error:", e, file=sys.stderr)
        return False
    
parser = argparse.ArgumentParser(description="Install packages from a requirements file.")
parser.add_argument("-f", "--file", default="requirements.txt", help="Path to requirements file")
args = parser.parse_args()

if not install_requirements(args.file):
    sys.exit(1)
print("Installation completed successfully.")    

    
# -------------------------
# Script start
# -------------------------    
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib #%pip install joblib

# For PCA and classification
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, 
                             roc_auc_score, confusion_matrix, roc_curve)

import statsmodels.api as sm

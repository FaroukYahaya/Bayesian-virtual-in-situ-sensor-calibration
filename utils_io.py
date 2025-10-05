#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Farouk Yahaya
Contact: faroya2011@gmail.com
Date: 2025-10-04

Title: Utilities for I/O, logging, and metrics

Dataset citation:
Granderson, J., Lin, G., Chen, Y., Casillas, A., Im, P., Jung, S., Benne, K., 
Ling, J., Gorthala, R., Wen, J., Chen, Z., Huang, S., & Vrabie, D. (2022).
LBNL Fault Detection and Diagnostics Datasets. [Data set].
Open Energy Data Initiative (OEDI), Lawrence Berkeley National Laboratory.
https://doi.org/10.25984/1881324
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np

def setup_logger(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(log_file.stem)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

def mape(y, yhat, eps: float = 1e-8) -> float:
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.mean(np.abs((y - yhat) / (np.abs(y) + eps))) * 100)

def rmse(y, yhat) -> float:
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.sqrt(np.mean((y - yhat)**2)))

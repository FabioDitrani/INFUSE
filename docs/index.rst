.. SHINE documentation master file, created by
   sphinx-quickstart on Wed Dec 11 17:33:24 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


==========================
Documentation of INFUSE
==========================

**Authors:** Fabio Rosario Ditrani 

**Date:**  14/12/2025

.. contents::
   :depth: 2
   :local:


Introduction
============

**Project Name:** ``INFUSE``

``INFUSE`` (Full-INdex Fitting for Uncovering Stellar Evolution) is a Python-based tool designed to infer stellar population properties
from observed spectra of galaxies and stellar systems. 
The code performs a **full-index fitting** analysis by comparing
observed spectral indices with predictions from
**stellar population synthesis (SPS) models** (e.g. **sMILES**).
``INFUSE`` uses a **nested sampling algorithm** to efficiently explore
the multi-dimensional parameter space and to derive robust posterior
probability distributions for the stellar population parameters.

Installation
============

**Requirements:**

- Python version: ``>=3.8``
- Dependencies: ``numpy``, ``corner``, ``astropy``, ``ultranest``, ``matplotlib``

**Steps to Install:**

1. Clone the repository:
   ::

       git clone https://github.com/FabioDitrani/INFUSE.git
       cd INFUSE

2. Install the code:
   ::

       python -m pip install .


Directory Contents
==================
The github distribution includes a INFUSE/ directory that contains the following codes:

- ``INFUSE.py``: The main Python file containing the code for the extraction process.
- ``GUI_SHINE.py``: The Python file containing the code for the GUI.
- ``Make_Im_SHINE.py``: The Python file for the tool used to analyze the extraction results. It generates 2D images.


Usage of INFUSE and tools
=================
``SHINE`` and ``Make_Im_SHINE`` are installed as executables and can be run directly from the terminal. Below is an explanation of the different ways to execute the code.

The extraction is performed using ``SHINE``. The basic idea behind the code is as follows:


- Reading and handling **observed spectra**
- Loading **stellar population synthesis models** (e.g. sMILES)
- Measurement and selection of **spectral indices**
- Full-index fitting between models and observations
- Bayesian inference using **nested sampling**
- Diagnostic plots to inspect **posterior distributions**
- Control plots comparing **observed and best-fit spectral indices**


Scientific Method
-----------------

INFUSE implements a generalised full-index fitting approach based on
a **pixel-by-pixel flux comparison** restricted to selected
spectral features.

For each spectral index, both the observed spectrum and the SPS model
spectra are **normalised to the pseudo-continuum** defined by the
classical index bandpasses. The fitting is then performed by comparing
the **flux values at each wavelength pixel** within the index
feature window, rather than by directly comparing integrated index
measurements.

The likelihood function is constructed by evaluating the agreement
between the continuum-normalized observed and model fluxes within the
selected index regions, taking into account the observational
uncertainties at each pixel.

This approach preserves the conceptual framework of classical
absorption-line index analysis, while extending it to a more flexible
and information-rich fitting scheme.

The posterior distribution of the model parameters is sampled using
a nested sampling approach, allowing for:
- efficient parameter space exploration
- evidence computation
- robust uncertainty estimates

Applications
------------

INFUSE is suited for:

- Stellar population analysis of galaxies
- studies of age, metallicity, and abundance patterns
- comparison of different SPS model libraries
- testing spectral diagnostics based on absorption-line indices


Changelog
=========

.. include:: ../CHANGELOG




Contributing
============

If you are interested in contributing to the project, please contact us and follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your feature/bugfix.
3. Submit a pull request.




API
===

None

License
=======

Copyright (C) 2024 The Authors
  
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.  
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License.


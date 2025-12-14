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

``INFUSE`` (Full-INdex Fitting for Uncovering Stellar Evolution) is a simple Python-based script that identifies connected structures above a user-given signal-to-noise (S/N) threshold in 2D and 3D datasets.
It also allows masking bad regions where extraction should not be performed.

Installation
============

**Requirements:**

- Python version: ``>=3.8``
- Dependencies: ``numpy``, ``scipy``, ``astropy``, ``connected-components-3d``

**Steps to Install:**

1. Clone the repository:
   ::

       git clone https://github.com/matteofox/SHINE.git
       cd SHINE

2. Install the code:
   ::

       python -m pip install .


Directory Contents
==================
The github distribution includes a shine/ directory that contains the following codes:

- ``SHINE.py``: The main Python file containing the code for the extraction process.
- ``GUI_SHINE.py``: The Python file containing the code for the GUI.
- ``Make_Im_SHINE.py``: The Python file for the tool used to analyze the extraction results. It generates 2D images.


Usage of SHINE and tools
=================
``SHINE`` and ``Make_Im_SHINE`` are installed as executables and can be run directly from the terminal. Below is an explanation of the different ways to execute the code.

The extraction is performed using ``SHINE``. The basic idea behind the code is as follows:

1. (Optional, applicable to 3D data only) Select a portion of the cube (in the z-direction or wavelength direction) where the user wants to focus the extraction.
2. Mask certain voxels using a user-provided mask (e.g., continuum sources).
3. Spatially filter the cube/image and the associated 2D or 3D variance using a user-defined kernel dimension.
4. Apply a threshold to the cube/image based on the user-defined S/N threshold.
5. Group connected voxels (3D)/pixels (2D) that meet the S/N threshold and other user-defined parameters.
6. Generate and save the catalog along with the labeled cube/image.

For 3D data only, it is possible to use Make_Im_SHINE to create the associated image by collapsing the voxels in the z-direction using the labeled cube.
This tool is designed to create three different types of images (``flux``, ``mean`` or ``median``) both selecting only certain voxels based on a 3D mask with associated Ids (e.g. the output labels cube of ``SHINE``, thus creating an extraction image) and all the voxels (thus creating a narrow band image). If ``flux`` is selected, the units of the output image are :math:`1 \times 10^{-18} \, \mathrm{erg \, s^{-1} \, cm^{-2} \, arcsec^{-2}}`.
Using a single Id object it is also possible to obtain an image with the representative pseudo narrow band around it (the width is specified by the user with ``--nsl`` and ``--nsladd``.


.. _changelog:

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


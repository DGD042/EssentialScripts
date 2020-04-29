# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#______________________________________________________________________________
#______________________________________________________________________________

'''

This package uses functions from Matlab to run models made in COMSOL, it is 
necessary to have access to the main folder of COMSOL to run the algorithms
in Matlab.

This package can also open the information from exported files and use them
to generate new data. Although this package is focused on flow through 
porosity media in 2D right now, it can be use widely to other applications.
____________________________________________________________________________
This class is of free use and can be modify, if you have some 
problem please contact the programmer to the following e-mails:

- daniel.gonzalez@vanderbilt.edu
- danielgondu@gmail.com 
____________________________________________________________________________
'''
from setuptools import setup

setup(
    name="pyDGDutil",
    version="1.0.1",
    author="Daniel González Duque",
    description="Complementary scripts of other codes",
    license="MIT",
    packages=["pyDGDutil"],
    pyhon_requires='>=3.6'
)

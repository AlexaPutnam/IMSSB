#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 13:39:52 2020

@author: alexaputnam

Author: Alexa Angelina Putnam Shilleh
Institution: University of Colorado Boulder
Software: sea state bias model generator (interpolation method)
Thesis: Sea State Bias Model Development and Error Analysis for Pulse-limited Altimetry
Semester/Year: Fall 2021
"""
import numpy as np

import main_matrices2models_ext as mm2m
import lib_dict_ext as ldic
####################################################################
'''
~~~Input~~~~~~
        MISSIONNAME: input string corresponding to mission name
            options: 
                'tx' for Topex/Poseidon
                'j1' or 'j2' or 'j3' for Jason missions 1-3
                's6A' for Sentinel-6 Michael Freilich
                
        MNtag: define year of data used to create model
            ** see README_MNtag
        
        METHOD: input string corresponding to data method
            options:
                'dir': for an alongtrack (DIR or COE) model
                'xov': for crossover (XOV) model
                'col': for collinear (COL) model
                'jnt': for joint (JNT) model
                'coe': for coestimated (COE) model
                
        SOURCE: input string corresponding to altimeter data source
            options:
                "jpl" for data received from JPL files
                "aviso" for data received from AVISO
                "rads" for data received from RADS software
            ** additional sources can be added in lib_dict_ext.py/product_paths
                if a new source is added then be sure to make a new directory
                corresponding to the source name in ~/imssb_software/output/mn_template/product2data/
                
               
        DIM: input string corresponding to second dimension (the first dimension = SWH)
            default: DIM = 'u10'
            options:
                altimeter wind speed: 'u10'
                backscatter coefficient: 'sig0'
            ** more options can be added in lib_geo_gridding_ext.py/grid_dimensions
                this function allows you to define a new parameter, as well as
                resize the current parameters.
                    i.e. dx and x_grid define the bin spacing and nodes for the 
                    SWH axis, respectively.
            
        OMIT: enter string to define a third dimension, or leave as empty bracket
            default (2D model): OMIT = []
            options for a 3D model: 
                mean wave period: 'mean_wave_period_t02'
            ** more options can be added in lib_geo_gridding_ext.py/grid_dimensions
                this function allows you to define a new parameter, as well as
                resize the current parameters.
                    i.e. dz and z_grid define the bin spacing and nodes for the 
                    Z axis, respectively.
            
        WEIGHT: True or False
            default: WEIGHT = False
            options:
                True: weight measurements based on latitude
                False: do not weight measurements based on latitude

        OPEN_OCEAN: True or False
            default: OPEN_OCEAN = True
            options:
                True: open ocean filter applied (depth<-300 and distance to coast > 50 km)
                False: coast filter applied (coast < 50 km)
'''
#### Define Input Parameters ####
MISSIONNAME = 'j3'
METHOD = 'xov'
MNtag = 'y456'#'y3456w'# 'y5'#'y6'
SOURCE = 'rads'
DIM = 'u10'
OMIT = []
WEIGHT = True
setSSB = False # fix high densitybin to specific value
OPEN_OCEAN = True 
VAR='usla'#'hemi_north'#'usla'
#### Run step-3 ####
print('__________________________________________')
print('model for '+MISSIONNAME+' '+METHOD)
print('__________________________________________')
CYC,badcyc = ldic.mission_cycles(MISSIONNAME,MNtag)
# Create observation matrices
if 'col' in METHOD:
    badc = np.append(badcyc,badcyc-1)
    print('METHOD==col')
else:
    badc = np.copy(badcyc)
    print('METHOD not col')
CYCLES = np.setdiff1d(CYC,badc)
if 'col' in METHOD:
    CYCLES = CYCLES[:-1]
print('MISSIONNAME,METHOD')
print(MISSIONNAME,METHOD)
print('SOURCE,DIM,OMIT')
print(SOURCE,DIM,OMIT)
print('CYCLES,WEIGHT')
print(CYCLES,WEIGHT)

x_grid,y_grid,z_grid = mm2m.run_matrices2models(MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,OMIT,DIM,WEIGHT,ERR=[],setSSB=setSSB,OPEN_OCEAN=OPEN_OCEAN) 

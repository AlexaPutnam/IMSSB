#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:58:00 2020

@author: alexaputnam

Author: Alexa Angelina Putnam Shilleh
Institution: University of Colorado Boulder
Software: sea state bias model generator (interpolation method)
Thesis: Sea State Bias Model Development and Error Analysis for Pulse-limited Altimetry
Semester/Year: Fall 2021
"""
import numpy as np

import main_data2matrices_ext as md2m
import lib_dict_ext as dic
 
'''
Step-2: Filter and convert along-track cycle files to observation files 

~~~~Input~~~~~~
        MISSIONNAME: input string corresponding to mission name
            options: 
                'tx' for Topex/Poseidon
                'j1' or 'j2' or 'j3' for Jason missions 1-3
                's6A' for Sentinel-6 Michael Freilich
                
        cyc: an array containing the numbers for each cycle that will be processed
            i.e. cyc = np.arange(48,101)
            (Bad cycles will be removed. This can be check in lib_dict_ext.py/bad_cycles)
        
        METHOD: input string corresponding to data method
            options:
                'dir': for direct (along-track) measurements
                'xov': for crossover differences
                'col': for collinear differences
            ** 'dir' files must first be created before generating 'xov' and 
                'col' files.
                
        SOURCE: input string corresponding to altimeter data source
            options:
                'jpl' for data received from JPL files
                'aviso' for data received from AVISO
                'rads' for data received from RADS software
            ** additional sources can be added in lib_dict_ext.py/product_paths
                if a new source is added then be sure to make a new directory
                corresponding to the source name in ~/imssb_software/output/mission_name/product2data/
                
        ERR: enter string to define independent-variable error/analysis, or leave as empty bracket
            options:
                nominal model, no added error: []
                tandem analysis: 'j3s6A','j2j3','j1j2', or 'txj1'
                    e.g. jason-2 and jason-3 tandem analysis = 'j2j3'
                create mle3 SSB model: 'mle3'
                add noise for error analysis:'rms_w','rms_s' or 'rms_'ws'
                    ** _w = added SWH error, _s = added sig0 error
                smooth SWH measurements more than 1Hz: 'smth_swh'
                pseuro low resolution mode (plrm) analysis: 'plrm'
             ** more options can be added in lib_data_ext.py/pull_observables
             
        GEO: True or False
            options:
                True: generate observation matrices and vectors required for all
                    model types (i.e. DIR, XOV, COL, JNT and COE), as well as 
                    additional matrices required for coestimated (COE) model
                    See thesis for details.
                False: only generate observaion matrices and vectors required
                    for direct (DIR), collinear (COL), crossover (XOV) and 
                    joint (JNT) models. See thesis for details.
                    
        MSS: True or False
            options:
                True: interpolate additional MSS (CNES 2011 and 2015) solutions
                False: do not add new MSS solution
            ** currently interpolates CNES 2011 and 2015 solutions, but this can be changed
                in lib_geo_gridding_ext.py/spline_interp_mss and main_data2matrices.py/data2matrices
               
        DIM: input string corresponding to second dimension (the first dimension = SWH)
            options:
                altimeter wind speed (nominal): 'u10'
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
'''
#### Define Input Parameters ####
MISSIONNAME = 'j3'#'s6Ah'#'tx'
METHOD = 'xov'
SOURCE = 'rads'
ERR=[]#'txj1'
GEO = True
MSS = False
DIM = 'u10' 
OMIT = []

if MISSIONNAME == 'tx': # Topex models: side A (cycles 48-100), side B (cycles 280-364)
    cyc = np.arange(139,364)#np.arange(48,101)
elif MISSIONNAME == 'j1':
    cyc = np.arange(23,34)#np.hstack((np.arange(33,120),np.arange(150,240)))#np.arange(1,23) #np.arange(1,23)
elif MISSIONNAME == 'j2': # Jason-2 2009 model (cycles 19-55)
    cyc = np.hstack((np.arange(24,56),np.arange(93,281))) #np.arange(56,93)#np.arange(281,307)
elif MISSIONNAME == 'j3':
    cyc = np.arange(107,217)#(217,229)
elif MISSIONNAME=='s3':
    cyc = np.arange(1,54)
elif MISSIONNAME in ['s6A','s6Ah']:
    cyc = np.arange(5,44) #5-10,13,14,20,21,22,23
elif MISSIONNAME =='sw':
    cyc = np.arange(102,279) #301,306)#(174,278) #5-10,
elif MISSIONNAME =='sa':
    cyc = np.arange(36,48)
#### Run step-2 ####
badcyc = dic.bad_cycles(MISSIONNAME)
CYCLES = np.setdiff1d(cyc,badcyc)
print('MISSIONNAME,METHOD,SOURCE,ERR,GEO,MSS,DIM,OMIT')
print(MISSIONNAME,METHOD,SOURCE,ERR,GEO,MSS,DIM,OMIT)
print('CYCLES')
print(CYCLES)
DS_obs2lut = md2m.run_data2matrices(MISSIONNAME,METHOD,SOURCE,CYCLES,OMIT,DIM,ERR=ERR,GEO=GEO,MSS=MSS)




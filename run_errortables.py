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
                
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ERROR-ANALYSIS-SPECIFIC
        ERR: define independent-variable error/analysis, or leave as empty bracket
            options:
                nominal model, no added error: [] (idential to nominal SSB solution if VAR='usla')
                tandem analysis: 'j3s6A','j2j3','j1j2', or 'txj1'
                    e.g. jason-2 and jason-3 tandem analysis = 'j2j3'
                create mle3 SSB model: 'mle3'
                added noise for error analysis:'rms_w','rms_s' or 'rms_'ws'
                    ** _w = added SWH error, _s = added sig0 error
                smooth SWH measurements more than 1Hz: 'smth_swh'
                pseuro low resolution mode (plrm) analysis: 'plrm'
             ** more options can be added in lib_data_ext.py/pull_observables
             
                        
        VAR: define substitution for dependent variable analysis
            options (dependent variable):
                'usla': uncorrected SLA (idential to nominal SSB solution if ERR=[])
                'dISSB_tdm': difference between altimeter ionosphere corrections (tandem analysis)
                'dSWH_tdm': difference between 1 Hz SWH measurements (tandem analysis)
                'dWTC_tdm': difference between radiometer wet troposphere corrections (tandem analysis)
                'dORM_tdm': difference between orbit-range-mss (tandem analysis)
                'bm3': Gaspar, 1994 BM3 model (for verification purposes)
                'ib_corr': usla without HF correction
                'alt_cnes': replace Goddard orbit solution with CNES solution in usla (for TX)
                'alt_jpl': replace CNES orbit solution with JPL solution in usla (for J2)
                'wet_tropo_model': replace radiometer WTC correction with model correction in usla
                'iono_tran': replace GIM ionosphere correction with ALT (i.e. Tran) correction in usla
                'iono_tran_smooth': replace GIM ionosphere correction with smooted ALT (i.e. Tran) correction in usla
                'iono_corr_alt': replace GIM ionosphere correction with ALT correction from SSB model in usla
                'ocean_fes': replace GOT ocean tide correction with FES correction in usla
                'mss_dtu': replace CNES 2011 MSS correction with DTU correction in usla' (for TX)
                'mss_cnes15': replace CNES 2011 MSS correction with CNES 2015 correction in usla' (for J1-3)
            ** more options can be added in lib_filter_ext.py/var2observables
'''


#### Define Input Parameters ####
MISSIONNAME = 'j2'
METHOD = 'coe'
MNtag = 'y2'
SOURCE = 'jpl'
DIM = 'u10'
OMIT = []
WEIGHT = True

#### Error analysis specific ####
ERR = []
VAR = 'alt_jpl'

#### Run step-3 (for error analysis) ####
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
print('MISSIONNAME,METHOD,ERR')
print(MISSIONNAME,METHOD,ERR)
print('SOURCE,DIM,OMIT')
print(SOURCE,DIM,OMIT)
print('CYCLES,WEIGHT')
print(CYCLES,WEIGHT)

x_grid,y_grid,z_grid = mm2m.run_matrices2models(MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,OMIT,DIM,WEIGHT,ERR=ERR) 

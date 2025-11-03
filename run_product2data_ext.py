#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 08:11:38 2019

@author: alexaputnam

Author: Alexa Angelina Putnam Shilleh
Institution: University of Colorado Boulder
Software: sea state bias model generator (interpolation method)
Thesis: Sea State Bias Model Development and Error Analysis for Pulse-limited Altimetry
Semester/Year: Fall 2021 

"""
import numpy as np
import lib_dict_ext as dic
import main_product2data_ext as lp2d

####################################################################
'''
Step-1: Pull data from product, filter the content, and stack the data by-cycle.

~~~~~~Input~~~~~~
        MISSIONNAME: input string corresponding to mission name
            options: 
                'tx' for Topex/Poseidon
                'j1' or "j2" or "j3" for Jason missions 1-3
                's6A' for Sentinel-6 Michael Freilich (LRM)
                's6Ah' for Sentinel-6 Michael Freilich (SAR)
                
        cyc: an array containing the numbers for each cycle that will be processed
            i.e. cyc = np.arange(48,101)
            (Bad cycles will be removed. This can be check in lib_dict_ext.py/bad_cycles)
            
        VERSION: input integer to define data format version
            options:
                0 for Topex GDRF
                1 for Jason 1-3 GDRD
                3 for Jason-3 GDRF and Sentinel-6 MF
                4 for Sentinel-3
                5 for SWOT 1-day repeat
                
        ALTINTERP: input string that defines whether to interpolate new orbit solution or not. 
            options:
                'full': pull external orbit files, interpolate and add new orbit variable to 
                    altimetry files.
                'part': external files have already been pulled. Just interpolate and 
                    add new orbit variable to altimetry files.
                'none': do not include new orbit solution
            ** additional orbit solutions can be added in lib_dict_ext.py/orbit_paths
        SOURCE: input string corresponding to altimeter data source
            options:
                'jpl' for data received from JPL files
                'aviso' for data received from AVISO
                'rads' for data received from RADS software
            ** additional sources can be added in lib_dict_ext.py/product_paths
                if a new source is added then be sure to make a new directory
                corresponding to the source name in ~/imssb_software/output/mn_template/product2data/
'''
#### Define Input Parameters ####
MISSIONNAME = 'j3'
VERSION = 3
ALTINTERP= 'none'
SOURCE='rads'
if MISSIONNAME == 'sw': # Topex models: side A (cycles 48-100), side B (cycles 280-364)
    cyc = np.arange(138,140)#np.hstack((np.arange(48,103),np.arange(280,344))) 
elif MISSIONNAME == 'j1':
    cyc = np.arange(224,240)#np.hstack((np.arange(33,120),np.arange(150,240)))
elif MISSIONNAME == 'j2': # Jason-2 2009 model (cycles 19-55)
    cyc = np.arange(214,281)#np.hstack((np.arange(24,56),np.arange(93,281)))
elif MISSIONNAME == 'j3':
    cyc = np.arange(107,217)#(217,302)#(26,70)
elif MISSIONNAME=='s3':
    cyc = np.arange(12,27)# issues with 21,22
elif MISSIONNAME in ['s6A','s6Ah']:
    cyc = np.arange(15,64)# last ended at 47 #start with cycle 5 to avoid bias before 2020-12-24
elif MISSIONNAME=='sw':
    cyc = np.arange(102,279)# 1day-repeat (102-278)
#### Run step-1  ####
badcyc = dic.bad_cycles(MISSIONNAME)
CYCLES = np.setdiff1d(cyc,badcyc)
print('MISSIONNAME,CYCLES,VERSION')
print(MISSIONNAME,CYCLES,VERSION)
NC,NC_attr,idVAR = lp2d.run_product2direct(CYCLES,MISSIONNAME,SOURCE,VERSION,ALTINTERP)




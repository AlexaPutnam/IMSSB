#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:32:20 2020

@author: alexaputnam

Author: Alexa Angelina Putnam Shilleh
Institution: University of Colorado Boulder
Software: sea state bias model generator (interpolation method)
Thesis: Sea State Bias Model Development and Error Analysis for Pulse-limited Altimetry
Semester/Year: Fall 2021
"""

import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
import time
import warnings
from glob import glob
import os
from scipy import stats

import lib_dict_ext as ldic
'''
IMPORTANT NOTE:
Prior to running "run_product2data_ext.py", make sure that the data is being 
    pulled from the desired altimeter product folder. To check, go to "lib_dict_ext.py"
    and check the paths defined for each mission in the "product_paths" function. 
    Do the same if you wish to include the JPL orbit solution. i.e. go to "lib_dict_ext.py"
    and check the paths defined for each mission in the "orbit_paths" function.

DEFINITIONS:
        MISSIONNAME: input string corresponding to mission name
            choices: 
                "tx" for Topex/Poseidon
                "j1" or "j2" or "j3" for Jason missions 1-3
                "s6A" for Sentinel-6 Michael Freilich
        VERSION: input integer
            choices:
                0 for Topex GDRF
                1 for Jason 1-3 GDRD
                3 for Jason-3 GDRF and Sentinel-6 MF
                4 for Sentinel-3
        ALTINTERP: True or Falso to add new orbit solution
        RTRK: "ocean" or "adaptive"
            Default to "ocean". Must go into "run_product2direct" function in 
            "main_product2data_ext.py" to change.

        NC: Python dictionary containing all filtered altimetry data for a 
            single cycle.
        NC_attr: Python dictionary containing the attribute of the filtered 
            altimetry data for a single cycle. Attributes include; quality_flag,
            scale_factor, add_offset and units.
        CYC: cycle number as string (nCYC = cycle number as integer)
        FILENAME: 
            name assigned to final, saved netcdf4 file
        ds: dataset corresponding to the netcdf4 altimeter GDR product file 
            i.e. ds = Dataset("altimeterproduct.nc")
        dsVAR: Python dictionary containing thresholds of filtered variables 
            i.e. dsVAR['swh_ku'] = [0,20] 
        idVAR_filt: Python dictionary containing names of all variables required
            to pass through the filter
        idVAR_app (or idVAR_err): Python dictionary containing names of all variables that do
            not pass through the filter
        idVARCYC: Python dictionary containing names of cycle information that
            does not conform to the dataset (i.e. pass number, first measurement
            time, etc)
        idVAR: idVAR_filt+idVAR_app

'''
###############################################
###############################################
# save data by cycle
def crt_product2direct(MISSIONNAME,SOURCE,NC,NC_attr,nCYC,FILENAME):
    '''
    Purpose: create product2data (step-1) netCDF4 file
    NC = library containing dataset to save
    NC_attr = library containing attributes corresponding to NC
    nCYC = cycle number
    FILENAME = name of file to save
    '''
    root_grp = Dataset(FILENAME, 'w', format='NETCDF4')
    root_grp.description = str(nCYC)+"cycle NetCDF containing variables required for an SSB model and error analysis. Along-track (direct) data is taken from "+SOURCE+" and all passes are compiled into a single file of that cycle. Step-1 in documentation."
    root_grp.history = "Author: Alexa Angelina Putnam Shilleh. Institute: University of Colorado Boulder" + time.ctime(time.time())
    vrs =  NC.keys()
    Nid = np.shape(vrs)[0]
    Nsz = np.shape(NC['swh_ku'])[0]
    root_grp.createDimension('obs', Nsz)
    for ii in np.arange(Nid):
        if vrs[ii] not in ['alt_echo_type','surface_type','surface_classification_flag','rad_sea_ice_flag','ice_flag','rad_rain_flag','rain_flag']: 
            print('save variable: '+vrs[ii])
        if vrs[ii] in ['pass_number','equator_longitude']:
            xii = root_grp.createVariable(vrs[ii], 'f8', ('obs',))
            print(str(np.shape(NC[vrs[ii]])[0])+'/'+str(Nsz))
            xii[:] = NC[vrs[ii]]
        else:
            xii = root_grp.createVariable(vrs[ii], 'f8', ('obs',))
            print('size of '+vrs[ii])
            print(np.size(NC[vrs[ii]]))
            xii[:] = NC[vrs[ii]]
            if vrs[ii] == 'ssha':
                xii.units = NC_attr['units']['sea_state_bias_ku']
            else:
                xii.units = NC_attr['units'][vrs[ii]]
    root_grp.close()  

###############################################
###############################################
#-----------------------------------------------------------------------
# [Step 1] Filter setup for product data
#-----------------------------------------------------------------------
def rtrk_variable_match(MISSIONNAME,RTRK,VERSION):
    '''
    Purpose: Rename variables to match nomenclature and transfer into library (LIB)
    RTRK = ‘ocean’ or ‘adaptive’ for VERSION = 3 (default: ocean)
    '''
    LIB={}
    if VERSION in [1]:
        LIB['ocean_tide_got']='ocean_tide_sol1' #var_change
        LIB['ocean_tide_fes']='ocean_tide_sol2' #var_change
        LIB['load_tide_got']='ocean_tide_sol1' #var_change
        LIB['load_tide_fes']='ocean_tide_sol2' #var_change
        LIB['ocean_tide_non_eq']='ocean_tide_non_equil' #var_change
        LIB['surface_classification_flag']='surface_type' #var_change
        LIB['mean_dynamic_topography']='mean_topography'
    elif VERSION in [0,2]:
        LIB['lat']='latitude' #var_change
        LIB['lon']='longitude' #var_change
        LIB['alt']='altitude' #var_change
        LIB['rad_surf_type'] = 'rad_surface_type_flag'
        LIB['rad_wet_tropo_corr']='rad_wet_tropo_cor' #var_change
        LIB['model_dry_tropo_corr']='model_dry_tropo_cor_zero_altitude' #var_change
        LIB['wind_speed_rad']='rad_wind_speed' # var_change
        LIB['model_wet_tropo_corr']='model_wet_tropo_cor_zero_altitude' #var_change
        LIB['iono_corr_alt_ku']='iono_cor_alt_ku' #var_change
        LIB['iono_corr_alt_ku_mle3']='iono_cor_alt_ku_mle3'#ocean only #var_change
        LIB['iono_corr_gim_ku']='iono_cor_gim_ku' #var_change
        LIB['mean_sea_surface']='mean_sea_surface_cnescls'#var_change
        LIB['bathymetry']='depth_or_elevation'
    elif VERSION in [3]:
        if MISSIONNAME=='j3':
            #Ku-band
            LIB['swh_ku']='swh_'+RTRK
            LIB['sig0_ku']='sig0_'+RTRK
            LIB['range_ku']='range_'+RTRK
            LIB['off_nadir_angle_wf_ku']='off_nadir_angle_wf_'+RTRK
            LIB['off_nadir_angle_wf_rms_ku']='off_nadir_angle_wf_'+RTRK+'_rms'
            LIB['iono_corr_gim_ku']='iono_cor_gim' #var_change
            LIB['range_numval_ku']='range_'+RTRK+'_numval'
            LIB['range_rms_ku']='range_'+RTRK+'_rms'
            LIB['sig0_rms_ku']='sig0_'+RTRK+'_rms'
            LIB['swh_rms_ku']='swh_'+RTRK+'_rms'
            #Ku-newly added
            LIB['sea_state_bias_3d_mp2']='sea_state_bias_3d_mp2'
            LIB['sea_state_bias_adaptive_3d_mp2']='sea_state_bias_adaptive_3d_mp2'
            LIB['wvf_main_class']='wvf_main_class'
            #1Hz file
            LIB['lat']='latitude' #var_change
            LIB['lon']='longitude' #var_change
            LIB['alt']='altitude' #var_change
            LIB['rad_wet_tropo_corr']='rad_wet_tropo_cor' #var_change
            LIB['model_dry_tropo_corr']='model_dry_tropo_cor_zero_altitude' #var_change
            LIB['solid_earth_tide']='solid_earth_tide'
            LIB['internal_tide_hret']='internal_tide' #!!!
            LIB['mean_sea_surface']='mean_sea_surface_cnescls'#var_change
            #LIB['alt_echo_type'] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            LIB['wind_speed_rad']='rad_wind_speed' # var_change
            LIB['model_wet_tropo_corr']='model_wet_tropo_cor_zero_altitude' #var_change
            LIB['mean_sea_surface_dtu']='mean_sea_surface_dt'
            LIB['bathymetry']='depth_or_elevation'
            LIB['rad_surf_type']='rad_surface_type_flag'
            # newly added
            LIB['inv_bar_corr']='inv_bar_cor'
            LIB['sst']='sst'
            LIB['mean_wave_period_t02']='mean_wave_period_t02'
            LIB['mean_wave_direction']='mean_wave_direction'
            LIB['sea_ice_concentration']='sea_ice_concentration'
            LIB['angle_of_approach_to_coast']='angle_of_approach_to_coast'
            LIB['wave_model_map_availability_flag']='wave_model_map_availability_flag'
            LIB['altitude_rate']='altitude_rate'
            LIB['model_dry_tropo_cor_measurement_altitude']='model_dry_tropo_cor_measurement_altitude'
            LIB['model_wet_tropo_cor_measurement_altitude']='model_wet_tropo_cor_measurement_altitude'
            LIB['surface_slope_cor']='surface_slope_cor'
            LIB['ocean_tide_non_eq']='ocean_tide_non_eq'
            LIB['dac']='dac'
            if RTRK=='ocean':
                #ssb
                LIB['sea_state_bias_ku'] = 'sea_state_bias'
                LIB['sea_state_bias_c'] = 'sea_state_bias'
                #iono
                LIB['iono_corr_alt_ku']='iono_cor_alt_filtered' #var_change
                LIB['iono_corr_alt_ku_mle3']='iono_cor_alt_filtered_mle3'#ocean only #var_change
                #range
                LIB['range_c']='range_'+RTRK # not available for adaptive
                LIB['range_rms_c']='range_'+RTRK+'_rms'# not available for adaptive
                LIB['range_ku_mle3']='range_'+RTRK+'_mle3'#ocean only
                LIB['range_rms_ku_mle3']='range_'+RTRK+'_mle3_rms' #ocean only
                LIB['range_numval_c']='range_'+RTRK+'_numval' # not available for adaptive
                LIB['range_numval_ku_mle3']='range_'+RTRK+'_mle3_numval' #ocean only
                #sig0
                LIB['sig0_c']='sig0_'+RTRK# not available for adaptive
                LIB['sig0_rms_c']='sig0_'+RTRK+'_rms'# not available for adaptive
                LIB['sig0_ku_mle3']='sig0_'+RTRK+'_mle3'#ocean only
                LIB['sig0_rms_ku_mle3']='sig0_'+RTRK+'_mle3_rms' #ocean only
                #swh
                LIB['swh_c']='swh_'+RTRK# not available for adaptive
                LIB['swh_rms_c']='swh_'+RTRK+'_rms'# not available for adaptive
                LIB['swh_ku_mle3']='swh_'+RTRK+'_mle3'#ocean only
                LIB['swh_rms_ku_mle3']='swh_'+RTRK+'_mle3_rms' #ocean only
            elif RTRK=='adaptive':
                LIB['wind_speed_alt']='wind_speed_alt_'+RTRK
                LIB['sea_state_bias_ku'] = 'sea_state_bias_'+RTRK
                LIB['sea_state_bias_c'] = 'sea_state_bias_'+RTRK
                LIB['iono_corr_alt_ku']='iono_cor_alt_filtered'+RTRK #var_change
        elif MISSIONNAME =='s6A':
            #Ku-band
            LIB['swh_ku']='swh_'+RTRK
            LIB['sig0_ku']='sig0_'+RTRK
            LIB['range_ku']='range_'+RTRK # 
            LIB['off_nadir_angle_wf_ku']='off_nadir_angle_wf_'+RTRK
            LIB['off_nadir_angle_wf_rms_ku']='off_nadir_angle_wf_'+RTRK+'_rms'
            LIB['iono_corr_gim_ku']='iono_cor_gim' #var_change
            LIB['iono_corr_gim_c']='iono_cor_gim' #var_change
            LIB['sig0_rms_ku']='sig0_'+RTRK+'_rms'
            LIB['swh_rms_ku']='swh_'+RTRK+'_rms'
            #1Hz file
            LIB['lat']='latitude' #var_change
            LIB['lon']='longitude' #var_change
            LIB['alt']='altitude' #var_change
            LIB['rad_wet_tropo_corr']='rad_wet_tropo_cor' #var_change
            LIB['model_dry_tropo_corr']='model_dry_tropo_cor_zero_altitude' #var_change
            LIB['solid_earth_tide']='solid_earth_tide'
            LIB['internal_tide_hret']='internal_tide' #!!!
            LIB['mean_sea_surface']='mean_sea_surface_sol1'# CNES/CLS 2015
            LIB['mean_sea_surface_dtu']='mean_sea_surface_sol2'# DTU 2018
            LIB['ocean_tide_got']='ocean_tide_sol1' #var_change GOT4.10
            LIB['ocean_tide_fes']='ocean_tide_sol2' #var_change FES 2014
            LIB['load_tide_got']='ocean_tide_sol1' #var_change
            LIB['load_tide_fes']='ocean_tide_sol2' #var_change
            #LIB['alt_echo_type'] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            LIB['wind_speed_rad']='rad_wind_speed' # var_change
            LIB['model_wet_tropo_corr']='model_wet_tropo_cor_zero_altitude' #var_change
            LIB['bathymetry']='depth_or_elevation'
            LIB['rad_surf_type']='rad_surface_type_flag'
            # newly added
            LIB['inv_bar_corr']='inv_bar_cor'
            LIB['angle_of_approach_to_coast']='angle_of_approach_to_coast'
            LIB['altitude_rate']='altitude_rate'
            LIB['model_dry_tropo_cor_measurement_altitude']='model_dry_tropo_cor_measurement_altitude'
            LIB['model_wet_tropo_cor_measurement_altitude']='model_wet_tropo_cor_measurement_altitude'
            LIB['ocean_geo_corrections']='ocean_geo_corrections'
            LIB['ocean_tide_non_eq']='ocean_tide_non_eq'
            LIB['dac']='dac'
            if RTRK=='ocean':
                #ssb
                LIB['sea_state_bias_ku'] = 'sea_state_bias'
                LIB['sea_state_bias_c'] = 'sea_state_bias'
                #iono
                LIB['iono_corr_alt_ku']='iono_cor_alt_filtered' #var_change
                LIB['iono_corr_alt_ku_mle3']='iono_cor_alt_filtered_mle3'#ocean only #var_change
                #range
                LIB['range_numval_ku']='range_'+RTRK+'_numval'
                LIB['range_rms_ku']='range_'+RTRK+'_rms'
                LIB['range_c']='range_'+RTRK # not available for adaptive
                LIB['range_numval_c']='range_'+RTRK+'_numval' # not available for adaptive
                LIB['range_rms_c']='range_'+RTRK+'_rms'# not available for adaptive
                LIB['range_ku_mle3']='range_'+RTRK+'_mle3'#ocean only
                LIB['range_rms_ku_mle3']='range_'+RTRK+'_mle3_rms' #ocean only
                LIB['range_numval_ku_mle3']='range_'+RTRK+'_mle3_numval' #ocean only
                #sig0
                LIB['sig0_c']='sig0_'+RTRK# not available for adaptive
                LIB['sig0_rms_c']='sig0_'+RTRK+'_rms'# not available for adaptive
                LIB['sig0_ku_mle3']='sig0_'+RTRK+'_mle3'#ocean only
                LIB['sig0_rms_ku_mle3']='sig0_'+RTRK+'_mle3_rms' #ocean only
                #swh
                LIB['swh_c']='swh_'+RTRK# not available for adaptive
                LIB['swh_rms_c']='swh_'+RTRK+'_rms'# not available for adaptive
                LIB['swh_ku_mle3']='swh_'+RTRK+'_mle3'#ocean only
                LIB['swh_rms_ku_mle3']='swh_'+RTRK+'_mle3_rms' #ocean only
        elif MISSIONNAME =='s6Ah':
            '''
            Does not contain: 'off_nadir_angle_wf_ocean', all c-band, all mle3
            Newly added: iono_corr_alt_ku_notsmooth,off_nadir_pitch_angle_pf,off_nadir_roll_angle_pf,off_nadir_yaw_angle_pf,ocean_geo_corrections
            distance_to_coast, rad_distance_to_land
            '''
            #Ku-band
            LIB['swh_ku']='swh_'+RTRK
            LIB['sig0_ku']='sig0_'+RTRK
            LIB['range_ku']='range_'+RTRK # 
            LIB['off_nadir_pitch_angle_pf']='off_nadir_pitch_angle_pf' #!!!
            LIB['off_nadir_roll_angle_pf']='off_nadir_roll_angle_pf' #!!!
            LIB['off_nadir_yaw_angle_pf']='off_nadir_yaw_angle_pf' #!!!
            LIB['iono_corr_gim_ku']='iono_cor_gim' #var_change
            LIB['sig0_rms_ku']='sig0_'+RTRK+'_rms'
            LIB['swh_rms_ku']='swh_'+RTRK+'_rms'
            #1Hz file
            LIB['lat']='latitude' #var_change
            LIB['lon']='longitude' #var_change
            LIB['alt']='altitude' #var_change
            LIB['rad_wet_tropo_corr']='rad_wet_tropo_cor' #var_change
            LIB['model_dry_tropo_corr']='model_dry_tropo_cor_zero_altitude' #var_change
            LIB['solid_earth_tide']='solid_earth_tide'
            LIB['internal_tide_hret']='internal_tide' #!!!
            LIB['mean_sea_surface']='mean_sea_surface_sol1'# CNES/CLS 2015
            LIB['mean_sea_surface_dtu']='mean_sea_surface_sol2'# DTU 2018
            LIB['ocean_tide_got']='ocean_tide_sol1' #var_change GOT4.10
            LIB['ocean_tide_fes']='ocean_tide_sol2' #var_change FES 2014
            LIB['load_tide_got']='ocean_tide_sol1' #var_change
            LIB['load_tide_fes']='ocean_tide_sol2' #var_change
            #LIB['alt_echo_type'] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            LIB['wind_speed_rad']='rad_wind_speed' # var_change
            LIB['model_wet_tropo_corr']='model_wet_tropo_cor_zero_altitude' #var_change
            LIB['bathymetry']='depth_or_elevation'
            LIB['rad_surf_type']='rad_surface_type_flag'
            # newly added
            LIB['inv_bar_corr']='inv_bar_cor'
            LIB['angle_of_approach_to_coast']='angle_of_approach_to_coast'
            LIB['altitude_rate']='altitude_rate'
            LIB['model_dry_tropo_cor_measurement_altitude']='model_dry_tropo_cor_measurement_altitude'
            LIB['model_wet_tropo_cor_measurement_altitude']='model_wet_tropo_cor_measurement_altitude'
            LIB['ocean_geo_corrections']='ocean_geo_corrections'
            LIB['ocean_tide_non_eq']='ocean_tide_non_eq'
            LIB['dac']='dac'
            if RTRK=='ocean':
                #ssb
                LIB['sea_state_bias_ku'] = 'sea_state_bias'
                #iono
                LIB['iono_corr_alt_ku']='iono_cor_alt_filtered' #var_change
                LIB['iono_corr_alt_ku_notsmooth']='iono_cor_alt' #var_change
                #range
                LIB['range_numval_ku']='range_'+RTRK+'_numval'
                LIB['range_rms_ku']='range_'+RTRK+'_rms'
    elif VERSION in [4]: # Sentinel-3
        #VARTRK
        LIB['time']='time_01'
        LIB['lat']='lat_01'
        LIB['lon']='lon_01'
        #VARPAR
        LIB['swh_ku']='swh_ocean_01_ku' #swh_ocean_01_plrm_ku
        LIB['swh_c']='swh_ocean_01_c' 
        LIB['sig0_ku']='sig0_ocean_01_ku' #sig0_ocean_01_plrm_ku
        LIB['sig0_c']='sig0_ocean_01_c' 
        LIB['wind_speed_alt']='wind_speed_alt_01_ku' #wind_speed_alt_01_plrm_ku
        #VARALT
        LIB['alt']='alt_01'
        LIB['range_ku']='range_ocean_01_ku' #range_ocean_01_plrm_ku
        LIB['range_c']='range_ocean_01_c' 
        LIB['off_nadir_angle_wf_ku']='corrected_off_nadir_angle_wf_ocean_01_plrm_ku' #'corrected_off_nadir_angle_wf_ocean_01_plrm_ku'
        LIB['sea_state_bias_ku'] = 'sea_state_bias_01_ku' #sea_state_bias_01_plrm_ku
        LIB['sea_state_bias_c'] = 'sea_state_bias_01_c'
        LIB['ssha'] = 'ssha_01_ku' #'ssha_01_plrm_ku'
        # VARATM
        LIB['rad_wet_tropo_corr']='rad_wet_tropo_cor_01_ku' #rad_wet_tropo_cor_01_plrm_ku
        LIB['model_dry_tropo_corr']='mod_dry_tropo_cor_zero_altitude_01' 
        LIB['iono_corr_gim_ku']='iono_cor_gim_01_ku' 
        # VARTID
        LIB['pole_tide']='pole_tide_01'
        LIB['solid_earth_tide']='solid_earth_tide_01'
        LIB['ocean_tide_got'] = 'ocean_tide_sol1_01'
        LIB['ocean_tide_non_eq'] = 'ocean_tide_non_eq_01'
        LIB['ocean_tide_fes'] = 'ocean_tide_sol2_01'
        # VARMSS
        LIB['mean_sea_surface']='mean_sea_surf_sol1_01'
        LIB['mean_sea_surface_dtu']='mean_sea_surf_sol2_01'
        # VARNUM
        LIB['range_numval_ku']='range_ocean_numval_01_ku'#'range_ocean_numval_01_plrm_ku'
        LIB['range_numval_c']='range_ocean_numval_01_c'
        # VARRMS
        LIB['range_rms_ku']='range_ocean_rms_01_ku'#'range_ocean_rms_01_plrm_ku'
        LIB['range_rms_c']='range_ocean_rms_01_c'
        LIB['sig0_rms_ku']='sig0_ocean_rms_01_ku'#'sig0_ocean_rms_01_plrm_ku'
        LIB['sig0_rms_c']='sig0_ocean_rms_01_c'
        LIB['swh_rms_ku']='swh_ocean_rms_01_ku'#'swh_ocean_rms_01_plrm_ku'
        LIB['swh_rms_c']='swh_ocean_rms_01_c'
        # VARFLG
        LIB['surface_classification_flag'] = 'surf_type_01' # 0 =ocean_or_semi_enclosed_sea
        LIB['ice_flag'] = 'open_sea_ice_flag_01_ku' # 0 = ocean # 'open_sea_ice_flag_01_plrm_ku'
        LIB['rain_flag'] = 'rain_flag_01_ku' # 'rain_flag_01_plrm_ku' 0 = no_rain
        LIB['rad_surf_type'] = 'rad_surf_type_01' # [0 1] = ocean land
        # VERSION 4 specific - 1
        LIB['inv_bar_corr']='inv_bar_cor_01'
        LIB['hf_fluctuations_corr']='hf_fluct_cor_01'
        LIB['instr_op_mode']='instr_op_mode_01' #[0 1 2] = LRM SAR LRM_and_SAR
        LIB['surf_class']='surf_class_01' # 0 =open_ocean
        LIB['rad_along_track_avg_flag'] = 'rad_along_track_avg_flag_01'# 0 = good [1 = fail]
        # 2nd tier variables
        # NO MLE3
        LIB['iono_corr_alt_ku']='iono_cor_alt_01_ku' #iono_cor_alt_01_plrm_ku
        LIB['model_wet_tropo_corr']='mod_wet_tropo_cor_zero_altitude_01'
        LIB['load_tide_got']='load_tide_sol1_01'
        LIB['load_tide_fes']='load_tide_sol2_01'
        LIB['mean_dynamic_topography']='geoid_01'
        LIB['geoid']='geoid_01'
        LIB['bathymetry']='odle_01'
        # VERSION 4 specific - 2
        LIB['off_nadir_angle_wf_rms_ku']='off_nadir_angle_rms_01_ku' ##'off_nadir_angle_rms_01_plrm_ku'
        LIB['UTC_day']='UTC_day_01'
        LIB['off_nadir_roll_angle_pf']='off_nadir_roll_angle_pf_01'
        LIB['off_nadir_pitch_angle_pf']='off_nadir_pitch_angle_pf_01'
        LIB['off_nadir_yaw_angle_pf']='off_nadir_yaw_angle_pf_01'
        LIB['climato_use_flag_ku'] = 'climato_use_flag_01_ku'
        LIB['rad_distance_to_land'] = 'dist_coast_01'
        # PLRM
        LIB['swh_ku_plrm']='swh_ocean_01_plrm_ku'
        LIB['sig0_ku_plrm']='sig0_ocean_01_plrm_ku'
        LIB['wind_speed_alt_plrm']='wind_speed_alt_01_plrm_ku'
        LIB['range_ku_plrm']='range_ocean_01_plrm_ku'
        LIB['off_nadir_angle_wf_ku_sar']='corrected_off_nadir_angle_wf_ocean_01_ku'
        LIB['sea_state_bias_ku_plrm'] = 'sea_state_bias_01_plrm_ku'
        LIB['ssha_plrm'] = 'ssha_01_plrm_ku'
        LIB['rad_wet_tropo_corr_plrm']='rad_wet_tropo_cor_01_plrm_ku'
        LIB['range_numval_ku_plrm']='range_ocean_numval_01_plrm_ku'
        LIB['range_rms_ku_plrm']='range_ocean_rms_01_plrm_ku'
        LIB['sig0_rms_ku_plrm']='sig0_ocean_rms_01_plrm_ku'
        LIB['swh_rms_ku_plrm']='swh_ocean_rms_01_plrm_ku'
        LIB['ice_flag_plrm'] = 'open_sea_ice_flag_01_plrm_ku'
        LIB['rain_flag_plrm'] = 'rain_flag_01_plrm_ku'
        LIB['iono_corr_alt_ku_plrm']='iono_cor_alt_01_plrm_ku'
        LIB['off_nadir_angle_wf_rms_ku_plrm']='off_nadir_angle_rms_01_plrm_ku'
    if VERSION in [5]: #SWOT 1-day repeat
        LIB['alt']='alt_gdrf'
        LIB['off_nadir_angle_wf_ku']='off_nadir_angle2_wf_ku'
        LIB['sea_state_bias_ku']='ssb_cls'
        LIB['sea_state_bias_c']='ssb_cls_c'
        # VARATM
        LIB['rad_wet_tropo_corr']='wet_tropo_rad' 
        LIB['model_dry_tropo_corr']='dry_tropo_ecmwf' 
        # VARTID
        LIB['pole_tide']='tide_pole'
        LIB['solid_earth_tide']='tide_solid'
        LIB['ocean_tide_got'] = 'ocean_tide_sol1_01'
        LIB['ocean_tide_non_eq'] = 'ocean_tide_non_eq_01'
        LIB['ocean_tide_fes'] = 'ocean_tide_sol2_01'
        LIB['ocean_tide_got']='tide_ocean_got410' #var_change
        LIB['ocean_tide_fes']='tide_ocean_fes14' #var_change
        LIB['load_tide_got']='tide_load_got410' #var_change
        LIB['load_tide_fes']='tide_load_fes14' #var_change
        LIB['ocean_tide_non_eq']='tide_non_equil' #var_change
        # VARMSS
        LIB['mean_sea_surface']='mss_cnescls15'
        LIB['mean_sea_surface_dtu']='mss_dtu18'
        # VARNUM
        LIB['range_numval_ku']='range_numval'
        # VARFLG
        LIB['surface_classification_flag'] = 'surface_class' #
        LIB['ice_flag'] = 'qual_alt_rain_ice' # 0 = ocean 
        LIB['rain_flag'] = 'qual_rad1_rain_ice' # 'rain_flag_01_plrm_ku' 0 = no_rain
        LIB['rad_surf_type'] = 'surface_type_rad1' # [0 1] = ocean land
        LIB['dac']='inv_bar_mog2d'
        LIB['inv_bar_corr']='inv_bar_mog2d'
        # NO MLE3
        LIB['iono_corr_alt_ku']='iono_alt' #iono_cor_alt_01_plrm_ku
        LIB['model_wet_tropo_corr']='wet_tropo_ecmwf'
        LIB['mean_dynamic_topography']='geoid_egm2008'
        LIB['geoid']='geoid_egm2008'
        LIB['bathymetry']='topo_dtu10'
        # VERSION 4 specific - 2
        LIB['off_nadir_angle_wf_rms_ku']='off_nadir_angle2_wf_rms_ku' ##'off_nadir_angle_rms_01_plrm_ku'
        LIB['rad_distance_to_land'] = 'dist_coast'
        # MLE3
        LIB['iono_corr_alt_ku_mle3']='iono_alt_mle3'#ocean only #var_change
        LIB['wind_speed_alt_mle3']='wind_speed_alt_mle3'
        LIB['ssb_cls_mle3']='ssb_cls_mle3'
    return LIB

def rtrk_variables(RTRK,MISSIONNAME):
    '''
    Purpose: Extra step for new data format provided by the product (i.e. VERSION = 3). The Ku-band, C-band and band-independent variables are all placed into different subgroups within the product netCDF4 file.
    RTRK = ‘ocean’ or ‘adaptive’ for VERSION = 3 (default: ocean)
    vrk,vrc,vr1 = list of Ku-band, C-band and band-independent variables, respectively
    '''
    if MISSIONNAME=='j3':
        vrk=['alt_state_band_status_flag', 'range_ocean_compression_qual', 'range_ocean_mle3_compression_qual',
         'range_adaptive_compression_qual', 'swh_ocean_compression_qual', 'swh_ocean_mle3_compression_qual',
         'swh_adaptive_compression_qual', 'sig0_ocean_compression_qual', 'sig0_ocean_mle3_compression_qual',
         'sig0_adaptive_compression_qual', 'off_nadir_angle_wf_ocean_compression_qual', 'range_cor_ocean_net_instr_qual',
         'swh_cor_ocean_net_instr_qual', 'sig0_cor_ocean_net_instr_qual', 'iono_cor_alt_filtered_qual',
         'iono_cor_alt_filtered_mle3_qual', 'iono_cor_alt_filtered_adaptive_qual', 'range_ocean', 'range_ocean_rms',
         'range_ocean_numval', 'range_ocean_mle3', 'range_ocean_mle3_rms', 'range_ocean_mle3_numval', 'range_adaptive',
         'range_adaptive_rms', 'range_adaptive_numval', 'range_cor_ocean_net_instr', 'range_cor_ocean_mle3_net_instr',
         'range_cor_adaptive_net_instr', 'iono_cor_alt', 'iono_cor_alt_mle3', 'iono_cor_alt_adaptive', 'iono_cor_alt_filtered',
         'iono_cor_alt_filtered_mle3', 'iono_cor_alt_filtered_adaptive', 'iono_cor_gim', 'sea_state_bias',
         'sea_state_bias_mle3', 'sea_state_bias_adaptive', 'sea_state_bias_3d_mp2', 'sea_state_bias_adaptive_3d_mp2',
         'swh_ocean', 'swh_ocean_rms', 'swh_ocean_numval', 'swh_ocean_mle3', 'swh_ocean_mle3_rms', 'swh_ocean_mle3_numval',
         'swh_adaptive', 'swh_adaptive_rms', 'swh_adaptive_numval', 'swh_cor_ocean_net_instr', 'swh_cor_ocean_mle3_net_instr',
         'swh_cor_adaptive_net_instr', 'sig0_ocean', 'sig0_ocean_rms', 'sig0_ocean_numval', 'sig0_ocean_mle3', 'sig0_ocean_mle3_rms',
         'sig0_ocean_mle3_numval', 'sig0_adaptive', 'sig0_adaptive_rms', 'sig0_adaptive_numval', 'agc', 'agc_rms', 'agc_numval',
         'sig0_cor_ocean_net_instr', 'sig0_cor_ocean_mle3_net_instr', 'sig0_cor_adaptive_net_instr', 'sig0_cor_atm',
         'off_nadir_angle_wf_ocean', 'off_nadir_angle_wf_ocean_rms', 'off_nadir_angle_wf_ocean_numval', 'wvf_main_class', 'ssha',
         'ssha_mle3']

    
        vrc=['alt_state_c_band_flag', 'alt_state_band_status_flag', 'range_ocean_compression_qual', 'swh_ocean_compression_qual',
         'sig0_ocean_compression_qual', 'range_cor_ocean_net_instr_qual', 'swh_cor_ocean_net_instr_qual',
         'sig0_cor_ocean_net_instr_qual', 'range_ocean', 'range_ocean_rms', 'range_ocean_numval', 'range_cor_ocean_net_instr',
         'sea_state_bias', 'sea_state_bias_mle3', 'sea_state_bias_adaptive', 'swh_ocean', 'swh_ocean_rms', 'swh_ocean_numval',
         'swh_cor_ocean_net_instr', 'sig0_ocean', 'sig0_ocean_rms', 'sig0_ocean_numval', 'agc', 'agc_rms', 'agc_numval',
         'sig0_cor_ocean_net_instr', 'sig0_cor_atm']

    
        vr1=['time', 'time_tai', 'index_first_20hz_measurement', 'numtotal_20hz_measurement', 'latitude', 'longitude', 
         'rad_surface_type_flag', 'rad_distance_to_land', 'surface_classification_flag', 'angle_of_approach_to_coast', 
         'distance_to_coast', 'rad_tb_187_qual', 'rad_tb_238_qual', 'rad_tb_340_qual', 'rad_averaging_flag', 
         'rad_land_frac_187', 'rad_land_frac_238', 'rad_land_frac_340', 'alt_state_oper_flag', 'alt_state_band_seq_flag', 
         'rad_state_oper_flag', 'orb_state_diode_flag', 'orb_state_rest_flag', 'meteo_map_availability_flag', 
         'wave_model_map_availability_flag', 'sig0_cor_atm_source', 'rain_flag', 'rad_rain_flag', 'ice_flag', 
         'rad_sea_ice_flag', 'rad_tb_interp_qual', 'mean_sea_surface_cnescls_interp_qual', 'mean_sea_surface_dtu_interp_qual', 
         'mean_dynamic_topography_interp_qual', 'ocean_tide_got_interp_qual', 'ocean_tide_fes_interp_qual', 
         'internal_tide_interp_qual', 'meteo_zero_altitude_interp_qual', 'meteo_measurement_altitude_interp_qual', 
         'sea_ice_concentration_interp_qual', 'wave_model_interp_qual', 'altitude', 'altitude_rate', 
         'model_dry_tropo_cor_zero_altitude', 'model_dry_tropo_cor_measurement_altitude', 'model_wet_tropo_cor_zero_altitude', 
         'model_wet_tropo_cor_measurement_altitude', 'rad_wet_tropo_cor', 'surface_slope_cor', 'rad_tmb_187', 'rad_tmb_238', 
         'rad_tmb_340', 'rad_tb_187', 'rad_tb_238', 'rad_tb_340', 'mean_sea_surface_cnescls', 'mean_sea_surface_dt', 
         'mean_dynamic_topography', 'geoid', 'depth_or_elevation', 'inv_bar_cor', 'dac', 'ocean_tide_got', 'ocean_tide_fes', 
         'ocean_tide_eq', 'ocean_tide_non_eq', 'load_tide_got', 'load_tide_fes', 'solid_earth_tide', 'pole_tide', 'internal_tide', 
         'wind_speed_mod_', 'wind_speed_mod_v', 'wind_speed_alt', 'wind_speed_alt_mle3', 'wind_speed_alt_adaptive', 'rad_wind_speed', 
         'rad_water_vapor', 'rad_cloud_liquid_water', 'sst', 'mean_wave_period_t02', 'mean_wave_direction', 'sea_ice_concentration']

    elif MISSIONNAME == 's6A':
        vrk=['atm_cor_sig0','index_first_20hz_measurement','iono_cor_gim','model_instr_cor_off_nadir_angle_wf_ocean',
             'model_instr_cor_range_ocean','model_instr_cor_range_ocean_mle3','model_instr_cor_sig0_ocean',
             'model_instr_cor_sig0_ocean_mle3','model_instr_cor_swh_ocean','model_instr_cor_swh_ocean_mle3',
             'net_instr_cor_off_nadir_angle_wf_ocean','net_instr_cor_range_ocean','net_instr_cor_range_ocean_mle3',
             'net_instr_cor_sig0_ocean','net_instr_cor_sig0_ocean_mle3','net_instr_cor_swh_ocean','net_instr_cor_swh_ocean_mle3',
             'numtotal_20hz_measurement','ocean_geo_corrections','ocean_geo_corrections_mle3','off_nadir_angle_wf_ocean',
             'off_nadir_angle_wf_ocean_numval','off_nadir_angle_wf_ocean_qual','off_nadir_angle_wf_ocean_rms','rad_atm_cor_sig0_qual',
             'range_cor_doppler','range_ocean','range_ocean_mle3','range_ocean_mle3_numval','range_ocean_mle3_qual',
             'range_ocean_mle3_rms','range_ocean_numval','range_ocean_qual','range_ocean_rms','sea_state_bias','sea_state_bias_mle3',
             'sig0_ocean','sig0_ocean_mle3','sig0_ocean_mle3_numval','sig0_ocean_mle3_qual','sig0_ocean_mle3_rms','sig0_ocean_numval',
             'sig0_ocean_qual','sig0_ocean_rms','ssha','ssha_mle3','swh_ocean','swh_ocean_mle3','swh_ocean_mle3_numval',
             'swh_ocean_mle3_qual','swh_ocean_mle3_rms','swh_ocean_numval','swh_ocean_qual','swh_ocean_rms']
        
        vrc=['atm_cor_sig0','index_first_20hz_measurement','iono_cor_gim','model_instr_cor_range_ocean','model_instr_cor_sig0_ocean',
             'model_instr_cor_swh_ocean','net_instr_cor_range_ocean','net_instr_cor_sig0_ocean','net_instr_cor_swh_ocean',
             'numtotal_20hz_measurement','rad_atm_cor_sig0_qual','range_cor_doppler','range_ocean','range_ocean_numval',
             'range_ocean_qual','range_ocean_rms','sea_state_bias','sig0_ocean','sig0_ocean_numval','sig0_ocean_qual',
             'sig0_ocean_rms','swh_ocean','swh_ocean_numval','swh_ocean_qual','swh_ocean_rms']
        
        vr1=['altitude','altitude_rate','altitude_rate_mean_sea_surface','angle_of_approach_to_coast','climato_use_flag','dac',
             'delta_ellipsoid_tp_wgs84','depth_or_elevation','distance_to_coast','geoid','internal_tide','inv_bar_cor',
             'iono_cor_alt','iono_cor_alt_filtered','iono_cor_alt_filtered_mle3','iono_cor_alt_mle3','l2_record_counter',
             'latitude','load_tide_sol1','load_tide_sol2','longitude','manoeuvre_flag','mean_dynamic_topography',
             'mean_dynamic_topography_acc','mean_dynamic_topography_qual','mean_sea_surface_sol1','mean_sea_surface_sol1_acc',
             'mean_sea_surface_sol1_qual','mean_sea_surface_sol2','mean_sea_surface_sol2_acc','mean_sea_surface_sol2_qual',
             'meteo_map_availability_flag','model_dry_tropo_cor_measurement_altitude','model_dry_tropo_cor_zero_altitude',
             'model_wet_tropo_cor_measurement_altitude','model_wet_tropo_cor_zero_altitude','ocean_tide_eq','ocean_tide_non_eq',
             'ocean_tide_sol1','ocean_tide_sol1_qual','ocean_tide_sol2','ocean_tide_sol2_qual','off_nadir_pitch_angle_pf',
             'off_nadir_roll_angle_pf','off_nadir_yaw_angle_pf','orbit_type_flag','pass_direction_flag','pole_tide',
             'rad_cloud_liquid_water','rad_cloud_liquid_water_qual','rad_distance_to_land','rad_land_frac_187',
             'rad_land_frac_238','rad_land_frac_340','rad_rain_flag','rad_sea_ice_flag','rad_state_oper_flag',
             'rad_surface_type_flag','rad_tb_187','rad_tb_187_qual','rad_tb_238','rad_tb_238_qual','rad_tb_340',
             'rad_tb_340_qual','rad_tmb_187','rad_tmb_187_qual','rad_tmb_238','rad_tmb_238_qual','rad_tmb_340',
             'rad_tmb_340_qual','rad_water_vapor','rad_water_vapor_qual','rad_wet_tropo_cor','rad_wet_tropo_cor_qual',
             'rad_wind_speed','rad_wind_speed_qual','rain_attenuation','rain_flag','solid_earth_tide',
             'surface_classification_flag','time','time_tai','total_electron_content','wind_speed_alt',
             'wind_speed_alt_mle3','wind_speed_mod_','wind_speed_mod_v']
    elif MISSIONNAME == 's6Ah':
        vrk=['atm_cor_sig0','index_first_20hz_measurement','iono_cor_gim','model_instr_cor_range_ocean','model_instr_cor_sig0_ocean',
             'model_instr_cor_swh_ocean','net_instr_cor_range_ocean','net_instr_cor_sig0_ocean','net_instr_cor_swh_ocean',
             'numtotal_20hz_measurement','ocean_geo_corrections','rad_atm_cor_sig0_qual','range_cor_doppler','range_ocean',
             'range_ocean_numval','range_ocean_qual','range_ocean_rms','sea_state_bias','sig0_ocean','sig0_ocean_numval',
             'sig0_ocean_qual','sig0_ocean_rms','ssha','swh_ocean','swh_ocean_numval','swh_ocean_qual','swh_ocean_rms']
        
        vrc=[]
        
        vr1=['altitude','altitude_rate','altitude_rate_mean_sea_surface','angle_of_approach_to_coast','climato_use_flag','dac',
             'delta_ellipsoid_tp_wgs84','depth_or_elevation','distance_to_coast','geoid','internal_tide','inv_bar_cor',
             'iono_cor_alt','iono_cor_alt_filtered','l2_record_counter','latitude','load_tide_sol1','load_tide_sol2','longitude',
             'manoeuvre_flag','mean_dynamic_topography','mean_dynamic_topography_acc','mean_dynamic_topography_qual',
             'mean_sea_surface_sol1','mean_sea_surface_sol1_acc','mean_sea_surface_sol1_qual','mean_sea_surface_sol2',
             'mean_sea_surface_sol2_acc','mean_sea_surface_sol2_qual','meteo_map_availability_flag',
             'model_dry_tropo_cor_measurement_altitude','model_dry_tropo_cor_zero_altitude',
             'model_wet_tropo_cor_measurement_altitude','model_wet_tropo_cor_zero_altitude','ocean_tide_eq','ocean_tide_non_eq',
             'ocean_tide_sol1','ocean_tide_sol1_qual','ocean_tide_sol2','ocean_tide_sol2_qual','off_nadir_pitch_angle_pf',
             'off_nadir_roll_angle_pf','off_nadir_yaw_angle_pf','orbit_type_flag','pass_direction_flag','pole_tide',
             'rad_cloud_liquid_water','rad_cloud_liquid_water_qual','rad_distance_to_land','rad_land_frac_187',
             'rad_land_frac_238','rad_land_frac_340','rad_rain_flag','rad_sea_ice_flag','rad_state_oper_flag',
             'rad_surface_type_flag','rad_tb_187','rad_tb_187_qual','rad_tb_238','rad_tb_238_qual','rad_tb_340',
             'rad_tb_340_qual','rad_tmb_187','rad_tmb_187_qual','rad_tmb_238','rad_tmb_238_qual','rad_tmb_340',
             'rad_tmb_340_qual','rad_water_vapor','rad_water_vapor_qual','rad_wet_tropo_cor','rad_wet_tropo_cor_qual',
             'rad_wind_speed','rad_wind_speed_qual','rain_attenuation','rain_flag','solid_earth_tide',
             'surface_classification_flag','time','time_tai','total_electron_content','wind_speed_alt',
             'wind_speed_mod_u','wind_speed_mod_v']
    return vrk,vrc,vr1



def product_filter(VERSION,MISSIONNAME):
    '''
    Purpose: List variables to extract from product, and identify which of the extracted variables require a filtration (idVAR_filt), which don’t (idVAR_app) and which belong to ancillary data (idVARCYC). 
    Set thresholds for variables that require filtration (dsVAR). 
    '''
    ### Variables with filtration criteria applied
    # Cycle info
    idVARCYC = ['pass_number','equator_time','equator_longitude','first_meas_time','last_meas_time']
    # Data tracking
    VARTRK = ['time','lat','lon'] 
    # Model parameters
    VARPAR = ['swh_ku','sig0_ku','wind_speed_alt']
    # Altimeter
    VARALT = ['alt','range_ku','sea_state_bias_ku','ssha']
    # Atmospheric corrections
    VARATM = ['rad_wet_tropo_corr','model_dry_tropo_corr','iono_corr_gim_ku']
    # Tides
    VARTID = ['pole_tide','solid_earth_tide','ocean_tide_got','ocean_tide_non_eq']
    # Surface
    VARMSS = ['mean_sea_surface'] #_cnescls
    # Numval
    VARNUM = ['range_numval_ku']
    # RMS (1Hz vs. 20Hz)
    VARRMS = ['range_rms_ku','sig0_rms_ku','swh_rms_ku']
    # For HRM (SAR)
    if MISSIONNAME != 's6Ah':
        VARPAR = VARPAR+['swh_c','sig0_c']
        VARALT = VARALT+['range_c','sea_state_bias_c','off_nadir_angle_wf_ku']
        VARNUM = VARNUM+['range_numval_c']
        VARRMS = VARRMS+['range_rms_c','sig0_rms_c','swh_rms_c']
    #Flags (filter for ocean applications only) 'surf_class'-->[0 1 2 3 4 5 6]=open_ocean land continental_water aquatic_vegetation continental_ice_snow floating_ice salted_basin
    if MISSIONNAME in ['s6A','s6Ah']:
        VARFLG = ['surface_classification_flag','rain_flag','rad_surf_type']
    else:
        VARFLG = ['surface_classification_flag','ice_flag','rain_flag','rad_surf_type']
    # retracked-data-specific
    if VERSION in [1]:
        VARVER = ['inv_bar_corr','hf_fluctuations_corr','alt_echo_type','rad_distance_to_land','rad_sea_ice_flag','rad_rain_flag','ocean_tide_fes']
    elif VERSION in [2,0]:
        VARVER = ['dac','internal_tide_hret','alt_echo_type','rad_distance_to_land','rad_sea_ice_flag','rad_rain_flag','ocean_tide_fes']
    elif VERSION in [3]:
        VARVER = ['inv_bar_corr','dac','internal_tide_hret','rad_distance_to_land','rad_sea_ice_flag','rad_rain_flag','ocean_tide_fes']
    elif VERSION in [4]:
	    VARVER = ['inv_bar_corr','hf_fluctuations_corr','rad_distance_to_land','instr_op_mode','surf_class','rad_along_track_avg_flag']
    # MLE3
    VARPAR2 = []#['swh_ku_mle3','sig0_ku_mle3','wind_speed_rad','wind_speed_alt_mle3']
    VARALT2 = []#['ssha_mle3','range_ku_mle3']
    VARATM2 = ['iono_corr_alt_ku','model_wet_tropo_corr']
    VARTID2 = ['load_tide_got','load_tide_fes']
    VARMSS2 = ['mean_dynamic_topography','geoid','bathymetry'] # for deep water: bathymetry>= 1000m]
    VARNUM2 = []
    VARRMS2 = []#['range_rms_ku_mle3','sig0_rms_ku_mle3','swh_rms_ku_mle3','range_numval_ku_mle3']#'agc_rms_ku','agc_rms_c'
    ### Variables that are not passed through a filter
    # retracked-data-specific
    if VERSION in [1]:
        if 'j1' not in MISSIONNAME:
            VARVER2 = ['swh_ku_mle3','sig0_ku_mle3','wind_speed_rad','wind_speed_alt_mle3','ssha_mle3','range_ku_mle3','iono_corr_alt_ku_mle3',
		    'range_rms_ku_mle3','sig0_rms_ku_mle3','swh_rms_ku_mle3','range_numval_ku_mle3']
        else:
	        VARVER2 = []
    elif VERSION in [2,0]:
        VARVER2 = ['altitude_cnes','composite_wet_tropo_gpd','mean_sea_surface_dtu','distance_to_coast','off_nadir_angle_wf_rms_ku',
		    'range_rms_ku_mle3','sig0_rms_ku_mle3','swh_rms_ku_mle3','range_numval_ku_mle3',
		    'swh_ku_mle3','sig0_ku_mle3','wind_speed_rad','wind_speed_alt_mle3','ssha_mle3','range_ku_mle3','iono_corr_alt_ku_mle3']
    elif VERSION in [3]:
        if MISSIONNAME=='j3':
            VARVER2 = ['swh_ku_mle3','sig0_ku_mle3','wind_speed_rad','wind_speed_alt_mle3','ssha_mle3','range_ku_mle3','iono_corr_alt_ku_mle3',
		   'range_rms_ku_mle3','sig0_rms_ku_mle3','swh_rms_ku_mle3','range_numval_ku_mle3',
		   'sst','mean_wave_period_t02','mean_wave_direction','sea_ice_concentration',
                   'angle_of_approach_to_coast','wave_model_map_availability_flag','altitude_rate',
                   'model_dry_tropo_cor_measurement_altitude','model_wet_tropo_cor_measurement_altitude','surface_slope_cor',
                   'distance_to_coast','sea_state_bias_3d_mp2','sea_state_bias_adaptive_3d_mp2','wvf_main_class','off_nadir_angle_wf_rms_ku']#'mean_sea_surface_dtu',
        elif MISSIONNAME =='s6A':
            VARVER2 = ['swh_ku_mle3','sig0_ku_mle3','wind_speed_rad','wind_speed_alt_mle3','ssha_mle3','range_ku_mle3','iono_corr_alt_ku_mle3',
                               'range_rms_ku_mle3','sig0_rms_ku_mle3','swh_rms_ku_mle3','range_numval_ku_mle3','sea_state_bias_mle3',
                               'angle_of_approach_to_coast','altitude_rate','iono_corr_gim_c',
                               'pass_direction_flag','rad_cloud_liquid_water','total_electron_content',
                               'off_nadir_pitch_angle_pf','off_nadir_roll_angle_pf','off_nadir_yaw_angle_pf',
                               'model_dry_tropo_cor_measurement_altitude','model_wet_tropo_cor_measurement_altitude',
                               'distance_to_coast','off_nadir_angle_wf_rms_ku','wind_speed_mod_v','manoeuvre_flag','mean_sea_surface_dtu']
        elif MISSIONNAME =='s6Ah':
            VARVER2 = ['wind_speed_rad','angle_of_approach_to_coast','altitude_rate',
                               'pass_direction_flag','rad_cloud_liquid_water','total_electron_content',
                               'off_nadir_pitch_angle_pf','off_nadir_roll_angle_pf','off_nadir_yaw_angle_pf',
                               'model_dry_tropo_cor_measurement_altitude','model_wet_tropo_cor_measurement_altitude',
                               'distance_to_coast','wind_speed_mod_v','manoeuvre_flag','mean_sea_surface_dtu']
    elif VERSION in [4]:
        VARVER2 = ['off_nadir_angle_wf_rms_ku','UTC_day','off_nadir_roll_angle_pf','off_nadir_pitch_angle_pf','off_nadir_yaw_angle_pf',
                   'mean_sea_surface_dtu','climato_use_flag_ku','swh_ku_plrm','range_ku_plrm',
                   'sig0_ku_plrm','wind_speed_alt_plrm','sea_state_bias_ku_plrm','ssha_plrm','rad_wet_tropo_corr_plrm',
                   'range_numval_ku_plrm','range_rms_ku_plrm','sig0_rms_ku_plrm','swh_rms_ku_plrm','ice_flag_plrm',
                   'rain_flag_plrm','iono_corr_alt_ku_plrm','off_nadir_angle_wf_rms_ku_plrm','off_nadir_angle_wf_ku_sar','ocean_tide_fes']
    # Thresholds for filtered variables
    idVAR_filt = VARTRK+VARPAR+VARALT+VARATM+VARTID+VARMSS+VARNUM+VARRMS+VARFLG+VARVER
    idVAR_app = VARPAR2+VARALT2+VARATM2+VARTID2+VARMSS2+VARNUM2+VARRMS2+VARVER2
    dsVAR={}
    dsVAR['swh_ku'] = [0,20]
    dsVAR['swh_c'] = [0,20]
    dsVAR['sig0_ku'] = [0,40]
    dsVAR['sig0_c'] = [0,40]
    dsVAR['wind_speed_alt'] = [0,30]
    dsVAR['off_nadir_angle_wf_ku'] = [-0.20,0.64]
    dsVAR['rad_wet_tropo_corr'] = [-0.5,-0.001] 
    dsVAR['model_dry_tropo_corr'] = [-2.5,-1.9]
    dsVAR['iono_corr_gim_ku'] = [-0.4,0.04]
    dsVAR['pole_tide'] = [-0.15,0.15] 
    dsVAR['solid_earth_tide'] = [-1,1] 
    dsVAR['ocean_tide_got'] = [-5,5] #GOT
    dsVAR['ocean_tide_fes'] = [-5,5] #FES
    dsVAR['load_tide_got'] = [-0.5,0.5] #GOT (already in geocentric ocean tide)
    dsVAR['load_tide_fes'] = [-0.5,0.5] # FES (already in geocentric ocean tide)
    dsVAR['dac'] = [-2,2]# [-0.4,0.6] NEW CHANGE (2020-11-18)
    dsVAR['inv_bar_corr'] = [-0.4,0.6]
    dsVAR['range_numval_ku'] = [10.,21.5]#[10.0,20.5] # number of measurements used in the 1Hz measurements
    dsVAR['range_numval_c'] = [10.,21.5]#[10.0,20.5] 
    dsVAR['range_rms_ku'] = [0.0,0.25]#0.2]
    dsVAR['range_rms_c'] = [0.0,0.5]#0.4]
    if VERSION!=4:
        dsVAR['sig0_rms_ku'] = [0.0,1.0]
        dsVAR['sig0_rms_c'] = [0.0,1.0]
        dsVAR['swh_rms_ku'] = [0.0,1.5]
        dsVAR['swh_rms_c'] = [0.0,3.0]#2.1]
    else:
        dsVAR['sig0_rms_ku'] = [0.0,5.0]
        dsVAR['sig0_rms_c'] = [0.0,5.0]
        dsVAR['swh_rms_ku'] = [0.0,1.5]
        dsVAR['swh_rms_c'] = [0.0,3.0]#2.1]
    if 's6' in MISSIONNAME:
        dsVAR['swh_rms_c'] = [0.0,5.0]
        dsVAR['range_rms_c'] = [0.0,0.7]
        dsVAR['alt_echo_type'] = [0]
        dsVAR['surface_classification_flag'] = [0]
        dsVAR['ice_flag'] = [0]
        dsVAR['rad_sea_ice_flag'] = [0]
        dsVAR['rad_rain_flag'] = [0]
        dsVAR['rain_flag'] = [0]
    if 's3' in MISSIONNAME:
        #######
        # Sentinel-3 Specific
        dsVAR['instr_op_mode']=[1]
        dsVAR['surf_class']=[0]
        dsVAR['rad_along_track_avg_flag']=[0]
    return dsVAR,idVAR_filt,idVAR_app,idVARCYC


def nomenclature_adjustment(ds,MISSIONNAME,RTRK,VAR,VERSION,directory=False,VAR_PARENT=[]): 
    '''
    Purpose: run function for rtrk_variables and rtrk_variable_match to extract the data (vr_out) of a specific variable (VAR)
    ds = netCDF4 pass file from product
    directory = False to accommodate RTRK = adaptive. Otherwise, directory = True.
    VAR_PARENT = variable from  idVAR_filt (product_filter) else []
    '''
    kband = ['swh_ku','sig0_ku','range_ku','sea_state_bias_ku','range_numval_ku','range_rms_ku','sig0_rms_ku','swh_rms_ku','iono_corr_gim_ku']
    cband = ['swh_c','sig0_c','range_c','sea_state_bias_c','range_numval_c','range_rms_c','sig0_rms_c','swh_rms_c','iono_corr_gim_c']
    if RTRK=='adaptive':
        raise('not enough information known about adaptive - why isnt there a corresponding C-band range?')
    # datasets
    if VERSION==3:
        ds1=ds.groups['data_01'].variables
        dsk=ds.groups['data_01'].groups['ku'].variables
        if MISSIONNAME != 's6Ah':
            dsc=ds.groups['data_01'].groups['c'].variables
        vrk,vrc,vr1=rtrk_variables(RTRK,MISSIONNAME)
    else:
        ds1=ds.variables
    if VERSION == 4:
        if VAR == 'interp_flag_mss_01':
            VAR = 'interp_flag_mss_sol1_01'
    #variables to datasets
    LIB = rtrk_variable_match(MISSIONNAME,RTRK,VERSION)
    if VAR in LIB.keys():
        VAR_sv = LIB[VAR]
    else:
        VAR_sv=VAR
    if VERSION==3:
      if MISSIONNAME=='s6Ah':
        # if variables is only in Ku
        if VAR_sv in vrk and VAR_sv not in vr1:
            if RTRK == 'adaptive' and VAR=='ssha':
                vr_out=dsk['ssha']
            else:
                vr_out=dsk[VAR_sv]
        # if variable in 1 Hz data
        elif VAR_sv in vr1:
            vr_out=ds1[VAR_sv]
        else:
            print(VAR_sv)
            print(VAR_PARENT)
      else:
        # if variables is only in Ku
        if VAR_sv in vrk and VAR_sv not in vrc:
            if RTRK == 'adaptive' and VAR=='ssha':
                vr_out=dsk['ssha']
            else:
                vr_out=dsk[VAR_sv]
        # if variable is only in C
        elif VAR_sv in vrc and VAR_sv not in vrk:
            vr_out=dsc[VAR_sv]
        # if variable is in both Ku and C
        elif VAR_sv in vrc and VAR_sv in vrk:
            # if variable is one of the variables of interest
            if np.size(VAR_PARENT)==0:
                if VAR in kband:
                    vr_out=dsk[VAR_sv]
                elif VAR in cband:
                    vr_out=dsc[VAR_sv]
                else:
                    print(VAR_sv)
            # if variable is a quality flag
            else:
                if VAR_PARENT in kband:
                    vr_out=dsk[VAR_sv]
                elif VAR_PARENT in cband:
                    vr_out=dsc[VAR_sv]
                else:
                    print(VAR_sv)
                print(VAR_PARENT)
                # if variable in 1 Hz data
        elif VAR_sv in vr1:
            vr_out=ds1[VAR_sv]
        else:
            print(VAR_sv)
        print(VAR_PARENT)
        # dataset of older retracking files
    else:
        vr_out=ds1[VAR_sv]
    if directory==False:
        if RTRK == 'adaptive' and VAR=='ssha':
            vr_out=dsk['ssha'][:]+(dsk['range_ocean'][:]-dsk['range_'+RTRK][:])+(dsk['iono_cor_alt_filtered'][:]-dsk['iono_cor_alt_filtered_'+RTRK][:])+(dsk['sea_state_bias'][:]-dsk['sea_state_bias'+RTRK][:])
        else:
            vr_out = vr_out[:]
    return vr_out


def filter_product(MISSIONNAME,ds,RTRK,VERSION):
    '''
    Purpose: find indices (idx) of valid data (valid = all measurements that pass all filtering criteria, unmasked and pass quality flags)
    ds = netCDF4 pass file from product
    NC_attr = variable attributes for variables in idVAR
    idVAR = list of variable names --> idVAR_filt+idVAR_app (see product_filter function)
    '''
    # pull variable name
    dsVAR,idVAR_filt,idVAR_err,idVARCYC=product_filter(VERSION,MISSIONNAME)
    alt=nomenclature_adjustment(ds,MISSIONNAME,RTRK,'alt',VERSION)
    rngK=nomenclature_adjustment(ds,MISSIONNAME,RTRK,'range_ku',VERSION)
    aMr = alt-rngK
    Nv = np.shape(idVAR_filt)[0]
    Nve = np.shape(idVAR_err)[0]
    Nd = np.shape(alt)[0] #ds[idVAR_filt[0]].shape[0]
    # INDICES FOR FILTER VARIABLES AND PARAMETERS
    idx_var = np.where((aMr>=-130.)&(aMr<=100.))[0]
    idx_ma = np.arange(Nd)
    Norig = np.shape(idx_ma)[0]
    for ii in np.arange(Nv):
        # find indices without masks
        var=nomenclature_adjustment(ds,MISSIONNAME,RTRK,idVAR_filt[ii],VERSION)
        idx_mai = np.where((ma.getmaskarray(var) == False))[0]
        #print(idVAR_filt[ii]+' unmasked elements: '+str(np.size(idx_mai))+'/'+str(Norig))
        if idVAR_filt[ii] == 'iono_cor_gim_ku':
            print('unmasked elements for '+idVAR_filt[ii]+': '+str(np.size(idx_mai)))
        idx_ma = np.intersect1d(idx_ma,idx_mai)
        # find indices within the limits of certain variables
        if idVAR_filt[ii] in dsVAR.keys():
            szVAR = np.size(dsVAR[idVAR_filt[ii]])
            if szVAR!=0:
                if szVAR==1:
                    idx_ii = np.where(var==dsVAR[idVAR_filt[ii]])[0]
                elif szVAR==2:
                    idx_ii = np.where((var>=dsVAR[idVAR_filt[ii]][0])&(var<=dsVAR[idVAR_filt[ii]][1]))[0]
                else:
                    warnings.warn('range limit has too many variables')
                idx_var = np.intersect1d(idx_var,idx_ii)
                print(idVAR_filt[ii]+', limits '+str(dsVAR[idVAR_filt[ii]])+': '+str(np.size(idx_ii))+'/'+str(Norig))
    #print('Number of valid points (variable filter): '+str(np.size(idx_var)))
    # INDICES FOR FILTER USING FLAGS
    NC_attr={}
    NC_attr['quality_flag']={}
    NC_attr['scale_factor']={}
    NC_attr['add_offset']={}
    NC_attr['units']={}
    for ii in np.arange(Nv):
        var_directory=nomenclature_adjustment(ds,MISSIONNAME,RTRK,idVAR_filt[ii],VERSION,directory=True)
        atts=dir(var_directory)
        if 'quality_flag' in atts:
            NC_attr['quality_flag'][idVAR_filt[ii]] = var_directory.quality_flag.encode()
        else:
            NC_attr['quality_flag'][idVAR_filt[ii]] = 'none'
        if 'scale_factor' in atts:
            NC_attr['scale_factor'][idVAR_filt[ii]] = var_directory.scale_factor
        else:
            NC_attr['scale_factor'][idVAR_filt[ii]] = 1.0
        if 'add_offset' in atts:
            NC_attr['add_offset'][idVAR_filt[ii]] = var_directory.add_offset
        else:
            NC_attr['add_offset'][idVAR_filt[ii]] = 0.0
        if 'units' in atts:
            NC_attr['units'][idVAR_filt[ii]] = var_directory.units.encode()
        else:
            NC_attr['units'][idVAR_filt[ii]] = 'none'

    # COMPILE INDICES USING QUALITY FLAGS
    idx_ii = np.arange(Nd)
    idx_flgs = np.arange(Nd)
    for ii in np.arange(Nv):
        QF = NC_attr['quality_flag'][idVAR_filt[ii]]
        spQF = QF.split()
        szQF = np.shape(spQF)[0]
        #print('quality flag with flag_meaning: '+QF+' of size '+str(szQF))
        if QF != 'none':
    	    if szQF == 1: #'flag_meanings' in QF_atts:
                var_directory=nomenclature_adjustment(ds,MISSIONNAME,RTRK,QF,VERSION,directory=True,VAR_PARENT=idVAR_filt[ii])
                var=nomenclature_adjustment(ds,MISSIONNAME,RTRK,QF,VERSION,directory=False,VAR_PARENT=idVAR_filt[ii])
                QF_atts = [jj.encode() for jj in var_directory.ncattrs()]
                if 'flag_meanings' in QF_atts:
                    if 'good' and 'bad' in var_directory.flag_meanings.encode(): # 'good bad' covers most quality flags
                        idx_ii = np.where(var==0)[0]
                        idx_flgs = np.intersect1d(idx_flgs,idx_ii)
                    elif 'four_points' in var_directory.flag_meanings.encode():
                        idx_ii = np.where(var==4)[0]
                        idx_flgs = np.intersect1d(idx_flgs,idx_ii)
                    elif 'Good' and 'Bad' in var_directory.flag_meanings.encode(): # Sentinel-3
                        idx_ii = np.where(var==0)[0]
                        idx_flgs = np.intersect1d(idx_flgs,idx_ii)
                    elif 'Valid' and 'Invalid' in var_directory.flag_meanings.encode(): # Sentinel-3
                        idx_ii = np.where(var==0)[0]
                        idx_flgs = np.intersect1d(idx_flgs,idx_ii)
                    elif QF=='iono_cor_alt_filtered_qual':
                        idx_ii = np.where(var==0)[0]
                        idx_flgs = np.intersect1d(idx_flgs,idx_ii)
                    elif QF=='orb_state_flag_rest':
                        idx_ii = np.where(var==3)[0]
                        idx_flgs = np.intersect1d(idx_flgs,idx_ii)
                    elif QF=='manoeuvre_flag':
                        idx_ii = np.where(var==0)[0]
                        idx_flgs = np.intersect1d(idx_flgs,idx_ii)
                    else:
                        print(idVAR_filt[ii])
                        print(QF)
                        raise('unfamiliar with flag type: '+QF)
                        #print('unflagged elements for '+QF+': '+str(np.size(idx_ii))+' from '+str(np.size(var)))
                else:
                    for vr in spQF:
                        if vr !='and' and vr!='or' and vr!='orb_state_flag_diode' and vr!='orb_state_diode_flag':
                            var_directory=nomenclature_adjustment(ds,MISSIONNAME,RTRK,vr,VERSION,directory=True,VAR_PARENT=idVAR_filt[ii])
                            var=nomenclature_adjustment(ds,MISSIONNAME,RTRK,vr,VERSION,directory=False,VAR_PARENT=idVAR_filt[ii])
                            QF_atts = [jj.encode() for jj in var_directory.ncattrs()]
                            if 'flag_meanings' in QF_atts:
                                if 'good' and 'bad' in var_directory.flag_meanings.encode():
                                    idx_ii = np.where(var==0)[0]
                                    idx_flgs = np.intersect1d(idx_flgs,idx_ii)
                                elif 'four_points' in var_directory.flag_meanings.encode():
                                    idx_ii = np.where(var==4)[0]
                                    idx_flgs = np.intersect1d(idx_flgs,idx_ii)
                                elif vr=='iono_cor_alt_filtered_qual':
                                    idx_ii = np.where(var==0)[0]
                                    idx_flgs = np.intersect1d(idx_flgs,idx_ii)
                                elif vr == 'orb_state_flag_rest':
                                    idx_ii = np.where(var==3)[0]
                                    idx_flgs = np.intersect1d(idx_flgs,idx_ii)
                                elif QF=='manoeuvre_flag':
                                    idx_ii = np.where(var==0)[0]
                                    idx_flgs = np.intersect1d(idx_flgs,idx_ii)
                                else:
                                    warnings.warn('unfamiliar with float type')
                                    #print('unflagged elements for '+vr+': '+str(np.size(idx_ii))+' from '+str(np.size(var)))
    # INCLUDE VARIABLES FOR ERROR ANALYSIS
    for ii in np.arange(Nve):
        var_directory=nomenclature_adjustment(ds,MISSIONNAME,RTRK,idVAR_err[ii],VERSION,directory=True)
        atts=dir(var_directory)
        if 'quality_flag' in atts:
            NC_attr['quality_flag'][idVAR_err[ii]] = var_directory.quality_flag.encode()
        else:
            NC_attr['quality_flag'][idVAR_err[ii]] = 'none'
        if 'scale_factor' in atts:
            NC_attr['scale_factor'][idVAR_err[ii]] = var_directory.scale_factor
        else:
            NC_attr['scale_factor'][idVAR_err[ii]] = 1.0
        if 'add_offset' in atts:
            NC_attr['add_offset'][idVAR_err[ii]] = var_directory.add_offset
        else:
            NC_attr['add_offset'][idVAR_err[ii]] = 0.0
        if 'units' in atts:
            NC_attr['units'][idVAR_err[ii]] = var_directory.units.encode()
        else:
            NC_attr['units'][idVAR_err[ii]] = 'none'
    # COMPILE VARIABLE LIMIT INDICES WITH UNMASKED INDICES
    idx_varma = np.intersect1d(idx_var,idx_ma)
    # COMPILE VARIABLE LIMIT INDICES AND UNMASKED INDICES WITH QUALITY FLAG INDICES
    idx = np.intersect1d(idx_varma,idx_flgs)
    print(ii)
    #print('Number of valid points (variable+flags filter): '+str(np.size(idx)))
    idVAR = idVAR_filt+idVAR_err
    print('index sizes: total='+str(np.size(idx))+', idx_var='+str(np.size(idx_var))+', idx_ma='+str(np.size(idx_ma))+', idx_flgs='+str(np.size(idx_flgs)))
    #print('Number of valid measurements/total measurements: '+str(np.shape(idx)[0])+'/'+str(Nd))
    return idx,NC_attr,idVAR

###############################################
###############################################
#-----------------------------------------------------------------------
# [Step 2] Filter data and save as 'direct' file
#-----------------------------------------------------------------------
def skew_measure(ds):
    '''
    Purpose: Measure the skewness, median, mean and standard deviation (skew,md,mn,sd) for every group of 20-Hz measurements used to estimate a single 1-Hz measurement.
    ds = netCDF4 pass file from product
    '''
    swh = ds['swh_ku'][:]
    iswh20 = ds['swh_used_20hz_ku'][:]
    swh20 = ds['swh_20hz_ku'][:]
    N = np.shape(swh)[0]
    skew = np.empty(N)*np.nan
    md = np.empty(N)*np.nan
    mn = np.empty(N)*np.nan
    sd = np.empty(N)*np.nan
    for ii in np.arange(N):
        idx = np.where(iswh20[ii,:]==0)[0]
        if np.size(idx):
            skew[ii] = stats.skew(swh20[ii,idx])
            md[ii] = np.median(swh20[ii,idx])
            mn[ii] = np.mean(swh20[ii,idx])
            sd[ii] = np.std(swh20[ii,idx])
    return skew,md,mn,sd

def product2direct(nCYC,MISSIONNAME,SOURCE,RTRK,VERSION,ALTINTERP):	
    '''
    Purpose: convert pass-by-pass product datasets to filtered, by-cycle datasets (step-1 in documentation – product2data)
    NC = library containing dataset to save
    NC_attr = library containing attributes corresponding to NC
    idVAR = list of variable names --> idVAR_filt+ idVAR_app (see product_filter function)
    '''
    t1 = time.time()
    # PRODUCT FILE HEADER AND PATH
    HEAD = ldic.aviso_file_starter(MISSIONNAME,int(nCYC),VERSION)
    CYC = '{:03d}'.format(nCYC)#'%03d' %nCYC
    PTH_src = ldic.product_paths(MISSIONNAME,SOURCE,CYC,VERSION)
    # VARIABLE NAMES
    dsVAR,idVAR_filt,idVAR_err,idVARCYC=product_filter(VERSION,MISSIONNAME)
    idVAR = idVAR_filt+idVAR_err
    # FILE SIZES 
    Nv = np.shape(idVAR)[0]
    if MISSIONNAME == 's3':
        PASSES = np.setdiff1d(np.arange(1,386),[155,156])#np.arange(1,386)#np.setdiff1d(np.arange(1,386),[84,87])#np.arange(1,386)
    else:
        PASSES = np.arange(1,255) #np.asarray([43,81,83])
    Np = np.shape(PASSES)[0]
    # PREDEFINE VARIABLE NAMES FOR OUTPUT FILE
    NC = {}
    for jj in np.arange(Nv):
        NC[idVAR[jj]] = []
    NC['pass_number'] = []
    NC['equator_longitude'] = []
    # ADDITIONAL VARIABLES: parameter statistics and JPL orbit solution 
    if MISSIONNAME !='tx' and VERSION!=3:
        #comment for TOPEX and Jason-3 GDRF
        NC['swh_skew_ku'] = []
        NC['swh_mean_ku'] = []
        NC['swh_median_ku'] = []
        NC['swh_std_ku'] = []
        if ALTINTERP=='full': #if 'j' in MISSIONNAME:
            ORB_USE_ONCE,fn_orb,pth_orb_sv,fn_orb2,split_cyc = ldic.orbit_paths(MISSIONNAME,CYC)
            #print('use once: '+ORB_USE_ONCE+fn_orb)
            #print('save as: '+pth_orb_sv+fn_orb)
            os.system('cp '+ORB_USE_ONCE+fn_orb+' '+pth_orb_sv+fn_orb)
            orbit_file = pth_orb_sv+fn_orb[:-3]
            os.system('gzip -d '+pth_orb_sv+fn_orb)
            if np.size(fn_orb2)!=0:
                os.system('cp '+ORB_USE_ONCE+fn_orb2+' '+pth_orb_sv+fn_orb2)
                orbit_file2 = pth_orb_sv+fn_orb2[:-3]
                os.system('gzip -d '+pth_orb_sv+fn_orb2)
            print('orbit interpolation of '+ORB_USE_ONCE+fn_orb+' to '+pth_orb_sv+fn_orb)
        if ALTINTERP=='full' or ALTINTERP=='part':
            NC['lat_nc'] = []
            NC['lon_nc'] = []
            NC['alt_nc'] = []
            NC['lat_jpl'] = []
            NC['lon_jpl'] = []
            NC['alt_jpl'] = []
            NC['dlat'] = []
            NC['dlon'] = []
            NC['dalt'] = []
    # STACK ALL FILTERED DATA INTO CYCLES
    for ii in np.arange(Np):
        # CALL FILE NAME BY CYCLE
        PASS = '{:03d}'.format(PASSES[ii])
        ### !!! TEMPORARY CONDITION !!
        if 's6' in MISSIONNAME:
            print('condition check for '+MISSIONNAME)
            ende_temp = 'F06_unvalidated.nc'
        else:
            ende_temp = '.nc'
        nc = glob(PTH_src+HEAD+CYC+'_'+PASS+'*'+ende_temp)
        print(PTH_src+HEAD+CYC+'_'+PASS)
        print('size nc:'+str(np.size(nc)))
        if np.size(nc)!=0:
            print(nc[0])
            ds = Dataset(nc[0])
            # FILTER ALL PASS FILES WITHIN A SINGLE CYCLE
            idx,NC_attr,idVAR = filter_product(MISSIONNAME,ds,RTRK,VERSION)
            #print('idx size: '+str(idx))
            # STACK DATA
            Nidx = np.size(idx)
            if Nidx!=0:
                for jj in np.arange(Nv):
                    var=nomenclature_adjustment(ds,MISSIONNAME,RTRK,idVAR[jj],VERSION,directory=False)
                    NC[idVAR[jj]] = np.hstack((NC[idVAR[jj]],np.take(var,idx)))
            #print('variable size: '+str(np.size(NC['swh_ku'][:])))
            NC['pass_number'] = np.hstack((NC['pass_number'],[ds.pass_number]*Nidx))
            NC['equator_longitude'] = np.hstack((NC['equator_longitude'],[ds.equator_longitude]*Nidx))
    	    # ADDITIONAL VARIABLES: parameter statistics and JPL orbit solution
            if MISSIONNAME!='tx' and VERSION!=3:
                #comment for TOPEX and Jason-3 GDRF
                # Create additional variables
                nv_skew,nv_md,nv_mn,nv_sd = skew_measure(ds)
                NC['swh_skew_ku'] = np.hstack((NC['swh_skew_ku'],np.take(nv_skew,idx)))
                NC['swh_mean_ku'] = np.hstack((NC['swh_mean_ku'],np.take(nv_mn,idx)))
                NC['swh_median_ku'] = np.hstack((NC['swh_median_ku'],np.take(nv_md,idx)))
                NC['swh_std_ku'] = np.hstack((NC['swh_std_ku'],np.take(nv_sd,idx)))
                # Orbit Data
                if ALTINTERP=='full' or ALTINTERP=='part': #if 'j' in MISSIONNAME:
                    ORB_USE_ONCE,fn_orb,pth_orb_sv,fn_orb2,split_cyc = ldic.orbit_paths(MISSIONNAME,CYC)
                    orbit_file = pth_orb_sv+fn_orb[:-3]
                    if ALTINTERP=='full':
                        os.system('./orbitinterp.e -posgoafile '+orbit_file+' -ncfile '+nc[0]+' > '+orbit_file[:-4]+'pass_'+PASS+'_altitude.txt')
                    f = open(orbit_file[:-4]+'pass_'+PASS+'_altitude.txt',"r")
                    lines = f.readlines()
                    print('open: '+orbit_file)
                    print('open: '+nc[0])
                    print('convert to: '+orbit_file[:-4]+'pass_'+PASS+'_altitude.txt')
                    Nl = np.shape(lines)[0]
                    print('size Nl: '+str(Nl))
                    Anlat,Anlon,Analt,Aolat,Aolon,Aoalt,Adlat,Adlon,Adalt = [],[],[],[],[],[],[],[],[]
                    for kk in np.arange(Nl):
                        if np.shape(lines[kk].split())[0] ==10:
                            fidx,nlat,nlon,nalt,olat,olon,oalt,dlat,dlon,dalt = lines[kk].split()
                            Anlat= np.hstack((Anlat,np.float(nlat)))
                            Anlon = np.hstack((Anlon,np.float(nlon)))
                            Analt = np.hstack((Analt,np.float(nalt)))
                            Aolat = np.hstack((Aolat,np.float(olat)))
                            Aolon = np.hstack((Aolon,np.float(olon)))
                            Aoalt = np.hstack((Aoalt,np.float(oalt)))
                            Adlat = np.hstack((Adlat,np.float(dlat)))
                            Adlon = np.hstack((Adlon,np.float(dlon)))
                            Adalt = np.hstack((Adalt,np.float(dalt)))
                    if np.size(fn_orb2)!=0:
                        orbit_file2 = pth_orb_sv+fn_orb2[:-3]
                        if ALTINTERP=='full':
                            os.system('./orbitinterp.e -posgoafile '+orbit_file2+' -ncfile '+nc[0]+' > '+orbit_file2[:-4]+'pass_'+PASS+'_altitude.txt')
                        f2 = open(orbit_file2[:-4]+'pass_'+PASS+'_altitude.txt',"r")
                        lines2 = f2.readlines()
                        Nl2 = np.shape(lines2)[0]
                        for kk2 in np.arange(Nl2):
                            if np.shape(lines2[kk2].split())[0] ==10:
                                fidx2,nlat2,nlon2,nalt2,olat2,olon2,oalt2,dlat2,dlon2,dalt2 = lines[kk2].split()
                                Anlat= np.hstack((Anlat,np.float(nlat2)))
                                Anlon = np.hstack((Anlon,np.float(nlon2)))
                                Analt = np.hstack((Analt,np.float(nalt2)))
                                Aolat = np.hstack((Aolat,np.float(olat2)))
                                Aolon = np.hstack((Aolon,np.float(olon2)))
                                Aoalt = np.hstack((Aoalt,np.float(oalt2)))
                                Adlat = np.hstack((Adlat,np.float(dlat2)))
                                Adlon = np.hstack((Adlon,np.float(dlon2)))
                                Adalt = np.hstack((Adalt,np.float(dalt2)))
                    print('indices should be same size '+str(np.shape(ds.variables['swh_ku'][:])[0])+' != '+str(np.shape(Anlon)[0]))
                    if np.shape(ds.variables['swh_ku'][:])[0]!=np.shape(Anlon)[0]:
                       raise('indices should be same size '+str(np.shape(ds.variables['swh_ku'][:])[0])+' != '+str(np.shape(Anlon)[0]))
                    NC['lat_nc'] = np.hstack((NC['lat_nc'],np.take(Anlat,idx)))
                    NC['lon_nc'] = np.hstack((NC['lon_nc'],np.take(Anlon,idx)))
                    NC['alt_nc'] = np.hstack((NC['alt_nc'],np.take(Analt,idx)))
                    NC['lat_jpl'] = np.hstack((NC['lat_jpl'],np.take(Aolat,idx)))
                    NC['lon_jpl'] = np.hstack((NC['lon_jpl'],np.take(Aolon,idx)))
                    NC['alt_jpl'] = np.hstack((NC['alt_jpl'],np.take(Aoalt,idx)))
                    NC['dlat'] = np.hstack((NC['dlat'],np.take(Adlat,idx)))
                    NC['dlon'] = np.hstack((NC['dlon'],np.take(Adlon,idx)))
                    NC['dalt'] = np.hstack((NC['dalt'],np.take(Adalt,idx)))
                    NC_attr['units']['lat_nc'] = 'degrees_north'
                    NC_attr['units']['lon_nc'] = 'degrees_east'
                    NC_attr['units']['alt_nc'] = 'm'
                    NC_attr['units']['lat_jpl'] = 'degrees_north'
                    NC_attr['units']['lon_jpl'] = 'degrees_east'
                    NC_attr['units']['alt_jpl'] = 'm'
                    NC_attr['units']['dlat'] = 'diff_degrees_north'
                    NC_attr['units']['dlon'] = 'diff_degrees_east'
                    NC_attr['units']['dalt'] = 'm'
                    NC_attr['units']['swh_skew_ku'] = 'unitless'
                    NC_attr['units']['swh_mean_ku'] = 'm'
                    NC_attr['units']['swh_median_ku'] = 'm'
                    NC_attr['units']['swh_std_ku'] = 'm'
    # CREATE OUTPUT FILE NAME
    FILENAME = ldic.create_filename(MISSIONNAME,'dir',[],'product2data','jpl',str(nCYC),str(nCYC),'na','na','nc')
    #FILENAME = FILENAME[:-3]+'_rad_surf_type_old'+FILENAME[-3:]
    # SAVE OUTPUT FILE
    print(FILENAME)
    print('variable size: '+str(np.size(NC['swh_ku'][:])))
    crt_product2direct(MISSIONNAME,SOURCE,NC,NC_attr,CYC,FILENAME)
    t2 = time.time()
    print('Time to create the along-track cycle '+CYC+' file from JPL/topex_gdrf: '+str(t2-t1))
    return NC,NC_attr,idVAR
###############################################
###############################################
#-----------------------------------------------------------------------
# [Step 3] Run step 2
#-----------------------------------------------------------------------
def run_product2direct(CYCLES,MISSIONNAME,SOURCE,VERSION,ALTINTERP,RTRK='ocean'):
    # Purpose: run function for product2direct
    Nc = np.shape(CYCLES)[0]
    for ii in np.arange(Nc):
        print('Compiling JPL/'+MISSIONNAME+' netCDFs into Direct files, VERSION '+str(VERSION))
        NC,NC_attr,idVAR = product2direct(CYCLES[ii],MISSIONNAME,SOURCE,RTRK,VERSION,ALTINTERP)
    return NC,NC_attr,idVAR
###############################################
###############################################
        

    
    

    


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:08:51 2020

@author: alexaputnam

Author: Alexa Angelina Putnam Shilleh
Institution: University of Colorado Boulder
Software: sea state bias model generator (interpolation method)
Thesis: Sea State Bias Model Development and Error Analysis for Pulse-limited Altimetry
Semester/Year: Fall 2021
"""

import time
import numpy as np
from netCDF4 import Dataset
from os import path

import lib_dict_ext as ldic
import lib_geo_gridding_ext as lgrid
import lib_estimation_ext as lest
import lib_post_processing_ext as lpost
import lib_iono_calib_ext as lion
import lib_filter_ext as lflt
# --------------------------------------------------------------------------#
#-------------------Save models as netCD------------------------------------#
# --------------------------------------------------------------------------#
def crt_matrices2models(METHOD,x_grid,y_grid,z_grid,CYCLES,B_Nd_s,cnt_Nd_s,FE_Nd_s,FILE3,DIM,OMIT,zs_s,zs_cyc_s,B_smooth_s,ce_smooth_s,mse,zs_TECu_alt,zs_TECu_gim,rs_zs_bias,cutoff,deg):
    '''
    Purpose: save SSB models/grids, grid parameters, cycles, formal error, mse, zero-significance and ionosphere calibration bias to a single netCDF file. Applies to DIR, XOV, COL and JNT models.
    x_grid,y_grid,z_grid = x-, y- and z-axis grid nodes (z only provided for 3D models)
    CYCLES = array containing cycle numbers used to create SSB model
    B_Nd_s = raw SSB solution
    cnt_Nd_s = number of observations per node
    FE_Nd_s = formal error for SSB raw modeling
    FILE3 = file name to save
    zs_s = mean zero-significance value (SSB model bias)
    zs_cyc_s = array of zero-significance values for each cycle
    B_smooth_s = smoothed and leveled SSB modeling
    ce_smooth_s = fitting coefficients of parametric SSB model
    mse = mean square error
    zs_TECu_alt = array of by-cycle zero-significance value for dual-frequency TECu.
    zs_TECu_gim = array of by-cycle zero-significance value for GIM TECu 
    rs_zs_bias= final relative ionosphere calibration bias. Mean difference between zs_TECu_gim and zs_TECu_alt.
    cutoff = percentile segment for polynomial fit (see dissertation)
    deg = degree of polynomial fit
    '''
    Nz = np.size(z_grid)
    root_grp = Dataset(FILE3, 'w', format='NETCDF4')
    root_grp.description = "This file contains the generated "+METHOD+" SSB model using the interpolation method and cycles "+str(CYCLES[0])+"-"+str(CYCLES[-1])+"."
    root_grp.history = "Author: Alexa Angelina Putnam Shilleh. Institute: University of Colorado Boulder" + time.ctime(time.time())
    xdim = np.shape(x_grid)[0]
    ydim = np.shape(y_grid)[0]
    cdim = np.shape(CYCLES)[0]
    cedim = np.shape(ce_smooth_s)[0]
    root_grp.createDimension('swh', xdim)
    root_grp.createDimension(DIM, ydim)
    root_grp.createDimension('cyc', cdim)
    root_grp.createDimension('scalar', 1)
    root_grp.createDimension('dual', 2)
    root_grp.createDimension('coeff', cedim)
    x1 = root_grp.createVariable('swh', 'f8', ('swh',))
    x2 = root_grp.createVariable(DIM, 'f8', (DIM,))
    x3 = root_grp.createVariable('cycles', 'f8', ('cyc',))
    x11 = root_grp.createVariable('model_bias','f8',('scalar'))
    x12 = root_grp.createVariable('zerosig','f8',('cyc'))
    if 'jnt' in FILE3:
        x13 = root_grp.createVariable('mse','f8',('cyc','dual'))
    else:
        x13 = root_grp.createVariable('mse','f8',('cyc'))
    x16 = root_grp.createVariable('zs_TECu_alt','f8',('cyc'))
    x17 = root_grp.createVariable('zs_TECu_gim','f8',('cyc'))
    x19 = root_grp.createVariable('iono_calib_bias','f8',('scalar'))
    x1[:] = x_grid
    x2[:] = y_grid
    x3[:] = CYCLES
    x11[:] = zs_s
    x12[:] = zs_cyc_s
    x13[:] = mse
    x16[:] = zs_TECu_alt
    x17[:] = zs_TECu_gim
    x19[:] = rs_zs_bias
    x1.units = "m"
    x1.comment = 'Significant wave height (swh): row dimensions of sea state bias model.'
    if DIM == 'sig0':
        x2.units = "dB"
        x2.comment = 'Backscatter coefficient (sig0): column dimensions of sea state bias model.'
    elif DIM == 'u10':
        x2.units = "m/s"
        x2.comment = 'Wind speed (u10): column dimensions of sea state bias model.'
    x11.units = "m"
    x11.comment = "Bias removed from the sea state bias model (ssb). cutoff("+str(cutoff)+"%), degree("+str(deg)+"). Corresponds to the mean of the zero-significance values (zerosig)."
    x12.units = "m"
    x12.comment = "Zero-significance values for all cycles used to level the sea state bias model. The mean of this value is equal to the bias removed from ssb (zerosig_mean). cutoff("+str(cutoff)+"%), degree("+str(deg)+")"
    x13.units = "m^2"
    x13.comment = "res = np.subtract(y2,H2.dot(B2)), mse = np.sum(wgt*np.square(res))/np.sum(wgt)"
    x16.units = "TECu"
    x16.comments = "dual-frequency TEC corresponding to cutoff=[1,10]% zero-significance using a linear fit"
    x17.units = "TECu"
    x17.comments = "GIM TEC corresponding to cutoff=[1,10]% zero-significance using a linear fit"
    x19.units = "TECu"
    x19.comments = "Ionosphere calibration bias from dTEC (GIM-DF) corresponding to cutoff=[1,10]% zero-significance using a linear fit"
    if Nz != 0:
        zdim = np.shape(z_grid)[0]
        root_grp.createDimension(OMIT, zdim)
        y0 = root_grp.createVariable(OMIT, 'f8', (OMIT,))
        y0[:] = z_grid
        y2 = root_grp.createVariable('ssb_raw', 'f8', ('swh',DIM,OMIT))
        y3 = root_grp.createVariable('count_ssb', 'f8', ('swh',DIM,OMIT))
        y5 = root_grp.createVariable('formal_error', 'f8', ('swh',DIM,OMIT))
        y7 = root_grp.createVariable('ssb', 'f8', ('swh',DIM,OMIT))
        y9 = root_grp.createVariable('ce_smooth', 'f8', ('coeff')) #root_grp.createVariable('formal_error_s_smooth', 'f8', ('swh',DIM,OMIT))
    elif Nz == 0:
        y2 = root_grp.createVariable('ssb_raw', 'f8', ('swh',DIM))
        y3 = root_grp.createVariable('count_ssb', 'f8', ('swh',DIM))
        y5 = root_grp.createVariable('formal_error', 'f8', ('swh',DIM))
        y7 = root_grp.createVariable('ssb', 'f8', ('swh',DIM))
        y9 = root_grp.createVariable('ce_smooth', 'f8', ('coeff')) #root_grp.createVariable('formal_error_s_smooth', 'f8', ('swh',DIM))
    y2[:] = B_Nd_s
    y3[:] = cnt_Nd_s
    y5[:] = FE_Nd_s
    y7[:] = B_smooth_s
    y9[:] = ce_smooth_s
    y2.units = "m"
    y2.comment = "Raw sea state bias model. Raw = not leveled and not smoothed. (assumption: E[SSHA'] = E[ssb] )"
    y3.comment = "Number of observations used to estimate each bin within the sea state bias models"
    y5.units = "m"
    y5.comment = "Unscaled (unitless) formal error of sea state bias model. Scale by multiplying unscaled formal error by RMS of the residuals (~0.077 m)."
    y7.units = "m"
    y7.comment = "Smoothed and leveled sea state bias model. PREFERRED MODEL."
    if Nz != 0:
        y9.units = 'cubic fit to each parameter of the model (i.e.swh,'+DIM+','+OMIT+')'
    elif Nz == 0:
        y9.units = 'cubic fit to each parameter of the model (i.e.swh,'+DIM+')'
    y9.comment = "Fitting coefficients for parametric, apriori sea state bias used to extrapolate and smooth ssb. ssb_apriori = (ce1*swh)+(ce2*swh^2)+(ce3*swh^3)+(ce4*u10)+(ce5*u10^2)+(ce6*u10^3)"
    root_grp.close()


def crt_matrices2models_coe(x_grid,y_grid,z_grid,CYCLES,B_ssb,FE_ssb,cnt_ssb,lat,lon,B_geo,FE_geo,cnt_geo,FN,DIM,OMIT,zs,zs_cyc,B_smooth,ce,zs_TECu_alt,zs_TECu_gim,rs_zs_bias,cutoff,deg):
    '''
    Purpose: save SSB models/grids, grid parameters, spatial error model, spatial parameters, cycles, formal error, mse, zero-significance and ionosphere calibration bias to a single netCDF file. Only applied to JNT model.
    x_grid,y_grid,z_grid = x-, y- and z-axis grid nodes (z only provided for 3D models)
    CYCLES = array containing cycle numbers used to create SSB model
    B_ssb= raw SSB solution
    cnt_ssb = number of observations per node in B_ssb
    FE_ssb= formal error for SSB raw modeling
    lat,lon = reduced Gaussian grid axis nodes
    B_geo = spatio-temporal error model
    FE_geo = formal error of B_geo
    cnt_geo = number of observations per node in B_geo
    FN = file name to save
    zs_s = mean zero-significance value (SSB model bias)
    zs_cyc = array of zero-significance values for each cycle
    B_smooth = smoothed and leveled SSB modeling
    ce = fitting coefficients of parametric SSB model
    zs_TECu_alt = array of by-cycle zero-significance value for dual-frequency TECu.
    zs_TECu_gim = array of by-cycle zero-significance value for GIM TECu 
    rs_zs_bias= final relative ionosphere calibration bias. Mean difference between zs_TECu_gim and zs_TECu_alt.
    cutoff = percentile segment for polynomial fit (see dissertation)
    deg = degree of polynomial fit
    '''
    Nz = np.size(z_grid)
    root_grp = Dataset(FN, 'w', format='NETCDF4')
    root_grp.description = "This file contains the generated coe SSB model using the interpolation method and cycles "+str(CYCLES[0])+"-"+str(CYCLES[-1])+"."
    root_grp.history = "Author: Alexa Angelina Putnam Shilleh. Institute: University of Colorado Boulder" + time.ctime(time.time())
    xdim = np.shape(x_grid)[0]
    ydim = np.shape(y_grid)[0]
    ldim = np.shape(lat)[0]
    cdim = np.shape(cnt_geo)[1]
    cedim = np.shape(ce)[0]
    root_grp.createDimension('swh', xdim)
    root_grp.createDimension(DIM, ydim)
    root_grp.createDimension('geo', ldim)
    root_grp.createDimension('cyc', cdim)
    root_grp.createDimension('scalar', 1)
    root_grp.createDimension('coeff', cedim)
    xc = root_grp.createVariable('cycles', 'f8', ('cyc',))
    x0 = root_grp.createVariable('swh', 'f8', ('swh',))
    x1 = root_grp.createVariable(DIM, 'f8', (DIM,))
    x2 = root_grp.createVariable('lat', 'f8', ('geo',))
    x3 = root_grp.createVariable('lon', 'f8', ('geo',))
    x4 = root_grp.createVariable('spatiotemporal_ssh_var', 'f8', ('geo','cyc'))
    x5 = root_grp.createVariable('formal_error_spatiotemporal_ssh_var', 'f8', ('geo','cyc'))
    x6 = root_grp.createVariable('count_spatiotemporal_ssh_var', 'f8', ('geo','cyc'))
    x7 = root_grp.createVariable('model_bias','f8',('scalar'))
    x8 = root_grp.createVariable('zerosig','f8',('cyc'))
    if Nz != 0:
        zdim = np.shape(z_grid)[0]
        root_grp.createDimension(OMIT, zdim)
        z0 = root_grp.createVariable(OMIT, 'f8', (OMIT,))
        z0[:] = z_grid
        x9 = root_grp.createVariable('ssb_raw', 'f8', ('swh',DIM,OMIT))
        x10 = root_grp.createVariable('formal_error', 'f8', ('swh',DIM,OMIT))
        x11 = root_grp.createVariable('count_ssb', 'f8', ('swh',DIM,OMIT))
        x12 = root_grp.createVariable('ssb', 'f8', ('swh',DIM,OMIT))
        x13 = root_grp.createVariable('ce_smooth', 'f8', ('coeff'))
    elif Nz == 0:
        x9 = root_grp.createVariable('ssb_raw', 'f8', ('swh',DIM))
        x10 = root_grp.createVariable('formal_error', 'f8', ('swh',DIM))
        x11 = root_grp.createVariable('count_ssb', 'f8', ('swh',DIM))
        x12 = root_grp.createVariable('ssb', 'f8', ('swh',DIM))
        x13 = root_grp.createVariable('ce_smooth', 'f8', ('coeff'))
    x17 = root_grp.createVariable('zs_TECu_alt','f8',('cyc'))
    x18 = root_grp.createVariable('zs_TECu_gim','f8',('cyc'))
    x20 = root_grp.createVariable('iono_calib_bias','f8',('scalar'))
    xc[:] = CYCLES
    x0[:] = x_grid
    x1[:] = y_grid
    x2[:] = lat
    x3[:] = lon
    x4[:] = B_geo
    x5[:] = FE_geo
    x6[:] = cnt_geo
    x7[:] = zs
    x8[:] = zs_cyc
    x9[:] = B_ssb
    x10[:] = FE_ssb
    x11[:] = cnt_ssb
    x12[:] = B_smooth
    x13[:] = ce
    x17[:] = zs_TECu_alt
    x18[:] = zs_TECu_gim
    x20[:] = rs_zs_bias
    x0.units = "m"
    x0.comment = 'Significant wave height (swh): row dimensions of sea state bias model.'
    if DIM == 'sig0':
        x1.units = "dB"
        x1.comment = 'Backscatter coefficient (sig0): column dimensions of sea state bias model.'
    elif DIM == 'u10':
        x1.units = "m/s"
        x1.comment = 'Wind speed (u10): column dimensions of sea state bias model.'
    x2.units = "degrees_north"
    x2.comment = "Latitude of 3-degree equal-area grid corresponding to spatial_ssh_var"
    x3.units = "degrees_east"
    x3.comment = "Longitude of 3-degree equal-area grid corresponding to spatial_ssh_var"
    x4.units = "m"
    x4.comment = "Mean spatiotemporal variations of sea surface height (ssh) binned by 3-degree equal-area grid. (coe assumption: E[SSHA'] = E[ssb] + E[spatiotemporal_ssh_var])"
    x5.units = "m"
    x5.comment = "Unscaled (unitless) formal error of mean spatiotemporal variations of sea surface height (spatiotemporal_ssh_var)."
    x6.comment = "Number of observations used to estimate each bin within spatiotemporal_ssh_var"
    x7.units = "m"
    x7.comment = "Bias removed from the sea state bias model (ssb). cutoff("+str(cutoff)+"%), degree("+str(deg)+"). Corresponds to the mean of the zero-significance values (zerosig)."
    x8.units = "m"
    x8.comment = "Zero-significance values for all cycles used to level the sea state bias model. The mean of this value is equal to the bias removed from ssb (zerosig_mean). cutoff("+str(cutoff)+"%), degree("+str(deg)+")"
    x9.units = "m"
    x9.comment = "Raw, coestimated sea state bias model. Raw = not leveled and not smoothed. (assumption: E[SSHA'] = E[ssb] + E[spatiotemporal_ssh_var] )"
    x10.units = "m"
    x10.comment = "Unscaled (unitless) formal error of sea state bias model. Scale by multiplying unscaled formal error by RMS of the residuals (~0.077 m)."
    x11.comment = "Number of observations used to estimate each bin within the sea state bias models"
    x12.units = "m"
    x12.comment = "Smoothed and leveled coestimated sea state bias model. PREFERRED MODEL. (assumption: E[SSHA'] = E[ssb] + E[spatiotemporal_ssh_var] )"
    if Nz != 0:
        x13.units = 'cubic fit to each parameter of the model (i.e.swh,'+DIM+','+OMIT+')'
    elif Nz == 0:
        x13.units = 'cubic fit to each parameter of the model (i.e.swh,'+DIM+')'
    x13.comment = "Fitting coefficients for parametric, apriori sea state bias used to extrapolate and smooth ssb. ssb_apriori = (ce1*swh)+(ce2*swh^2)+(ce3*swh^3)+(ce4*u10)+(ce5*u10^2)+(ce6*u10^3)"
    x17.units = "TECu"
    x17.comments = "dual-frequency TEC corresponding to cutoff=[1,10]% zero-significance using a linear fit"
    x18.units = "TECu"
    x18.comments = "GIM TEC corresponding to cutoff=[1,10]% zero-significance using a linear fit"
    x20.units = "TECu"
    x20.comments = "Ionosphere calibration bias from dTEC (GIM-DF) corresponding to cutoff=[1,10]% zero-significance using a linear fit"
    root_grp.close()


# --------------------------------------------------------------------------#
#-------------------find observed bins - full model-------------------------#
def Hmat_sum(H_sparse,idx_bin,idx_obs):
    '''
    Purpose: Produces the number of observations per node (sH_bin) and the number of nodes assigned to each observation (sH_obs) 
    H_sparse: design matrix for a single cycle
    idx_bin = index of valid nodes
    idx_obs = index of valid observations
    '''
    H1=H_sparse.copy()
    H1[H1!=0]=1
    H1b=H1[:,idx_bin]
    H1bo=H1b[idx_obs,:]
    sH_bin= np.squeeze(np.asarray(H1bo.sum(axis=0)))
    sH_obs= np.squeeze(np.asarray(H1bo.sum(axis=1)))
    return sH_bin,sH_obs


def obs2bin_bin2obs(H_sparse,min_obs):
    '''
    Purpose: Iterative run function for Hmat_sum to provide an apriori estimate for which observations and nodes will be used to model SSB. This is required to make sure that there are suffiecient observations per node and nodes assigned to observations to solve for the SSB grid using least squares.
    H_sparse: original design matrix for a single cycle
    min_obs = integer defining the minimum number of observations per node
    idx_bin = index of valid nodes
    idx_obs = index of valid observations
    H_new: new (filtered) design matrix for a single cycle
    '''
    print('original H shape: '+str(H_sparse.shape))
    idx_bin=np.arange(np.shape(H_sparse)[1])
    idx_obs=np.arange(np.shape(H_sparse)[0])
    idx_bin2=[]
    idx_obs2=[]
    itera=1
    while np.size(idx_bin)!=np.size(idx_bin2) and np.size(idx_obs)!=np.size(idx_obs2):
        print('iteration number '+str(itera))
        sHb1,sHo1=Hmat_sum(H_sparse,idx_bin,idx_obs)
        idx_bin2=idx_bin[np.where(sHb1>=min_obs)[0]]
        
        sHb2,sHo2=Hmat_sum(H_sparse,idx_bin2,idx_obs)
        idx_obs2=idx_obs[np.where((sHo1-sHo2)==0)[0]]
        
        sHb3,sHo3=Hmat_sum(H_sparse,idx_bin2,idx_obs2)
        idx_bin=idx_bin2[np.where(sHb3>=min_obs)[0]]
        idx_obs=idx_obs2[np.where((sHo1[idx_obs2]-sHo3)==0)[0]]
        itera+=1
    Hb=H_sparse[:,idx_bin]
    H_new=Hb[idx_obs,:]
    print('filtered H shape: '+str(H_new.shape))
    return idx_bin,idx_obs,H_new

def bin_count_cycles(MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,BAND,DIMND,min_obs_cnt,P_ssb,P_geo,ERR,SSBk,SSBc,Irel,OPEN_OCEAN):
    '''
    Purpose: uses obs2bin_bin2obs to provide an apriori estimate for which observations and nodes will be used to model SSB given all the cycles (CYCLES) used to create the model.
    min_obs_cnt = integer defining the minimum number of observations per nodes
    P_ssb = number of nodes in the full SSB model/grid
    P_geo = number of nodes in the full spatio-temporal SSHA error model/grid (COE only)
    idx_ssb,idx_geo = index of valid SSB nodes and spatio-temoral SSHA error nodes, respectively
    col_cnt_sum_ssb,col_cnt_sum_geo = total number of observations per node for all CYCLES for the SSB model and spatial-temporal SSHA error model, respectively.
    col_cnt_geo,col_cnt_ssb = number of observations per node for each cycle for the SSB model and spatial-temporal SSHA error model, respectively.
    '''
    srt = time.time()
    if 'usla' in VAR:
        BAND1 = 'ku'
    else: 
        BAND1 = BAND
    if 'met3' in METHOD:
        MET = METHOD[:-5]
    else:
        MET = METHOD
    N = np.shape(CYCLES)[0]
    col_cnt_ssb = np.zeros((N,P_ssb))
    col_cnt_geo = np.zeros((N,P_geo))
    for ii in np.arange(N):
        CYC1 = str(CYCLES[ii])
        CYC2 = CYC1
        FILE1,FILE2,FILE2_GEO = ldic.filenames_specific(MISSIONNAME,MET,SOURCE,CYC1,CYC2,'ku',DIMND,ERR=ERR)
        fn_exist = path.exists(FILE1)
        fn_exist_geo = path.exists(FILE2_GEO)
        if fn_exist is True:
            print('bin_count_cycles: FILE1 = '+FILE1)
            ds = Dataset(FILE1).variables
            B_obs,inonan,inan = lflt.observables(METHOD,VAR,BAND1,ds,SSBk,SSBc,Irel,ERR,MISSIONNAME,OPEN_OCEAN)
            H_ssb_sparsep_npz = np.load(FILE2,allow_pickle=True)
            H_ssb_sparsep = H_ssb_sparsep_npz['arr_0'].tolist()[inonan,:]
            idx_bin,idx_obs,H_new = obs2bin_bin2obs(H_ssb_sparsep,min_obs_cnt/np.shape(CYCLES)[0])
            col_cnt_ssb[ii,:] = lgrid.obsmat_count(H_ssb_sparsep[idx_obs,:])
            if fn_exist_geo is True:
                H_geo_sparsep_npz = np.load(FILE2_GEO,allow_pickle=True)
                H_geo_sparsep = H_geo_sparsep_npz['arr_0'].tolist()[inonan[idx_obs],:]
                col_cnt_geo[ii,:] = lgrid.obsmat_count(H_geo_sparsep[:])
        else:
            print('bin_count_cycles: '+FILE1+' does not exist')
            print('obs for cycle '+CYC1+': '+str(np.sum(col_cnt_ssb[ii,:])))
    col_cnt_sum_ssb = np.nansum(col_cnt_ssb,axis=0)
    print('min number of obs: '+str(min_obs_cnt))
    idx_ssb = np.where(col_cnt_sum_ssb[:]>=min_obs_cnt)[0]
    if fn_exist_geo is True:
        col_cnt_sum_geo = np.nansum(col_cnt_geo,axis=0)
        idx_geo = np.where(col_cnt_sum_geo[:]>=4)[0]
    else:
        print('COE files do not exist')
        col_cnt_sum_geo = []
        idx_geo = []
    print('Time to find counts per bin for all cycles [sec]: '+str(time.time()-srt))
    return idx_ssb,idx_geo,col_cnt_sum_ssb,col_cnt_sum_geo,col_cnt_geo,col_cnt_ssb
# --------------------------------------------------------------------------#
#-------------------obs2lut to lut2ssb--------------------------------------#
def interpolation_method(MISSIONNAME,METHOD,VAR,SOURCE,BAND,DIMND,cutoff,deg,CYCLES,DIM,OMIT,min_obs_cnt,WEIGHT,ERR,SSBk,SSBc,Irel,OPEN_OCEAN):
    '''
    Purpose: Carry-out the interpolation method using many subfunctions to produce a library (LIB) that contains the model coordinates, raw SSB model, smoothed and leveled SSB model, formal error and number of observations per node.
    min_obs_cnt = integer defining the minimum number of observations per nodes
    cutoff = percentile segment for zero-significance polynomial fit (see dissertation) 
    deg = degree of zero-significance polynomial fit
    '''
    ########## Call grid ##########
    lon_grid,lat_grid,lon_diff,lat_diff,area = lgrid.reduced_gauss_grid()
    x_grid,y_grid,z_grid = lgrid.grid_dimensions(DIM,OMIT)
    mx,my,mz,fx,fy,fz = lgrid.parameter_grid(x_grid,y_grid,z_grid)
    ########## Determine sizes ##########
    P_ssb = np.shape(fx)[0]
    P_geo = np.shape(lon_grid)[0]
    mN = np.shape(mx)
    ########## Declare model ##########
    if METHOD=='coe': # simultaneously solve for spatial variations (COE, coestimated model)
        ########## Observations per bin over a span of multiple cycles (find which bins are observed) ##########
        idx_ssb,idx_geo,col_cnt_sum_ssb,col_cnt_sum_geo,col_cnt_geo,col_cnt_ssb = bin_count_cycles(
                MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,BAND,DIMND,min_obs_cnt,P_ssb,P_geo,ERR,SSBk,SSBc,Irel,OPEN_OCEAN) 
        ########## unleveled SSB model - solved using Cholesky ##########
        Z1,FE_s,idx_geo,mse,FN_temp_sv,HtH_ones,idx_ssb,col_ssb_update,col_geo_update = lest.cdssb_Z1(min_obs_cnt,MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,BAND,DIMND,OMIT,idx_ssb,col_cnt_ssb,col_cnt_geo,ERR,SSBk,SSBc,Irel,P_ssb,mN,OPEN_OCEAN)
        B_s = lgrid.data2grid(Z1,P_ssb,mN,idx_ssb)
        FE_s = lgrid.data2grid(FE_s,P_ssb,mN,idx_ssb)
        B_geo,FE_geo = lest.cdssb_Z2(min_obs_cnt,MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,BAND,DIMND,idx_ssb,col_cnt_geo,Z1,ERR,SSBk,SSBc,Irel,FN_temp_sv,OPEN_OCEAN)
        ########## Count observations per bin ##########
        cnt_geo = np.copy(col_geo_update)#np.copy(col_cnt_geo)   !!!
        ##
        col_cnt_sum_ssb0=np.zeros(np.shape(np.nansum(col_ssb_update,axis=0)))
        col_cnt_sum_ssb0[idx_ssb]=np.diag(HtH_ones)
        cnt_s = np.reshape(col_cnt_sum_ssb0,mN,order='F')
        ########## Leveing correction ##########
        zs_s,zs_mean_s,model_bias_s,model_bias_arr_s = lpost.run_model_level(MISSIONNAME,'dir',VAR,SOURCE,'ku',CYCLES,DIM,OMIT,B_s,FE_s,cutoff,deg,BAND,ERR,SSBk,SSBc,Irel,WEIGHT,OPEN_OCEAN)
        ########## Smoothing and extrapolation ##########
        B_se_s,ce_s = lpost.putnam_smoothing(np.subtract(B_s,zs_mean_s),FE_s,mx,my,mz) 
        B_cs,FE_cs,zs_cs,zs_mean_cs,B_se_cs,ce_cs = [],[],[],[],[],[]
        Bc_s,FEc_s=[],[]
        zsc_s=[]
        zsc_mean_s=[]
        Bc_se_s=[]
        cec_s=[]
        model_bias_c_s=[]
        model_bias_arr_c_s=[]
        ########## Residuals ##########
        mseC = np.empty(np.size(mse))*np.nan
    else: # no coestimation, this includes: JNT, COl, XOV and DIR models
        ########## Observations per bin over a span of multiple cycles (find which bins are observed) ##########
        B_geo,FE_geo,cnt_geo=[],[],[]
        if METHOD == 'jnt':
            idx_ssbc,idx_geoc,col_cnt_sum_ssbc,col_cnt_sum_geoc,col_cnt_geoc,col_cnt_ssbc = bin_count_cycles(
                    MISSIONNAME,'col',VAR,SOURCE,CYCLES[:-1],BAND,DIMND,min_obs_cnt,P_ssb,P_geo,ERR,SSBk,SSBc,Irel,OPEN_OCEAN)  
            idx_ssbx,idx_geox,col_cnt_sum_ssbx,col_cnt_sum_geox,col_cnt_geox,col_cnt_ssbx = bin_count_cycles(
                    MISSIONNAME,'xov',VAR,SOURCE,CYCLES,BAND,DIMND,min_obs_cnt,P_ssb,P_geo,ERR,SSBk,SSBc,Irel,OPEN_OCEAN)  
            col_cnt_sum_ssb = (col_cnt_sum_ssbc+col_cnt_sum_ssbx)
            idx_ssb = np.where(col_cnt_sum_ssb[:]>=min_obs_cnt)[0]
            col_cnt_sum_geo = []
            idx_geo = []
        else:
            idx_ssb,idx_geo,col_cnt_sum_ssb,col_cnt_sum_geo,col_cnt_geo,col_cnt_ssb = bin_count_cycles(
                    MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,BAND,DIMND,min_obs_cnt,P_ssb,P_geo,ERR,SSBk,SSBc,Irel,OPEN_OCEAN)  
            col_cnt_sum_geo = []
            idx_geo = []
        # IS THIS PART OF ELSE?
        ########## Per-cycle summation of the information and measurement matrices ##########
        HtH_lut_cs,HtB_lut_cs,HtH_lut_s,HtB_lut_s,HtH_ones,Noutliers,mse,FN_temp_sv = lest.matrix_summation_lse(MISSIONNAME,METHOD,VAR,OMIT,SOURCE,CYCLES,BAND,DIMND,idx_ssb,idx_geo,WEIGHT,ERR,SSBk,SSBc,Irel,[],OPEN_OCEAN)
        ichk_s = np.where((np.diag(HtH_ones)<4))[0]
        if np.size(ichk_s)>0:
                print('not positive definite '+str(np.size(ichk_s))+' zero elements--> find new indices')
                idx_ssb = np.delete(idx_ssb,ichk_s)
                HtH_lut_cs,HtB_lut_cs,HtH_lut_s,HtB_lut_s,HtH_ones,Noutliers,mse,FN_temp_sv = lest.matrix_summation_lse(MISSIONNAME,METHOD,VAR,OMIT,SOURCE,CYCLES,BAND,DIMND,idx_ssb,idx_geo,WEIGHT,ERR,SSBk,SSBc,Irel,[],OPEN_OCEAN)
        ########## unleveled SSB model - solved using Cholesky #########
        B_s,FE_s,dmp1,dmp2,Bc_s,FEc_s,dmp1,dmp2 = lest.lse2ssb(
                HtH_lut_s,HtB_lut_s,P_ssb,mN,idx_ssb,fN_geo=[],mN_geo=[],idx_geo_half=[])
        ########## post-fit residual outlier detector #########
        if VAR in ['usla','ssha_dfI','rng_mle','iono_corr_alt', 'mss_dtu','mss_cnes15', 'ocean_got', 'wet_tropo_model','sum_err','ib_corr','alt_jpl','alt_cnes','iono_tran','iono_tran_smooth','SO']:
            Noutliers = 1
            if VAR in ['rng_mle']:
                itermax=7
            else:
                itermax=6#5!!!
            if ERR in ['rms_r','sum_err']:#r','rms_s','rms:
                Noutliers=0
        else:
            Noutliers = 0
        iteri = 1
        while Noutliers!=0:
            if iteri>itermax:
                raise('TOO MANY ITERATIONS')
            else:
                ########## Per-cycle summation of the information and measurement matrices ##########
                print('minimum count after removal is :'+str(np.min(col_cnt_sum_ssb[idx_ssb])))
                HtH_lut_cs,HtB_lut_cs,HtH_lut_s,HtB_lut_s,HtH_ones,Noutliers,mse,FN_temp_sv = lest.matrix_summation_lse(MISSIONNAME,METHOD,VAR,OMIT,SOURCE,CYCLES,BAND,DIMND,idx_ssb,idx_geo,WEIGHT,ERR,SSBk,SSBc,Irel,B_s,OPEN_OCEAN,iteration=iteri)
                ichk_s = np.where((np.diag(HtH_ones)<4))[0]#np.where((np.diag(HtH_lut_s)==0)&(np.diag(HtH_ones)<4))[0]
                print('np.size(ichk_s) '+str(np.size(ichk_s))+' zero elements--> find new indices')
                if np.size(ichk_s)>0:
                    print('not positive definite '+str(np.size(ichk_s))+' zero elements--> find new indices')
                    idx_ssb = np.delete(idx_ssb,ichk_s)
                    HtH_lut_cs,HtB_lut_cs,HtH_lut_s,HtB_lut_s,HtH_ones,Noutliers,mse,FN_temp_sv = lest.matrix_summation_lse(MISSIONNAME,METHOD,VAR,OMIT,SOURCE,CYCLES,BAND,DIMND,idx_ssb,idx_geo,WEIGHT,ERR,SSBk,SSBc,Irel,B_s,OPEN_OCEAN,iteration=iteri)
                    ichk_s = np.where((np.diag(HtH_ones)<4))[0]#
                    ########## unleveled SSB model - solved using Cholesky #########
                if Noutliers!=0:
                    B_s,FE_s,dmp1,dmp2,Bc_s,FEc_s,dmp1,dmp2 = lest.lse2ssb(HtH_lut_s,HtB_lut_s,P_ssb,mN,idx_ssb,fN_geo=[],mN_geo=[],idx_geo_half=[])
                print('###################### Noutlier '+str(Noutliers)+' for iteration '+str(iteri)+'#######################################')
                iteri += 1
        ########## count #########
        col_cnt_sum_ssb0=np.zeros(np.shape(col_cnt_sum_ssb))
        col_cnt_sum_ssb0[idx_ssb]=np.diag(HtH_ones)
        cnt_s = np.reshape(col_cnt_sum_ssb0,mN,order='F')
        ########## Residuals ##########
        mseC = np.empty(np.size(mse))*np.nan
        if 'met3' in METHOD:
            mseC = np.empty(np.shape(CYCLES))*np.nan 
        ########## Nominal Model: level and smooth ##########
        print('nonan bins: '+str(np.shape(np.where(~np.isnan(B_s.flatten('F')))[0])[0]))
        print('nan bins: '+str(np.shape(np.where(np.isnan(B_s.flatten('F')))[0])[0]))
        zs_s,zs_mean_s,model_bias_s,model_bias_arr_s = lpost.run_model_level(MISSIONNAME,'dir',VAR,SOURCE,'ku',CYCLES,DIM,OMIT,B_s,FE_s,cutoff,deg,BAND,ERR,SSBk,SSBc,Irel,WEIGHT,OPEN_OCEAN)
        print('zero-significance model leveling value: '+str(zs_mean_s))
        if np.isnan(zs_mean_s) is True:
            raise('zs must be a number, not nan')
        B_se_s,ce_s = lpost.putnam_smoothing(np.subtract(B_s,zs_mean_s),FE_s,mx,my,mz)
        ########## static COE Model: level and smooth ##########
        if 'met3' in METHOD:
            ########## Nominal Model: level and smooth ##########
            zsc_s,zsc_mean_s,model_bias_c_s,model_bias_arr_c_s = lpost.run_model_level(MISSIONNAME,'dir',VAR,SOURCE,'ku',CYCLES,DIM,OMIT,Bc_s,FEc_s,cutoff,deg,BAND,ERR,SSBk,SSBc,Irel,WEIGHT,OPEN_OCEAN)
            Bc_se_s,cec_s = lpost.putnam_smoothing(np.subtract(Bc_s,zsc_mean_s),FEc_s,mx,my,mz)
        else:
            zsc_s,zsc_mean_s = [],[]
            Bc_se_s,cec_s = [],[]
            zsc_cs,zsc_mean_cs = [],[]
            Bc_se_cs,cec_cs = [],[]
            model_bias_c_s,model_bias_c_cs=[],[]
            model_bias_arr_c_s,model_bias_arr_c_cs=[],[]
    LIB = {}
    LIB['B_s']=B_s
    LIB['FE_s']=FE_s
    LIB['cnt_s'] = cnt_s
    LIB['zs_s'] = zs_s
    LIB['zs_mean_s'] = zs_mean_s
    LIB['B_se_s'] = B_se_s
    LIB['ce_s'] = ce_s
    LIB['mse'] = mse    
    LIB['model_bias_s']=model_bias_s
    LIB['model_bias_array_s']=model_bias_arr_s
    LIB['Bc_s']=Bc_s
    LIB['FEc_s']=FEc_s
    LIB['zsc_s'] = zsc_s
    LIB['zsc_mean_s'] = zsc_mean_s
    LIB['Bc_se_s'] = Bc_se_s
    LIB['cec_s'] = cec_s
    LIB['mseC'] = mseC
    LIB['model_bias_c_s']=model_bias_c_s
    LIB['model_bias_array_c_s']=model_bias_arr_c_s
    LIB['B_geo'] = B_geo
    LIB['FE_geo'] = FE_geo
    LIB['cnt_geo'] = cnt_geo
    return LIB
    

def interpolation_model(min_obs_cnt,MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,DIM,OMIT,deg,cutoff,WEIGHT,ERR,setSSB,OPEN_OCEAN):
    ''' 
    Purpose: run function for interpolation_method and ionosphere_zerosig1 to create both Ku-band and C-band SSB models, as well as estimate the ionosphere calibration bias, respectively. All outputs saved using crt_matrices2models or crt_matrices2models_coe.
    min_obs_cnt = integer defining the minimum number of observations per nodes
    cutoff = percentile segment for zero-significance polynomial fit (see dissertation) 
    deg = degree of zero-significance polynomial fit
    x_grid,y_grid,z_grid = x-, y- and z-axis grid nodes (z_grid empty if 2D)
    '''
    srt = time.time()
    PATHMISSIONS = ldic.save_path()
    DIMND,OMITlab,MODELFILE = ldic.file_labels(VAR,'ku',DIM,OMIT)
    if VAR in ['ssha_dfI','iono_corr_alt','ISSB_tdm','dISSB_tdm','dRIT_tdm']:
        ########## Save results ##########
        if MISSIONNAME=='tx':
            if CYCLES[0]<235:
                cm1,cm2 ='48','100'
            else:
                cm1,cm2 ='280','364'#str(CYCLES[0]),str(CYCLES[-1])#'280','364'#str(CYCLES[0]),str(CYCLES[-1])
        else:
            cm1,cm2=str(CYCLES[0]),str(CYCLES[-1])
        FNk = ldic.create_filename(MISSIONNAME,'jnt','usla',MODELFILE,SOURCE,cm1,cm2,'ku',DIMND,'nc',ERR=ERR)
        FNc = ldic.create_filename(MISSIONNAME,'jnt','usla',MODELFILE,SOURCE,cm1,cm2,'c',DIMND,'nc',ERR=ERR)
        if WEIGHT==True:
            FNk = FNk[:-3]+'_LatWgt'+FNk[-3:]
            FNc = FNc[:-3]+'_LatWgt'+FNc[-3:]
        if VAR=='iono_alt_diff_dir' or VAR=='ssb_alt_dir':
            MODMET = 'dir' #'jnt'
        else:
            MODMET = 'jnt'
            if MISSIONNAME == 'j1':
                FNk = PATHMISSIONS+'j1/'+MODMET+'/matrix2model/j1_'+MODMET+'_ssb_matrix2model_jpl_1_c_37_ku_u10_2d_LatWgt.nc'
                FNc = PATHMISSIONS+'j1/'+MODMET+'/matrix2model/j1_'+MODMET+'_ssb_matrix2model_jpl_1_c_37_c_u10_2d_LatWgt.nc'
            else:
                raise('Define models to pull from')
        print('alt iono using :'+FNk)
        dsk = Dataset(FNk).variables
        dsc = Dataset(FNc).variables
        SSBk = dsk['ssb'][:]
        SSBc = dsc['ssb'][:]
        Irel = dsk['iono_calib_bias'][:]*(-0.01216)
    else:
        SSBk=[]
        SSBc=[]
        Irel=[]

    ########## Call grid ##########
    lon_grid,lat_grid,lon_diff,lat_diff,area = lgrid.reduced_gauss_grid()
    x_grid,y_grid,z_grid = lgrid.grid_dimensions(DIM,OMIT)
    mx,my,mz,fx,fy,fz = lgrid.parameter_grid(x_grid,y_grid,z_grid)
    # --------------------------------------------------------------------------#
    ########## Determine Ku-band USLA Observation ##########
    # --------------------------------------------------------------------------#
    srtk = time.time()
    ########## Observations per bin over a span of multiple cycles (find which bins are observed) ##########
    LIB = interpolation_method(MISSIONNAME,METHOD,VAR,SOURCE,'ku',DIMND,cutoff,deg,CYCLES,DIM,OMIT,min_obs_cnt,WEIGHT,ERR,SSBk,SSBc,Irel,OPEN_OCEAN)
    ########## Save results ##########
    if setSSB is True:
        LIB['B_se_s'] = LIB['B_se_s']-LIB['B_se_s'][7,22]+(-0.088)
    B_s=LIB['B_s']
    FE_s=LIB['FE_s']
    cnt_s=LIB['cnt_s'] 
    zs_s=LIB['zs_s'] 
    zs_mean_s=LIB['zs_mean_s'] 
    B_se_s=LIB['B_se_s'] 
    ce_s=LIB['ce_s'] 
    mse = LIB['mse']
    mb_s = LIB['model_bias_s']
    mb_arr_s = LIB['model_bias_array_s']
    B_geo=LIB['B_geo']
    FE_geo=LIB['FE_geo']
    cnt_geo=LIB['cnt_geo'] 
    print('Time to estimate Ku-band SSB: '+str(time.time()-srtk))
    # --------------------------------------------------------------------------#
    ########## Determine C-band USLA Observation ##########
    # --------------------------------------------------------------------------#
    if 'met3' in METHOD:
        print('C-band SSB coestimated with Ku-band')
        B_sC=LIB['Bc_s']
        FE_sC=LIB['FEc_s']
        zs_sC=LIB['zsc_s'] 
        zs_mean_sC=LIB['zsc_mean_s'] 
        B_se_sC=LIB['Bc_se_s'] 
        ce_sC=LIB['cec_s'] 
        cnt_sC=LIB['cnt_s'] 
        mseC = LIB['mseC'] 
        mb_c_s = LIB['model_bias_c_s']
        mb_arr_c_s = LIB['model_bias_array_c_s']
    else:
        if np.size(OMIT)==0:
            print('C-band SSB found independently')
            srtc = time.time()
            # Repeat SSB modeling steps for c-band
            LIBc = interpolation_method(MISSIONNAME,METHOD,VAR,SOURCE,'c',DIMND,cutoff,deg,CYCLES,DIM,OMIT,min_obs_cnt,WEIGHT,ERR,SSBk,SSBc,Irel,OPEN_OCEAN)
            if setSSB is True:
                LIBc['B_se_s'] = LIBc['B_se_s']-LIBc['B_se_s'][7,22]+(-.077)
            B_sC=LIBc['B_s']
            FE_sC=LIBc['FE_s']
            cnt_sC=LIBc['cnt_s']
            zs_sC=LIBc['zs_s']
            zs_mean_sC=LIBc['zs_mean_s']
            B_se_sC=LIBc['B_se_s']
            ce_sC=LIBc['ce_s']
            mseC = LIBc['mse']
            mb_c_s = LIBc['model_bias_s']
            mb_arr_c_s = LIBc['model_bias_array_s']
            B_geoC=LIBc['B_geo']
            FE_geoC=LIBc['FE_geo']
            cnt_geoC=LIBc['cnt_geo']
            print('Time to estimate C-band SSB: '+str(time.time()-srtc))
    # --------------------------------------------------------------------------#
    ########## Range+SSB bias ##########
    # --------------------------------------------------------------------------#
    if ERR not in ['txj1','j1j2','j2j3','j3s6A'] and np.size(OMIT)==0:
        if MISSIONNAME!='s6Ah':
            md_TECu_alt,md_TECu_gim,zs_TECu_alt,zs_TECu_gim,rs_md_bias,rs_zs_bias=lion.ionosphere_zerosig1(MISSIONNAME,VAR,SOURCE,CYCLES,'ku',DIM,OMIT,B_se_s,B_se_sC,ERR=ERR)
            scl = -0.01216
            Irel_zs = np.round(np.round(rs_zs_bias,5)*scl,5)
            Irel_md = np.round(np.round(rs_md_bias,5)*scl,5)
            print('Relative ionosphere calibration value applied '+str(Irel_zs)+'m, from dTECu='+str(np.round(rs_zs_bias,5)))
        else:
            Irel_zs=np.nan
            Irel_md = np.nan
            rs_zs_bias = np.nan
            rs_md_bias=np.nan
            md_TECu_alt,md_TECu_gim,zs_TECu_alt,zs_TECu_gim=np.zeros(np.shape(CYCLES)),np.zeros(np.shape(CYCLES)),np.zeros(np.shape(CYCLES)),np.zeros(np.shape(CYCLES))
    else:
        Irel_zs=np.nan
        Irel_md = np.nan
        rs_zs_bias = np.nan
        rs_md_bias=np.nan
        md_TECu_alt,md_TECu_gim,zs_TECu_alt,zs_TECu_gim=np.zeros(np.shape(CYCLES)),np.zeros(np.shape(CYCLES)),np.zeros(np.shape(CYCLES)),np.zeros(np.shape(CYCLES))
    ########## Save results ##########
    if 'col' in METHOD:
        FILE3i = ldic.create_filename(MISSIONNAME,METHOD,VAR,MODELFILE,SOURCE,str(CYCLES[0]),str(CYCLES[-1]+1),'ku',DIMND,'nc',ERR=ERR)
        FILE3ic = ldic.create_filename(MISSIONNAME,METHOD,VAR,MODELFILE,SOURCE,str(CYCLES[0]),str(CYCLES[-1]+1),'c',DIMND,'nc',ERR=ERR)
    else:
        FILE3i = ldic.create_filename(MISSIONNAME,METHOD,VAR,MODELFILE,SOURCE,str(CYCLES[0]),str(CYCLES[-1]),'ku',DIMND,'nc',ERR=ERR)
        FILE3ic = ldic.create_filename(MISSIONNAME,METHOD,VAR,MODELFILE,SOURCE,str(CYCLES[0]),str(CYCLES[-1]),'c',DIMND,'nc',ERR=ERR)
    FILE3 = FILE3i
    FILE3c = FILE3ic
    if WEIGHT is False:
        FILE3 = FILE3i[:-3]+'_noLatWgt'+FILE3i[-3:]
        FILE3c = FILE3ic[:-3]+'_noLatWgt'+FILE3ic[-3:]
    else:
        FILE3 = FILE3i[:-3]+'_LatWgt'+FILE3i[-3:]
        FILE3c = FILE3ic[:-3]+'_LatWgt'+FILE3ic[-3:]
    if setSSB is True:
        FILE3 = FILE3[:-3]+'_setSSB'+FILE3[-3:]
        FILE3c = FILE3c[:-3]+'_setSSB'+FILE3c[-3:]
    if OPEN_OCEAN==False:
        FILE3 = FILE3[:-3]+'_coast'+FILE3[-3:]
        FILE3c = FILE3c[:-3]+'_coast'+FILE3c[-3:]
    if METHOD=='coe':
        crt_matrices2models_coe(x_grid,y_grid,z_grid,CYCLES,B_s,FE_s,cnt_s,lat_grid,lon_grid,B_geo,FE_geo,cnt_geo.T,
                        FILE3,DIM,OMIT,zs_mean_s,zs_s,B_se_s,ce_s,
                        zs_TECu_alt,zs_TECu_gim,rs_zs_bias,cutoff,deg)
        crt_matrices2models_coe(x_grid,y_grid,z_grid,CYCLES,B_sC,FE_sC,cnt_sC,lat_grid,lon_grid,B_geoC,FE_geoC,cnt_geoC.T,
                        FILE3c,DIM,OMIT,zs_mean_sC,zs_sC,B_se_sC,ce_sC,
                        zs_TECu_alt,zs_TECu_gim,rs_zs_bias,cutoff,deg)
    else:
        crt_matrices2models(METHOD,x_grid,y_grid,z_grid,CYCLES,B_s,cnt_s,FE_s,FILE3,DIM,OMIT,zs_mean_s,zs_s,B_se_s,ce_s,mse,
                               zs_TECu_alt,zs_TECu_gim,rs_zs_bias,cutoff,deg)
        crt_matrices2models(METHOD,x_grid,y_grid,z_grid,CYCLES,B_sC,cnt_sC,FE_sC,FILE3c,DIM,OMIT,zs_mean_sC,zs_sC,B_se_sC,ce_sC,mseC,
                               zs_TECu_alt,zs_TECu_gim,rs_zs_bias,cutoff,deg)
    print('Time to create '+FILE3+' [sec]: '+str(time.time()-srt))
    return x_grid,y_grid,z_grid
# --------------------------------------------------------------------------#



# --------------------------------------------------------------------------#
#-------------------run: obs2lut to lut2ssb---------------------------------#

def hardcoded_elements():
    # Purpose: hardcode certain variables for default
    min_obs_cnt = 50 #initial minimum number of observations/node
    deg = 3 # degree of polynomial fit for zerosignificance leveling
    cutoff = [0.1,2.0] # percentile cutoff for zerosignificance leveling fit
    return min_obs_cnt,deg,cutoff

def run_matrices2models(MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,OMIT,DIM,WEIGHT,ERR=[],setSSB=False,OPEN_OCEAN=True):
    strt = time.time()
    min_obs_cnt,deg,cutoff = hardcoded_elements()
    if SOURCE=='product':
        if 'col' in METHOD:
            CYCLES = CYCLES[:-1]
    x_grid,y_grid,z_grid = interpolation_model(min_obs_cnt,MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,DIM,OMIT,deg,cutoff,WEIGHT,ERR,setSSB,OPEN_OCEAN)
    print('#------Time to run matrices2models: '+str(time.time()-strt)+'------#')
    return x_grid,y_grid,z_grid






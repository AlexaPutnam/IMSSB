#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:47:38 2020

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


import lib_dict_ext as ldic
import lib_geo_gridding_ext as lgrid
import lib_post_processing_ext as lpost
import lib_trilinear_interpolation_ext as lti
import lib_bilinear_interpolation_ext as lbi
import lib_data_ext as ldata

# --------------------------------------------------------------------------#
#------------------- Ionosphere: range + SSB bias -------------------------#
def iono_level(x,VARLEV,cutoff,deg,XYflip,LINEAR,MP):
    '''
    Purpose: Determine the zero-significance value of the TECu histogram using zerosig_hist and find_zerosig
    x = TECu array (a single cycle)
    VARLEV = leveling variable (‘tec’ for TECu)
    cutoff = percentile segment for polynomial fit (see dissertation)
    deg = degree of polynomial fit (automatically adjusted to 1 if LINEAR = True)
    XYflip = True or False whether the x-axis of the CDF is used as the dependent variable for the polynomial fit (True for TECu)
    LINEAR = True or False to apply a linear fit to the cutoff region (True for TECu)
    MP = True or False to extrapolate to midpoint rather than zero-significance (Default to False)
    zsi = zero-significance value 
    cdf_out = CDF of TECu cutoff region
    poly_fit = polynomisl fit to TECu CDF cutoff region
    bin_out = CDF independent variable bins corresponding to poly_fit
    hist_x = full histogram of TECu
    bin_mid = bins corresponding to hist_x
    varres = mean square error of the fit 
    '''
    hist_x,cdf_x,bin_edges,bin_mid = lpost.zerosig_hist(x,VARLEV) #plt.plot(bin_mid,hist_xiono), #plt.plot(bin_mid,cdf_x)
    zsi,cdf_out,poly_fit,bin_out,varres = lpost.find_zerosig(hist_x,bin_mid,cutoff,deg,XYflip,VARLEV,LINEAR,MP) #plt.plot(bin_mid,cdf_rev,'o',) #plt.plot(bin_mid[idx],cdf_rev[idx],'o',)
    return zsi,cdf_out,poly_fit,bin_out,hist_x,bin_mid,varres


def ionosphere_zerosig1(MISSIONNAME,VAR,SOURCE,CYCLES,BAND,DIM,OMIT,Bku_model,Bc_model,cutoff=[1.,10.],deg=1,AVG=21,XYflip=True,LINEAR=True,ERR=[]): #AVG=np.asarray([0,3,5,7,11,15,21,31,51,71,101])
    '''
    Purpose: Determine the ionosphere calibration bias
    Bku_model = leveled and smoothed Ku-band SSB solution
    Bc_model = leveled and smoothed C-band SSB solution
    cutoff = percentile segment for polynomial fit (see dissertation)
    deg = degree of polynomial fit (automatically adjusted to 1 if LINEAR = True)
    XYflip = True or False whether the x-axis of the CDF is used as the dependent variable for the polynomial fit (True for TECu)
    LINEAR = True or False to apply a linear fit to the cutoff region (True for TECu)
    AVG = averaging window for smoothing the dual-frequency ionosphere correction before putting it into the histogram (or CDF)
    md_TECu_alt = array of by-cycle zero-significance value for dual-frequency TECu with MP = True (see iono_level)
    md_TECu_gim =  array of by-cycle zero-significance value for GIM TECu with MP = True (see iono_level)
    zs_TECu_alt = array of by-cycle zero-significance value for dual-frequency TECu with MP = False (see iono_level, DEFAULT)
    zs_TECu_gim = array of by-cycle zero-significance value for GIM TECu with MP = False (see iono_level, DEFAULT)
    rs_md_bias = final relative ionosphere calibration bias with MP = True (see iono_level)
    rs_zs_bias= final relative ionosphere calibration bias with MP = False (see iono_level, DEFAULT)
    '''
    Nc = np.shape(CYCLES)[0]
    fku = 13.575e9
    fc = 5.3e9
    k_iku = (fc**2)/((fku**2)-(fc**2))
    i2tec_ku = ((fku**2)/(-40.3))
    DIMND,OMITlab,MODELFILE = ldic.file_labels(VAR,'ku',DIM,OMIT)
    if 'usla' in VAR:
        BAND1 = 'ku'
    else: 
        BAND1 = BAND
    md_TECu_alt = np.empty((Nc))*np.nan
    md_TECu_gim = np.empty((Nc))*np.nan
    zs_TECu_alt = np.empty((Nc))*np.nan
    zs_TECu_gim = np.empty((Nc))*np.nan
    print('AVERAGING WINDOW = '+str(AVG)+' s')
    for ii in np.arange(Nc):
        srt = time.time()
        CYC1 = str(CYCLES[ii])
        CYC2 = CYC1
        FILE1,FILE2,FILE2_GEO = ldic.filenames_specific(MISSIONNAME,'dir',SOURCE,CYC1,CYC2,BAND1,DIMND,ERR=ERR)
        ds = Dataset(FILE1).variables
        x_grid,y_grid,z_grid = lgrid.grid_dimensions(DIM,OMIT)
        x_obs,y_obs = ldata.pull_observables(ds,DIM,ERR,MISSIONNAME)
        if np.size(OMIT) == 0:
            z_obs = []
        else:
            z_obs = ds[OMIT][:]
        if np.size(z_grid) == 0:
            B_ku = lbi.bilinear_interpolation_sp_griddata(x_obs,y_obs,x_grid,y_grid,Bku_model)
            B_c = lbi.bilinear_interpolation_sp_griddata(x_obs,y_obs,x_grid,y_grid,Bc_model)
        else:
            B_ku = lti.trilinear_interpolation_sp_griddata(x_obs,y_obs,z_obs,x_grid,y_grid,z_grid,Bku_model)
            B_c = lti.trilinear_interpolation_sp_griddata(x_obs,y_obs,z_obs,x_grid,y_grid,z_grid,Bc_model)
        if np.size(ERR)!=0 and ERR!='plrm':
            if 'rms_r' in ERR:
                Rku = np.add(ds['range_ku'][:],ldata.rand_err(ds['range_rms_ku'][:]))
                Rc = np.add(ds['range_c'][:],ldata.rand_err(ds['range_rms_c'][:]))
            elif 'rms_Ewr' in ERR:
                swhE = ds['swh_rms_Ew'][:]
                swhE[swhE<0]=-1.0
                swhE[swhE>=0]=1.0
                Rku = ds['range_ku'][:]+(swhE*ds['range_rms_ku'][:])
                Rc = ds['range_c'][:]+(swhE*ds['range_rms_c'][:])
            else:
                if 'simulate' in VAR:
                    Rku = ds['usla_'+ERR+'_ku'][:]
                    Rc = ds['usla_'+ERR+'_c'][:]
                elif 'switch' in ERR:
                    Rku = ds[VAR+'_ku'][:]
                    Rc = ds[VAR+'_c'][:]
                else:
                    if 'sum_err' in VAR:
                        Rku = ds['range_ku'][:]
                        Rc = ds['range_c'][:]
                    elif 'rng_corr' in ERR:
                        Rku = ds['range_ku'][:]
                        Rc = ds['range_c'][:]
                    else:
                        Rku = ds[VAR+'_'+ERR+'_ku'][:]
                        Rc = ds[VAR+'_'+ERR+'_c'][:]

        else:
            Rku = ds['range_ku'][:]
            Rc = ds['range_c'][:]
        ts = ds['time'][:]
        IkuG = ds['iono_corr_gim_ku'][:]
        IkuA = k_iku*((Rku+B_ku)-(Rc+B_c))
        
        inanIak = np.where(~np.isnan(IkuA))[0]
        inanIgk = np.where(~np.isnan(IkuG))[0]
        inanD = np.intersect1d(inanIgk,inanIak)
        #print('no nan:'+str(np.shape(inanD)[0]))
        
        sIgim_ku,ttt = ldata.imel_iono_avg(np.take(IkuG,inanD),np.take(ts,inanD),AVG)
        sIalt_ku,ttt = ldata.imel_iono_avg(np.take(IkuA,inanD),np.take(ts,inanD),AVG)
        
        TECu_gim = ((sIgim_ku)*i2tec_ku)/(1e16) #asc_Igim_ku = np.diff((Igim_ku))
        TECu_alt = ((sIalt_ku)*i2tec_ku)/(1e16) #asc_Ialt_ku = np.diff((Ialt_ku))
        print('mean/min/max TEC alt: '+str(np.round(np.mean(TECu_alt),3))+'/'+str(np.round(np.min(TECu_alt),3))+'/'+str(np.round(np.max(TECu_alt),3))) 
        md_TECu_alt[ii],cdf_out,poly_fit,bin_out,cdf_x,bin_mid,varres =iono_level(TECu_alt,'tec',cutoff,deg,XYflip,LINEAR,True)
        md_TECu_gim[ii],cdf_out,poly_fit,bin_out,cdf_x,bin_mid,varres =iono_level(TECu_gim,'tec',cutoff,deg,XYflip,LINEAR,True)
        zs_TECu_alt[ii],cdf_out,poly_fit,bin_out,cdf_x,bin_mid,varres =iono_level(TECu_alt,'tec',cutoff,deg,XYflip,LINEAR,False)
        zs_TECu_gim[ii],cdf_out,poly_fit,bin_out,cdf_x,bin_mid,varres =iono_level(TECu_gim,'tec',cutoff,deg,XYflip,LINEAR,False)
        print('cdf: '+str(cdf_out))
        print(FILE1)
        print((time.time()-srt)/60.)
        #print('Time to find iono-calib-bias for '+FILE1+': '+str((time.time()-srt)/60.))+' min'
    dTECga_md = np.subtract(md_TECu_gim,md_TECu_alt) 
    dTECga_zs = np.subtract(zs_TECu_gim,zs_TECu_alt) 
    zval =4.
    x_newA,ioutA,imd = ldata.iterative_outlier_function(dTECga_md,zval)
    dTECga_mdf = dTECga_md[imd]
    x_newA,ioutA,izs = ldata.iterative_outlier_function(dTECga_zs,zval)
    dTECga_zsf = dTECga_zs[izs]
    rs_md_bias = np.mean(dTECga_mdf[~np.isnan(dTECga_mdf)])
    rs_zs_bias = np.mean(dTECga_zsf[~np.isnan(dTECga_zsf)])
    if np.isnan(rs_zs_bias)==True:
        print('gim zero-sig: '+str(zs_TECu_gim))
        print('alt zero-sig: '+str(zs_TECu_alt))
    return md_TECu_alt,md_TECu_gim,zs_TECu_alt,zs_TECu_gim,rs_md_bias,rs_zs_bias  



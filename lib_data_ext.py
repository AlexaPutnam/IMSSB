#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:54:46 2020

@author: alexaputnam

Author: Alexa Angelina Putnam Shilleh
Institution: University of Colorado Boulder
Software: sea state bias model generator (interpolation method)
Thesis: Sea State Bias Model Development and Error Analysis for Pulse-limited Altimetry
Semester/Year: Fall 2021
"""

import numpy as np

###############################################
# DATA FILTERING FUNCTIONS
###############################################
def iterative_outlier_removal(x_in,zval):
    # Purpose: same as  iterative_outlier_function, but adjusted for both direct (Nx1) and difference (Nx2) data
    # x_in = input array
    # zval = z-value used to set threshold for outlier detection. i.e. [min,max] = [mean-zval*σ,  mean+zval*σ])
    # x_new = reduced input array (all outliers removed)
    # iout = indices for all removed data
    #ikp = indices for all preserved data 
    if np.size(x_in) != np.shape(x_in)[0]:
        x_newA,ioutA,ikpA = iterative_outlier_function(x_in[:,0],zval=zval)
        x_newD,ioutD,ikpD = iterative_outlier_function(x_in[:,1],zval=zval)
        ikp1 = np.intersect1d(ikpA,ikpD)
        iout1 = np.unique(np.hstack((ioutA,ioutD)))
        x_inAD = x_in[:,0]-x_in[:,1]
        x_newAD,iout2,ikp2= iterative_outlier_function(x_inAD,zval=zval)
        ikp = np.intersect1d(ikp1,ikp2)
        iout = np.unique(np.hstack((iout1,iout2)))
        x_new = np.copy(x_in[ikp,:])
    else:
        x_new,iout,ikp = iterative_outlier_function(x_in,zval=zval)
    Nkp = np.shape(ikp)[0]
    Nout = np.shape(iout)[0]
    Nin = np.shape(x_in)[0]
    if Nin != Nkp+Nout:
        raise('Nin != Nkp+Nout in iterative_outlier_removal')
    return x_new,iout,ikp

def iterative_outlier_function(x_in,zval):
    # Purpose: iterative outlier removal
    # x_in = input array
    # zval = z-value used to set threshold for outlier detection. i.e. [min,max] = [mean-zval*σ,  mean+zval*σ])
    # x_new = reduced input array (all outliers removed)
    # iout = indices for all removed data
    outliers = 1.
    x_keep = np.copy(x_in)
    x_rmv = np.ones(np.shape(x_in))
    i=0
    while outliers > 0:
        mn = np.mean(x_keep[~np.isnan(x_keep)])
        sig = np.std(x_keep[~np.isnan(x_keep)])
        sig5 = zval*sig
        ciL,ciU= mn-sig5,mn+sig5
        iRmL = np.where(x_keep[:]<ciL)[0] #&(x_keep[:]>ciU))[0]
        iRmU = np.where(x_keep[:]>ciU)[0]
        iRm = np.append(iRmL,iRmU)
        outliers = np.size(iRm)
        i += 1
        if outliers > 0:
            x_keep[iRm] = np.empty(outliers)*np.nan
            x_rmv[iRm] = np.zeros(outliers)
    ix = np.where(~np.isnan(x_keep))[0]
    iout = np.where(np.isnan(x_keep))[0]
    ikpi = np.where(x_rmv==1)[0]
    ikp = np.intersect1d(ix,ikpi)
    x_new = x_keep[ikp]
    Nkp = np.shape(ikp)[0]
    Nout = np.shape(iout)[0]
    Nin = np.shape(x_in)[0]
    if Nin != Nkp+Nout:
        raise('Nin != Nkp+Nout in iterative_outlier_function')
    return x_new,iout,ikp

###############################################
# MATHEMATICAL FUNCTIONS
###############################################
def imel_iono_avg(vI,vT,sW,CUT=True):
    '''
    Purpose: ionosphere smoothing function (similar to Imel’s 1994 paper)
    vI: array of ionosphere correction values
    vT: array of time stamps corresponding to vI
    sW: integer smoothing window size (i.e. 11)
    CUT: True or False to remove datapoints that do not uphold to smoothing criteria
    vIs_out: array of smoothed ionosphere corrections
    iW = indices corresponding to data points that pass the smoothing criteria
    '''
    N = np.shape(vI)[0]
    res = np.round(np.median(np.diff(vT)),3)
    if res>1.2:
        raise('resolution is too large (should be ~1-s), not '+str(res)+'-s')
    buffH = (res*(float(sW)-1.0))+0.5
    buffL = (res*(float(sW)-1.0))-0.5
    if sW>1:
        dT = np.subtract(vT[sW-1:], vT[:-sW+1]) #np.asarray(list(map(operator.sub, vT[sW-1:], vT[:-sW+1])))
        iW = np.where((dT[:]<buffH)&(dT[:]>=buffL))[0]
        '''
        if np.size(iW)!=0:
            print('window ('+str(sW)+') with resolution = '+str(res)+': min = '+str(np.round(dT[iW].min(),3))+' s, max = '+str(np.round(dT[iW].max(),3))+' s')
        else:
            print('window ('+str(sW)+') --> no measurements')
        '''
        vIs = np.convolve(vI, np.ones(sW), 'valid') / sW
    else:
        vIs = np.copy(vI)
        iW = np.arange(N)
    if CUT is True:
        vIs_out = np.take(vIs,iW)
    elif CUT is False:
        vIs_out=np.empty(np.shape(vI))*np.nan
        vIs_out[int((sW-1)/2):int((-sW+1)/2)]=np.copy(vIs)
        iW = iW+int((sW-1)/2)
    return vIs_out,iW

def iono_smooth(vI,vT,sW):
    '''
    Purpose: same as  imel_iono_avg, but adjusted for both direct (Nx1) and difference (Nx2) data
    All variables the same as   imel_iono_avg
    '''
    Nd = np.size(np.shape(vI))
    if Nd == 1:
        vIs_out,iW = imel_iono_avg(vI,vT,sW,CUT=False)
    elif Nd>1:
        vIs_out = np.empty(np.shape(vI))*np.nan
        vIs_out[:,0],ikp0 = imel_iono_avg(vI[:,0],vT[:,0],sW,CUT=False)
        vIs_out[:,1],ikp1 = imel_iono_avg(vI[:,1],vT[:,1],sW,CUT=False)
        iW = np.intersect1d(ikp0,ikp1)
    return vIs_out,iW

###############################################
# altimetry-specific equations
###############################################
def collard_jpl(sigma0, swh, sSat):
    '''******************************************************************************************
    * FUNCTION: GENCOLLARDWINDSPEED
    * Purpose:
    * To compute the altimeter wind speed using the Jason-1 algorithm
    * Collard's algorithm tuned with Jason-1 data.
    * Need to apply a sigma0 calibration bias to align with Jason-1.
    *
    * Input:
    *    sigma0   - Ku-band sigma0 (dB)
    *    swh      - Ku-band swh (m)
    *    sig0bias - Bias to add to sigma0 to make consistent with Jason-1 data (dB)
    *    sSat     - satellite altimeter mission (j1, j2, j3 or tx)
    * Output:
    *    altwind  - Altimeter wind speed (m/s)
    *
    ******************************************************************************************'''
    if sSat == 'j1':
        sig0bias = 0.0 # I found = 0.001
    elif sSat == 'j2':
        sig0bias = 0.32 #supposidly 0.28 for MLE4 (but doesn't match)
    elif sSat == 'j3':
        sig0bias = 0.14    
    elif sSat == 'tx':
        sig0bias = 2.69 #2.789 #0.102+2.69 #2.688
    M_KU=0.0806452
    B_KU=-0.6322581
    M_SWH=0.0683395
    B_SWH=0.0991799
    W_FIRST_11=58.76821
    W_FIRST_12=22.08226
    W_FIRST_21=3.05118
    W_FIRST_22=0.18608
    B_FIRST_1=-33.61161
    B_FIRST_2=1.06796
    W_SECOND_1=-0.20367
    W_SECOND_2=-22.35999
    B_SECOND=20.10259
    B_WIND=0.0869731
    M_WIND=0.0321528
    calsig0=np.add(sigma0,sig0bias)
    p1=np.add(np.multiply(M_KU,calsig0),B_KU)
    p2=np.add(np.multiply(M_SWH,swh),B_SWH)
    e1=np.add(np.add(np.multiply(W_FIRST_11,p1),np.multiply(W_FIRST_12,p2)),B_FIRST_1)
    e2=np.add(np.add(np.multiply(W_FIRST_21,p1),np.multiply(W_FIRST_22,p2)),B_FIRST_2)
    x1f=np.add(1.0,np.exp(-e1))
    x2f=np.add(1.0,np.exp(-e2))
    x1=np.divide(1.0,x1f)
    x2=np.divide(1.0,x2f)
    ee=np.add(np.add(np.multiply(W_SECOND_1,x1),np.multiply(W_SECOND_2,x2)),B_SECOND)
    uf=np.add(1.0,np.exp(-ee))
    uu=np.divide(1.0,uf)
    altwind=np.divide(np.subtract(uu,B_WIND),M_WIND)
    return altwind
    
def rads_sigma0_adj(sigma0,sSat,psi2,BAND):
    '''
    Purpose: used only for RADS data → adjust sigma0 to align with product data
    sigma0 = Ku-band sigma0 from RADS (dB)
    sSat = satellite altimeter mission (j1, j2, j3 or tx)
    psi2 = off-nadit angle (deg^2)
    BAND = altimeter frequency band (ku or c)
    sig0  = adjusted Ku-band sigma0 (dB)
    dsig: Ku- and C-band Sigma0 bias of Jason-1
    scale: Adjust backscatter for correlation with off-nadir angle (See Quartly [2009])
    https://github.com/remkos/rads/blob/master/devel/rads_fix_jason.f90
    '''
    if sSat == 'j1':
        scale = 0.00
        dsig = 0.00
    elif sSat == 'j2':
        if 'ku' in BAND:
            scale = 11.34
            dsig = -2.40
        elif 'c' in BAND:
            scale = 2.01
            dsig = -0.73
    elif sSat == 'j3':
        if 'ku' in BAND:
            scale = 11.34
            dsig = -2.40
        elif 'c' in BAND:
            scale = 2.01
            dsig = -0.73
    # Make sure that sigma0 is adjusted to the product output 
    Nw = np.shape(sigma0)[0]
    noise = np.asarray([(scale*psi2[a]) for a in np.arange(Nw)])
    sig0 = np.asarray([sigma0[ii]-dsig+noise[ii] for ii in np.arange(Nw)])
    return sig0

###############################################
# TOPEX and JASON DATA
###############################################
def rand_skew_norm(skew,loc,scl): 
    '''
    Purpose: generate random numbers that conform to a skewed or normal distribution
    skew = skewness
    loc = location
    scl = scale (>0)
    rnd = random number
    '''
    sigma = skew/np.sqrt(1.0 +skew**2) 
    afRN = np.random.randn(2)
    u0 = afRN[0]
    v = afRN[1]
    u1 = sigma*u0 + np.sqrt(1.0 -sigma**2) * v 
    if u0 >= 0:
        rnd=u1*scl+loc
        return rnd
    rnd = (-u1)*scl+loc
    return rnd 

def randn_skew(size, skew=0.0,scl=1):
    '''
    Purpose: same as  rand_skew_norm, but adjusted to allow for multiple random values
    All variables the same as    rand_skew_norm
    size = size of array output containing random values
    '''
    if size == 1:
        rnd = rand_skew_norm(skew, 0, scl)
    else:
        rnd = [rand_skew_norm(skew, 0, scl) for x in range(size)]
    return rnd

def rand_err(rms):
    '''
    Purpose: same as  rand_skew, but adjusted for both direct (Nx1) and difference (Nx2) data given rms values from altimeter measurements
    rms = equivalent to scl on rand_skew
    err = equivalent to rnd in rand_skew
    '''
    N = np.shape(rms)[0]
    dim=np.size(np.shape(rms))
    # swh skew: skew[swh<1.5]=2 given skew = np.zeros(np.shape(swh))
    skew=np.zeros(np.shape(rms))
    if dim ==1:
        err = np.empty(N)*np.nan
        for ii in np.arange(N):
            if rms[ii]<=0:
                err[ii] = 0.0
            else:
                err[ii]=randn_skew(1, skew=skew[ii],scl=np.abs(rms[ii]))#np.random.normal(loc=0.0,scale=np.abs(rms[ii]),size=1)[0]
    elif dim==2:
        err = np.empty((N,2))*np.nan
        for ii in np.arange(N):
            if rms[ii,0]<=0:
                err[ii,0]=0.0
            else:
                err[ii,0] = randn_skew(1, skew=skew[ii,0],scl=np.abs(rms[ii,0]))#np.random.normal(loc=0.0,scale=np.abs(rms[ii,0]),size=1)[0]
            if rms[ii,1]<=0:
                err[ii,1] = 0.0
            else:
                err[ii,1] = randn_skew(1, skew=skew[ii,0],scl=np.abs(rms[ii,1]))#np.random.normal(loc=0.0,scale=np.abs(rms[ii,1]),size=1)[0]
    inn = np.where(np.isnan(err))[0]
    if np.size(inn)!=0:
        raise('error in creating random error')
    return err

def pull_observables(ds,DIM,ERR,MISSIONNAME,APP='matrix2model'):
    '''
    Purpose: provide the observables for creating the observation matrices and interpolation, as well as for the error analysis. They can change depending on the chosen parameters and whether or not ERR is defined in Steps -2 and -4. 
    ds = data2matrix dataset from /data2vector/ directory
    DIM, ERR, MISSIONNAME,APP = defined in ‘Running the IMSSB software’ section of documentation and run scripts
    '''
    if np.size(ERR)==0:
        ERR = 'none'
    else:
            if ERR=='rms_rws':
                ERR2='_rms_ws'
            elif ERR=='rms_rw':
                ERR2='_rms_w'
            elif ERR=='rms_wx':
                ERR2='_rms_w'
            elif ERR=='rms_Ewr':
                ERR2='_rms_Ew'
            elif ERR == 'rms_rs':
                ERR2='_rms_s'
            elif ERR =='rms_r':
                ERR2=''
            elif ERR =='rng_corr':
                ERR2=''
            else:
                ERR2='_'+ERR

    if ERR in ['rms_w','rms_ws','rms_rw','rms_rws']:
        if 'matrix2model' in APP:
            x_obs = ds['swh'+ERR2][:]
        else:
            x_obs = (np.add(ds['swh_ku'][:],rand_err(ds['swh_rms_ku'][:])))#np.abs
    elif ERR == 'mle3':
        x_obs = ds['swh_ku_mle3'][:]
    elif ERR == 'smth_swh':
        if 'matrix2model' in APP:
            x_obs = ds['swh_'+ERR2][:]
        else:
            H = ds['swh_ku'][:]
            T = ds['time'][:]
            x_obs,ikp = iono_smooth(H,T,11)
    elif ERR == 'plrm':
        x_obs = ds['swh_ku_plrm'][:]
    elif ERR == 'none':
        x_obs = ds['swh_ku'][:]
    elif ERR in ['txj1','j1j2','j2j3','j3s6A']:
        x_obs = ds['swh_ku'][:]
    elif ERR in ['rms_s','rms_rs','rms_r','rng_corr']:
        x_obs = ds['swh_ku'][:]
    if 'u10' in DIM:
        if ERR in ['rms_s','rms_ws','rms_rs','rms_rws']:
            if 'matrix2model' in APP:
                y_obs = ds[DIM+ERR2][:]
            else:
                y_obs = collard_jpl(np.add(ds['sig0_ku'][:],rand_err(ds['sig0_rms_ku'][:])),x_obs,MISSIONNAME)
        elif ERR =='mle3':
            y_obs = ds['wind_speed_alt_mle3'][:]
        elif ERR =='plrm':
            y_obs = ds['wind_speed_alt_plrm'][:]
        elif ERR=='none':
            y_obs = ds['wind_speed_alt'][:]
        elif ERR in ['txj1','j1j2','j2j3','j3s6A']:
            y_obs = ds['wind_speed_alt'][:]
        elif ERR in ['rms_w','rms_rw','rms_r','rng_corr','smth_swh']:
            y_obs = collard_jpl(ds['sig0_ku'][:],x_obs,MISSIONNAME)
    elif 'sig0' in DIM:
        if ERR =='mle3':
            y_obs = ds['sig0_ku_mle3'][:]
        elif ERR =='plrm':
            y_obs = ds['sig0_ku_plrm'][:]
        else:
            y_obs = np.copy(ds['sig0_ku'][:])
    else:
        raise('No other dimension defined')
    return x_obs,y_obs


###############################################
# SENTINEL-3 DATA
# 2018: /ALT1/usr/aputnam/jpl_summer_2018/sentinel3/s3_zip_out/s3_reduced/2018s3/
# 2019: /ALT1/usr/aputnam/jpl_summer_2018/sentinel3/s319_zip_out/s319_zip_reduced/
###############################################


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:28:02 2020

@author: alexaputnam

Author: Alexa Angelina Putnam Shilleh
Institution: University of Colorado Boulder
Software: sea state bias model generator (interpolation method)
Thesis: Sea State Bias Model Development and Error Analysis for Pulse-limited Altimetry
Semester/Year: Fall 2021
"""
import time
import numpy as np
import scipy as sp
import numpy.linalg as npl
from netCDF4 import Dataset
from os import path

import lib_dict_ext as ldic
import lib_geo_gridding_ext as lgrid
import lib_data_ext as ldata
import lib_bilinear_interpolation_ext as lbi
import lib_trilinear_interpolation_ext as lti
import lib_filter_ext as lflt

def crt_tempfile_idx(iout1,ikp1,FILE1):
    '''
    Purpose: create a temporary file (FILE1) that is saved to the directory postfit_outlier_idx_temp that provides the indices 
    for the outlying measurements (iout), as well as the indices for measurements that have passed (ikp1).
    '''
    root_grp = Dataset(FILE1,'w', format='NETCDF4')
    root_grp.description = "This file contains the valid USLA points for the postfit residual iterative outlier detector"
    root_grp.history = "Author: Alexa Angelina Putnam Shilleh. Created at/for: CU Boulder/JPL" + time.ctime(time.time())
    if np.size(iout1)!=0:
        dimO = np.shape(iout1)[0]
        root_grp.createDimension('idx_out', dimO)
        y2 = root_grp.createVariable('obs_out', 'f8', ('idx_out',))
        y2[:] = iout1
        y2.units = 'none'
    else:
        dimO = 1
        root_grp.createDimension('idx_out', dimO)
        y2 = root_grp.createVariable('obs_out', 'f8', ('idx_out',))
        y2[:] = np.nan
        y2.units = 'none'
    dimK = np.shape(ikp1)[0]
    root_grp.createDimension('idx_keep', dimK)
    y1 = root_grp.createVariable('obs_keep', 'f8', ('idx_keep',))
    y1[:] = ikp1
    y1.units = 'none'
    root_grp.close()


# --------------------------------------------------------------------------#
#-------------------weighting-----------------------------------------------#
# --------------------------------------------------------------------------#
def latitude_weight_beta(lat_deg,Vs,Ve,I_deg):
    '''
    Purpose: Beta calculation for latitude weighting (Nerem, 1995: Measuring global mean sea level variations using TOPEX/POSEIDON altimeter data)
    lat_deg= degree latitude 
    Ve = [km/s] linear velocity of the Earth's rotation
    Vs =  [km/s] along-track ground velocity of the satellite
    I_deg = [deg] satellite orbit inclination
    '''
    lat_deg_nn = np.copy(lat_deg)
    lat_rad = (np.pi/180.)*lat_deg_nn
    I_rad = (np.pi/180.)*I_deg
    alpha = np.arcsin(np.abs(np.divide(np.cos(I_rad),np.cos(lat_rad))))
    if I_deg>90.:
        gamma = np.abs(np.divide((Vs*np.sin(alpha))+(Ve*np.cos(lat_rad)),Vs*np.cos(alpha)))
    elif I_deg<90.:
        gamma = np.abs(np.divide((Vs*np.sin(alpha))-(Ve*np.cos(lat_rad)),Vs*np.cos(alpha)))
    beta = (np.pi/2.)-np.arctan(gamma)
    return beta

def latitude_weight_1995(lat_deg,Vs,Ve,I_deg):
    '''
    Purpose: Weighting factor defined by latitude and orbit characteristics (Nerem, 1995: Measuring global mean sea level variations using TOPEX/POSEIDON altimeter data)
    lat_deg = degree latitude 
    Ve = [km/s] linear velocity of the Earth's rotation
    Vs =  [km/s] along-track ground velocity of the satellite
    I_deg = [deg] satellite orbit inclination
    wgt_lat = weighting factor
    '''
    beta = latitude_weight_beta(lat_deg,Vs,Ve,I_deg)
    beta0 = latitude_weight_beta(0.0,Vs,Ve,I_deg)
    wgt_lat = np.divide(np.sin(beta),np.sin(beta0))
    #r_lat_1994 = np.divide(np.sin(beta),np.sin(beta0))*np.cos((np.pi/180.)*lat_deg)
    return wgt_lat

def latitude_weight(lat_deg,MISSIONNAME):
    # Purpose: running function for  latitude_weight_1995
    Ve,Vs,I_deg,alt = ldic.mission_specs(MISSIONNAME)
    iout = np.where(np.abs(lat_deg)>I_deg)[0]
    if np.size(iout)!=0:
        print('abs(latitude) > '+str(I_deg))
    wgt_lat = latitude_weight_1995(lat_deg,Vs,Ve,I_deg)
    return wgt_lat
       
def deltaT_weighting(dT):
    '''
    Purpose: exponential function for weighting the difference between timestamps from crossover or collinear differences.
    dT = timestamp#1 (ascending or pass I) – timestamp#2 (descending or pass i+1)
    wgt = weighting factor  from exponential function only
    '''
    wgt = np.exp(-np.abs(dT)/3.)
    return wgt

def colVxov_weighting(dT):
    '''
    Purpose: provides the weighting factor applied to collinear and crossover differences based on the difference between 
    timestamps and whether or not the observations are retrieved from collinear or crossover differences. Crossover differences 
    are given greater weight since there are significantly less crossover differences then collinear differences.
    dT = timestamp#1 (ascending or pass I) – timestamp#2 (descending or pass i+1)
    wgt_obs = weighting factor
    '''
    # scale the number of crossovers to match the collinear measurements
    # Number of collinear measurements per bin for every crossover measurement per bin
    #col_frac = np.nansum(col_cnt_sum_ssbc)/(np.nansum(col_cnt_sum_ssbx))
    wgt_dT = deltaT_weighting(dT)
    k_xov = 230.  #178 #(dUx_vr_inv-dUx_vr_inv.min()).max()-5.55283096
    b_xov = 120.
    wgt_obs = ((wgt_dT*k_xov)+b_xov)#/350.
    return wgt_obs

# --------------------------------------------------------------------------#
#-------------------information and measurement matrices--------------------# 
def cholesky(HtH,HtB):
    '''
    Purpose: perform Cholesky decomposition (linear equation: B = HB_est)
    HtH = H transpose H, where H = matrix containing partials
    HtB = H transpose B, where B = observation vector (i.e. usla)
    B_est = SSB estimates
    '''
    t_srt = time.time()
    U = sp.linalg.cholesky(HtH,lower=False)
    z = np.linalg.inv(U.T).dot(HtB)
    B_est = np.linalg.inv(U).dot(z)      
    t_end = time.time()
    print('Time to perform cholesky decomposition '+str(t_end-t_srt)+' seconds')
    return B_est

def lse_matrices(H,z,w=[],P=[]):
    '''
    Purpose: create matrices for least squares problem
    H = matrix containing partials
    z = dependent variable vector, or observation vector, (i.e. usla)
    w = weighting factors
    P = apriori
    HtH = H transpose H
    Htz = H transpose z
    '''
    Sw = np.size(w)
    if Sw == 0:
        HtH = (H.T.dot(H)).toarray()
        Htz = (H.T.dot(z)) 
    elif Sw == 1:
        HtH = np.multiply((H.T.dot(H)).toarray(),w)
        Htz = np.multiply((H.T.dot(z)),w)
    elif Sw > 1:
        HtH = (H.T.dot(w.dot(H))).toarray()
        Htz = (H.T.dot(w.dot(z)))
    if np.size(P) != 0:
        HtH = np.add(HtH,P) #,np.linalg.inv(P))
    return HtH,Htz

def Hmat_ones(H):
    # Purpose: convert all non-zero values in the partials matrix (H) to one and compute HtH to use as a method for counting the number of observations per node.
    H1=H.copy()
    #print('H1 shape: '+str(H1.shape))
    H1[H1!=0]=1
    HtH1=H1.T.dot(H1).toarray()
    return HtH1

 
def bilinear_lut2ssb_analysis_geo(H_ssb,H_geo,B_obs,idx_ssb_half,idx_geo_half,wgt,METHOD,P=[]):
    ##################
    # Full Model Info
    ##################
    Nssb = np.shape(H_ssb)[1]/2
    inn = np.where(~np.isnan(B_obs))[0]
    # only activate below for MLE3
    if 'met3' in METHOD:
        idx_ssb = np.hstack((idx_ssb_half,Nssb+idx_ssb_half))
    else:
        idx_ssb = np.copy(idx_ssb_half)
    H_ssb_col = H_ssb[:,idx_ssb]
    obs2keepA0= lgrid.row2keep(H_ssb,idx_ssb)
    obs2keepA = np.intersect1d(obs2keepA0,inn)
    if np.size(wgt)!=0:
        wgt2keep = np.where(wgt>0)[0]
        obs2keep = np.intersect1d(obs2keepA,wgt2keep)
    else:
        obs2keep = np.copy(obs2keepA)
    H_ssb_lut = H_ssb_col[obs2keep,:]
    B_lut = B_obs[obs2keep]
    if np.size(wgt) != 0:
        wgt_kp = wgt[obs2keep]
        wgt_lut = sp.sparse.spdiags(wgt_kp, 0, wgt_kp.size, wgt_kp.size)
	#np.savez_compressed('nowgts_'+METHOD, wgt_lut)
    else:
        wgt_lut = []
    ##### SSB analysis
    # unweighted least squares
    HtH_ones = Hmat_ones(H_ssb_lut)
    HtH_ssb,HtB_ssb = lse_matrices(H_ssb_lut,B_lut,w=wgt_lut,P=P)
    ##### GEO analysis
    #print('shape of H_geo: '+str(np.shape(H_geo)))
    if np.size(idx_geo_half)!=0:
        if 'met3' in METHOD:
            Ngeo = np.shape(H_geo)[1]/2
            idx_geo = np.hstack((idx_geo_half,Ngeo+idx_geo_half))
        else:
            idx_geo = np.copy(idx_geo_half)
	    #print('size idx_geo: '+str(np.size(idx_geo)))
        H_geo_lut_row = H_geo[obs2keep,:]
        H_geo_lut = H_geo_lut_row[:,idx_geo]
        H_lut = sp.sparse.hstack([H_ssb_lut,H_geo_lut])
        HtH,HtB = lse_matrices(H_lut,B_lut,w=wgt_lut,P=P)
    else:
        HtH,HtB=[],[]
    return HtH,HtB,HtH_ssb,HtB_ssb,HtH_ones

def lse2ssb(HtH,HtB,fN,mN,idx_half,fN_geo=[],mN_geo=[],idx_geo_half=[]):
    '''
    Purpose: determine unleveled SSB model using Cholesky
    HtH,HtB = SSB LS matrices 
    fN = size of flattened SSB model grid 
    mN = shape of SSB model grid
    idx_half = indices for SSB grid nodes with sufficient number of observations 
    fN_geo = size of flattened spatial grid (COE only) - UNUSED
    mN_geo=shape spatial grid (COE only) - UNUSED
    idx_geo_half  = indices for spatial grid nodes with sufficient number of observations – UNUSED
    B_Nd_1 = unleveled raw SSB model (Ku-band if simultaneously estimated (met3). Not default.)
    FE_Nd_1 = formal error (unscaled) for SSB model (Ku-band if simultaneously estimated (met3). Not default.)
    B_geo_1 = unused
    FE_geo_1 = unused
    B_Nd_2 = unleveled raw SSB model (C-band if simultaneously estimated (met3). Not default.)
    FE_Nd_2 = formal error (unscaled) for SSB model (C-band if simultaneously estimated (met3). Not default.)
    B_geo_2 = unused
    FE_geo_2 = unused
    '''
    srt = time.time()
    Nssb_half = np.shape(idx_half)[0]
    if np.size(fN_geo)!=0:
        Ngeo_half = np.shape(idx_geo_half)[0]
        sz_chk = int((HtH.shape[0]-2.*np.shape(idx_geo_half)[0])/2)
    else:
        sz_chk = int(HtH.shape[0]/2)
    if sz_chk==Nssb_half:
        met3 = True
        idx = np.hstack((idx_half,Nssb_half+idx_half))
    else:
        met3 = False
        idx = np.copy(idx_half)
    Nssb_full = np.shape(idx)[0]
    print('shape HtH: '+str(np.shape(HtH)))
    print('shape HtB: '+str(np.shape(HtB)))
    fB_est = cholesky(HtH,HtB)#cholesky(HtH[i50,:][:,i50],HtB[i50])
    fFE = np.diag(np.linalg.inv(HtH))#np.diag(np.linalg.inv(HtH[i50,:][:,i50]))
    if np.size(fN_geo)!=0:
        if met3 is True:
            fssb_all = fB_est[:Nssb_full]
            ffe_all = fFE[:Nssb_full]
            fgeo_all = fB_est[Nssb_full:]
            ffe_geo_all = fFE[Nssb_full:]
            
            B_Nd_1 = lgrid.data2grid(fssb_all[:Nssb_half],fN,mN,idx_half)
            B_Nd_2 = lgrid.data2grid(fssb_all[Nssb_half:],fN,mN,idx_half)
            
            FE_Nd_1 = lgrid.data2grid(np.sqrt(ffe_all[:Nssb_half]),fN,mN,idx_half)
            FE_Nd_2 = lgrid.data2grid(np.sqrt(ffe_all[Nssb_half:]),fN,mN,idx_half)
            
            B_geo_1 = lgrid.data2grid(fgeo_all[:Ngeo_half],fN_geo,mN_geo,idx_geo_half)
            B_geo_2 = lgrid.data2grid(fgeo_all[Ngeo_half:],fN_geo,mN_geo,idx_geo_half)
            
            FE_geo_1 = lgrid.data2grid(np.sqrt(ffe_geo_all[:Ngeo_half]),fN_geo,mN_geo,idx_geo_half)
            FE_geo_2 = lgrid.data2grid(np.sqrt(ffe_geo_all[Ngeo_half:]),fN_geo,mN_geo,idx_geo_half)
        else:
            B_Nd_1 = lgrid.data2grid(fB_est[:Nssb_full],fN,mN,idx_half)
            B_Nd_2 = []
            
            FE_Nd_1 = lgrid.data2grid(np.sqrt(fFE[:Nssb_full]),fN,mN,idx_half)
            FE_Nd_2 = []
            
            B_geo_1 = lgrid.data2grid(fB_est[Nssb_full:],fN_geo,mN_geo,idx_geo_half)
            B_geo_2 = []
            
            FE_geo_1 = lgrid.data2grid(np.sqrt(fFE[Nssb_full:]),fN_geo,mN_geo,idx_geo_half)
            FE_geo_2 = []
    else:
        if met3 is True:
            B_Nd_1 = lgrid.data2grid(fB_est[:Nssb_half],fN,mN,idx_half)
            B_Nd_2 = lgrid.data2grid(fB_est[Nssb_half:],fN,mN,idx_half)
            
            FE_Nd_1 = lgrid.data2grid(np.sqrt(fFE[:Nssb_half]),fN,mN,idx_half)
            FE_Nd_2 = lgrid.data2grid(np.sqrt(fFE[Nssb_half:]),fN,mN,idx_half)
            
            B_geo_1 = []
            B_geo_2 = []
            
            FE_geo_1 = []
            FE_geo_2 = []
            
        else:
            B_Nd_1 = lgrid.data2grid(fB_est,fN,mN,idx_half)
            B_Nd_2 = []
            
            FE_Nd_1 = lgrid.data2grid(np.sqrt(fFE),fN,mN,idx_half)
            FE_Nd_2 = []
            
            B_geo_1 = []
            B_geo_2 = []
            
            FE_geo_1 = []
            FE_geo_2 = []
    print('Least squares of summed matrices [sec]: '+str(time.time()-srt))
    return B_Nd_1,FE_Nd_1,B_geo_1,FE_geo_1,B_Nd_2,FE_Nd_2,B_geo_2,FE_geo_2

# --------------------------------------------------------------------------#

def met3_obs_vector(H_ssb_sparse_half,H_geo_sparse_half,Ik,Rk,Rc,P):
    '''
    Purpose: create partial matrices (H) for met3 SSB solver. Met3 simultaneously estimates Ku- and C-band SSB solutions using least squares. This is not the default approach.
    H_ssb_sparse_half = partials matrix for SSB model (Ku)
    H_geo_sparse_half = partials matrix for spatial model (Ku, COE only)
    Ik = Ku-band GIM ionosphere correction
    Rk = Ku-band 1 Hz range measurements
    Rc = C-band 1 Hz range measurements
    P = Ku-band usla+Ik+Rk
    H_ssb_sparse = partials matrix for SSB model (Ku and C)
    H_geo_sparse = partials matrix for spatial model (Ku and C, COE only)
    Y = Ku-band dependent variable stacked atop C-band dependent variable
    '''
    fku = 13.575e9
    fc = 5.3e9
    ka = (fc**2)/(fku**2-fc**2)
    tec = Ik*(fku**2)/(-40.3)
    y1 = P-((1.0+ka)*Rk)+(ka*Rc)
    y2 = (-40.3*tec*(1./(ka*fku**2)))-Rk+Rc
    Nd = np.size(np.shape(Ik))
    if Nd == 1:
        Y = np.hstack((y1,y2))
    elif Nd ==2:
        Y = np.hstack((y1[:,0]-y1[:,1],y2[:,0]-y2[:,1]))
    H11 = H_ssb_sparse_half*(1.0+ka)
    H12 = H_ssb_sparse_half*(-ka)
    H21 = H_ssb_sparse_half
    H22 = H_ssb_sparse_half*(-1.0)
    H1 = sp.sparse.hstack([H11,H12]) # H1 = np.hstack([H11,H12])
    H2 = sp.sparse.hstack([H21,H22]) # H2 = np.hstack([H21,H22])
    H_ssb_sparse = sp.sparse.csc_matrix(sp.sparse.vstack([H1,H2])) # H_ssb_sparse = np.vstack([H1,H2])
    #'''
    if np.size(H_geo_sparse_half)!=0:
        H11g = H_geo_sparse_half.multiply(1.0+ka)
        H12g = H_geo_sparse_half.multiply(-ka)
        H21g = H_geo_sparse_half.multiply(1.0)
        H22g = H_geo_sparse_half.multiply(-1.0)
        H1g = sp.sparse.hstack([H11g,H12g])
        H2g = sp.sparse.hstack([H21g,H22g])
        H_geo_sparse = sp.sparse.csc_matrix(sp.sparse.vstack([H1g,H2g]))
    else:
        H_geo_sparse=[]
    #'''
    return H_ssb_sparse,H_geo_sparse,Y

def weighting_matrix(lat,dt,METHOD,MISSIONNAME):
    '''
    Purpose: running function for colVxov_weighting and latitude_weight to provide the weighting matrix, which includes weighting factors due 
    to latitude (for all models) and the difference between timestamps (for collinear and crossover differences only).
    '''
    Nd = np.size(np.shape(lat))
    if Nd==1:
        lat_in = np.copy(lat)
    elif Nd==2:
        lat_in = np.copy(lat[:,0])
    # Delta-Time weights
    if np.size(dt)!=0:
        if 'met3' in METHOD:
            wgts_t_half = colVxov_weighting(dt)
            wgts_t = np.hstack((wgts_t_half,wgts_t_half))
            wgts_y_half = np.ones(np.shape(wgts_t_half))
            sY1 = 1./(0.0344**2) # eY1 = 0.0344 = eP - (1+ka)eRku + (ka*eRc)
            sY2 = 1./(0.1406**2) # eY2 = 0.1406 = -40.3 * ((fku**2-fc**2)/(fku**2 fc**2))eTEC - eRku + eRc
            wgts_y = np.hstack((wgts_y_half*sY1,wgts_y_half*sY2))
            wgts_m1 = wgts_t#*wgts_y
        else:
            wgts_m1 = colVxov_weighting(dt)
    else:
        wgts_m1=[]
    # Latitude weights
    if np.size(lat)!=0:
        if 'met3' in METHOD:
            wgts_half = latitude_weight(lat_in,MISSIONNAME)
            wgts_lat = np.hstack((wgts_half,wgts_half))
        else:
            wgts_lat = latitude_weight(lat_in,MISSIONNAME)
    else:
        wgts_lat = []
    # Combine weights
    if np.size(dt)!=0 and np.size(lat)!=0:
        print('weight dT and Lat')
        wgts = wgts_m1*wgts_lat
    elif np.size(dt)==0 and np.size(lat)!=0:
        print('weight Lat only')
        wgts = np.copy(wgts_lat)
    elif np.size(dt)!=0 and np.size(lat)==0:
        print('weight dT only')
        wgts = np.copy(wgts_m1)
    elif np.size(dt)==0 and np.size(lat)==0:
        print('no weight applied')
        wgts = []
    return wgts#,idx_valid_wgts


def post_fit_outlier(ds,MISSIONNAME,METHOD,VAR,DIMND,OMIT,BAND,ERR,SSBk,SSBc,Irel,B_raw,iteration,CYC1,OPEN_OCEAN):
    '''
    Purpose: find the indices for the outlying measurements, as well as the indices for measurements that have passed and create a 
    temporary file (FN_temp_sv ) that is saved to the directory postfit_outlier_idx_temp.
    SSBk,SSBc,Irel = SSB Ku and C band solutions, and relative ionosphere correction, respectively. This is only provided if the 
    sought SSB solution is using previously estimated SSB models to calculate the dual-frequency ionosphere correction.
    B_raw = raw SSB solution
    iteration = iteration number (integer)
    CYC1 = cycle number
    idx_keep_final = indices of SSB nodes that are kept (not removed using the iterative outlier detector)
    Nout = number of measurements that were removed from the iterative outlier detection 
    mse = mean square error 
    '''
    PATHMISSIONS = ldic.save_path()
    FN_temp=PATHMISSIONS+'postfit_outlier_idx_temp/'+MISSIONNAME+'_'+METHOD+'_'+BAND+'_c'+str(CYC1)+'_temp_'+str(iteration)+'.nc'
    FN_temp_sv=PATHMISSIONS+'postfit_outlier_idx_temp/'+MISSIONNAME+'_'+METHOD+'_'+BAND+'_c'+str(CYC1)+'_temp_'+str(iteration+1)+'.nc'
    if iteration>1:
        dsi = Dataset(FN_temp).variables
        if dsi['obs_out'][:][0]!=np.nan:
            print('keep size: '+str(np.size(dsi['obs_keep'][:]))+' + out size: '+str(np.size(dsi['obs_out'][:]))+' should equal '+str(np.shape(ds['swh_ku'][:])[0]))
            if np.size(dsi['obs_keep'][:])+np.size(dsi['obs_out'][:])!=np.shape(ds['swh_ku'][:])[0]:
                raise('postres indices do not match')
        else:
            print('keep size: '+str(np.size(dsi['obs_keep'][:]))+' should equal '+str(np.shape(ds['swh_ku'][:])[0]))
            if np.size(dsi['obs_keep'][:])!=np.shape(ds['swh_ku'][:])[0]:
                raise('postres indices do not match')

    Nsig = 5
    if 'u10' in DIMND:
        DIM = 'wind_speed_alt'
        if 'narrow' in DIMND:
            x_grid,y_grid,z_grid = lgrid.grid_dimensions('u10_narrow',OMIT)
        else:
            x_grid,y_grid,z_grid = lgrid.grid_dimensions('u10',OMIT)
    elif 'sig0' in DIMND:
        DIM = 'sig0'
        x_grid,y_grid,z_grid = lgrid.grid_dimensions(DIM,OMIT)
    if 'usla' or 'ssha_dfI' in VAR:
        BAND1 = 'ku'
        if 'c' in BAND:
            BAND2 = 'c'
        elif 'ku' in BAND:
            BAND2 = 'ku'
    else:
        BAND1 = BAND
        BAND2 = BAND
    B_obsi,inonan_pre,inan = lflt.observables(METHOD,VAR,BAND2,ds,SSBk,SSBc,Irel,ERR,MISSIONNAME,OPEN_OCEAN,KEEP2d=True)
    if iteration>1:
        inonan = np.intersect1d(inonan_pre,dsi['obs_keep'][:].astype(int)).astype(int)
    else:
        inonan = np.copy(inonan_pre)
    usla = np.take(B_obsi,inonan,axis=0)
    x_obs = np.take(ds['swh_ku'][:],inonan,axis=0)
    y_obs = np.take(ds[DIM][:],inonan,axis=0)
    if np.size(OMIT) == 0:
        z_obs = []
    else:
        z_obs = ds[OMIT][:]
    if METHOD in ['dir','coe']:
        if np.size(z_grid) == 0:
            ssb = lbi.bilinear_interpolation_sp_griddata(x_obs,y_obs,x_grid,y_grid,B_raw)
        else:
            ssb = lti.trilinear_interpolation_sp_griddata(x_obs,y_obs,z_obs,x_grid,y_grid,z_grid,B_raw)
        inn = np.where(~np.isnan(ssb))[0]
        res = usla[inn]-ssb[inn]
        tt,iout,ikp = ldata.iterative_outlier_removal(res,zval=Nsig)
    else:
        if np.size(z_grid) == 0:
            ssb1 = lbi.bilinear_interpolation_sp_griddata(x_obs[:,0],y_obs[:,0],x_grid,y_grid,B_raw)
            ssb2 = lbi.bilinear_interpolation_sp_griddata(x_obs[:,1],y_obs[:,1],x_grid,y_grid,B_raw)
        else:
            ssb1 = lti.trilinear_interpolation_sp_griddata(x_obs[:,0],y_obs[:,0],z_obs[:,0],x_grid,y_grid,z_grid,B_raw)
            ssb2 = lti.trilinear_interpolation_sp_griddata(x_obs[:,1],y_obs[:,1],z_obs[:,1],x_grid,y_grid,z_grid,B_raw)
        ssb = np.empty(np.shape(x_obs))
        ssb[:,0] = ssb1
        ssb[:,1] = ssb2
        inn = np.where((~np.isnan(ssb1))&(~np.isnan(ssb2)))[0]
        res0 = (usla[inn,0]-ssb[inn,0])
        res1 = (usla[inn,1]-ssb[inn,1])
        res = np.append(res0,res1)
        tt,iout0,ikp0 = ldata.iterative_outlier_removal(res0,zval=Nsig)
        tt,iout1,ikp1 = ldata.iterative_outlier_removal(res1,zval=Nsig)
        iout = np.intersect1d(iout0,iout1)
        ikp = np.intersect1d(ikp0,ikp1)
    mse = (1.0*np.sum(np.square(res)))/(1.0*np.size(res))
    print('mean/std of residuals: '+str(np.round(res.mean(),3))+'/'+str(np.round(res.std(),3)))
    Nout = np.size(iout)
    idx_keep_final = inonan[inn[ikp]]
    idx_out_final = np.setdiff1d(np.arange(np.shape(ds['swh_ku'][:])[0]),idx_keep_final)
    print(str(Nsig)+'-sigma postfit iterative  outlier detector: number of good res/bad res observations: '+str(np.size(idx_keep_final))+'/'+str(np.size(idx_out_final)))
    print(str(Nsig)+'-sigma postfit iterative  outlier detector only - outlier size: '+str(Nout))
    crt_tempfile_idx(idx_out_final,idx_keep_final,FN_temp_sv)
    return idx_keep_final,Nout,mse,FN_temp_sv

def obs_matrix(MISSIONNAME,METHOD,VAR,OMIT,SOURCE,CYC1,CYC2,BAND,DIMND,WEIGHT,WEIGHT_dT,ERR,SSBk,SSBc,Irel,B_raw,iteration,OPEN_OCEAN):
    '''
    Purpose: create the partial matrices and dependent variable vector after iterative outlier detection and applying a data filter
    WEIGHT_dT = True or False to weight timestamp differences
    WEIGHT = True or False to weight by latitude
    SSBk,SSBc,Irel = SSB Ku and C band solutions, and relative ionosphere correction, respectively. This is only provided if the 
    sought SSB solution is using previously estimated SSB models to calculate the dual-frequency ionosphere correction.
    B_raw = raw SSB solution
    iteration = iteration number (integer)
    H_ssb_sparse,H_geo_sparse = SSB and spatial (COE only) partial matrices
    B_obs = dependent variable vector (filtered)
    wgts_out = weighting factors from weighting_matrix
    Nout  = number of measurements that were removed from the iterative outlier detection and removed from filter 
    mse = mean square error
    FN_temp_sv = temporary file saved to the directory postfit_outlier_idx_temp used for the iterative outlier detection
    '''
    FN_temp_sv=[]
    mse = np.nan
    if 'usla' or 'ssha_dfI' in VAR:
        BAND1 = 'ku'
        if 'c' in BAND:
            BAND2 = 'c'
        elif 'ku' in BAND:
            BAND2 = 'ku'
    else: 
        BAND1 = BAND
        BAND2 = BAND
    if 'met3' in METHOD:
        MET = METHOD[:-5]
    else:
        MET = METHOD
    FILE1,FILE2,FILE2_GEO = ldic.filenames_specific(MISSIONNAME,MET,SOURCE,CYC1,CYC2,BAND1,DIMND,ERR=ERR)
    fn_exist_geo = path.exists(FILE2_GEO)
    fn1_exist = path.exists(FILE1)
    print(FILE1)
    if fn1_exist is True:
        ds = Dataset(FILE1).variables
        if WEIGHT_dT == True:
            t = ds['time'][:]
            dt = np.abs(t[:,0]-t[:,1])/(60.*60.*24.)
        else:
            dt= []
        if WEIGHT == True:
            if MET in ['dir','coe']:
                lat = ds['lat'][:]
            else:
                lat = ds['lat'][:,0]
        else:
            lat = []
        wgts = weighting_matrix(lat,dt,METHOD,MISSIONNAME)
        if 'met3' in METHOD:
            Ik = ds['iono_corr_gim_ku'][:]
            Rk4 = ds['range_ku'][:]
            P = ds['usla_ku'][:]+Ik+Rk4
            if ERR == 'mle3' or ERR=='rng_mle3':
                Rku = ds['range_ku_mle3'][:]
            else:
                Rku = ds['range_ku'][:]
            Rc = ds['range_c'][:]
            H_ssb_sparse_npz = np.load(FILE2,allow_pickle=True)
            H_ssb_sparse_half = H_ssb_sparse_npz['arr_0'].tolist()
            if fn_exist_geo is True:
                H_geo_sparse_npz = np.load(FILE2_GEO,allow_pickle=True)
                H_geo_sparse_half = H_geo_sparse_npz['arr_0'].tolist()
            else:
                H_geo_sparse_npz = []
                H_geo_sparse_half = []
            H_ssb_sparse,H_geo_sparse,B_obsi = met3_obs_vector(H_ssb_sparse_half,H_geo_sparse_half,Ik,Rku,Rc,P)
            inonan = np.where(~np.isnan(B_obsi))[0]
            B_obs = np.take(B_obsi,inonan)
            mse=np.nan
            raise('NO POSTFIT RESIDUALS')
        else:
            B_obsi,inonani,inan = lflt.observables(METHOD,VAR,BAND2,ds,SSBk,SSBc,Irel,ERR,MISSIONNAME,OPEN_OCEAN)
            if np.size(B_raw)!=0:
                ikp,Nout,mse,FN_temp_sv = post_fit_outlier(ds,MISSIONNAME,METHOD,VAR,DIMND,OMIT,BAND,ERR,SSBk,SSBc,Irel,B_raw,iteration,CYC1,OPEN_OCEAN)
                inonan = np.intersect1d(ikp,inonani)
            else:
                Nout = 0
                inonan = np.copy(inonani)
                ikp = np.copy(inonani)
            B_obs = np.take(B_obsi,inonan)
            H_ssb_sparse_npz = np.load(FILE2,allow_pickle=True)
            H_ssb_sparse = H_ssb_sparse_npz['arr_0'].tolist()[inonan,:]
            if fn_exist_geo is True:
                H_geo_sparse_npz = np.load(FILE2_GEO,allow_pickle=True)
                H_geo_sparse = H_geo_sparse_npz['arr_0'].tolist()[inonan,:]
            else:
                H_geo_sparse = []
        if np.size(wgts)!=0:
            wgts_out = wgts[inonan]
        else:
            print('no weighting applied AT ALL')
            wgts_out = []
    else:
        H_ssb_sparse,H_geo_sparse,B_obs,wgts_out,Nout,mse,FN_temp_sv = [],[],[],[],[],[],[]
    return H_ssb_sparse,H_geo_sparse,B_obs,wgts_out,Nout,mse,FN_temp_sv

# --------------------------------------------------------------------------#
# --------------------------------------------------------------------------#
#-------------------summed information and measurement matrices-------------#
#-------------------static (DIR, XOV, COL) solution-------------#
def matrix_summation_lse(MISSIONNAME,METHOD,VAR,OMIT,SOURCE,CYCLES,BAND,DIMND,idx_ssb,idx_geo,WEIGHT,ERR,SSBk,SSBc,Irel,B_raw,OPEN_OCEAN,iteration=[]):
    '''
    Purpose: running function for obs_matrix and bilinear_lut2ssb_analysis_geo to sum all by-cycle HtH and Htz matrices for the final least squares problem.
    idx_keep_final = indices of SSB nodes that are observed
    idx_geo = indices of spatial nodes that are observed (COE only)
    HtH_lut,HtB_lut = spatial LS matrices (for COE only)
    HtH_lut_ssb,HtB_lut_ssb = SSB LS matrices 
    HtH_ones = number of observations per node (see  Hmat_ones)
    Noutliers  = number of measurements that were removed from the iterative outlier detection and removed from filter 
    mse = mean square error
    FN_temp_sv = temporary file saved to the directory postfit_outlier_idx_temp used for the iterative outlier detection
    '''
    srt = time.time()
    Noutliers = 0.
    N = np.shape(CYCLES)[0]
    if 'jnt' in METHOD:
        mse = np.empty((N,2))*np.nan
    else:
        mse = np.empty(N)*np.nan
    if 'jnt' in METHOD:
        WEIGHT_dT =True
        if 'met3' in METHOD:
            metx = 'xov_met3'
            metc = 'col_met3'
        else:
            metx = 'xov'
            metc = 'col'
    elif 'xov' in METHOD:
        WEIGHT_dT = True
    else:
        WEIGHT_dT = False
    if 'jnt' in METHOD:
        for ii in np.arange(N):
            srt2 = time.time()
            CYC1 = str(CYCLES[ii])
            CYC2 = CYC1
            # check to see whether collinear file exists
            FILE1,FILE2,FILE2_GEO = ldic.filenames_specific(MISSIONNAME,metc,SOURCE,CYC1,CYC2,'ku',DIMND,ERR=ERR)
            fn_exist = path.exists(FILE1)
            ## Combine collinear and crossover
            H_ssb_sparsex,H_geo_sparsex,B_obsx,dwgtx,nonan2kpX,mse[ii,0],FN_temp_sv = obs_matrix(MISSIONNAME,metx,VAR,OMIT,SOURCE,CYC1,CYC2,BAND,DIMND,WEIGHT,WEIGHT_dT,ERR,SSBk,SSBc,Irel,B_raw,iteration,OPEN_OCEAN)
            if fn_exist is True:
                H_ssb_sparsec,H_geo_sparsec,B_obsc,dwgtc,nonan2kpC,mse[ii,1],FN_temp_svC = obs_matrix(MISSIONNAME,metc,VAR,OMIT,SOURCE,CYC1,CYC2,BAND,DIMND,WEIGHT,WEIGHT_dT,ERR,SSBk,SSBc,Irel,B_raw,iteration,OPEN_OCEAN)
                HtHx,HtBx,HtH_ssbx,HtB_ssbx,HtH_1x = bilinear_lut2ssb_analysis_geo(H_ssb_sparsex,H_geo_sparsex,B_obsx,idx_ssb,idx_geo,dwgtx,METHOD)
                HtHc,HtBc,HtH_ssbc,HtB_ssbc,HtH_1c = bilinear_lut2ssb_analysis_geo(H_ssb_sparsec,H_geo_sparsec,B_obsc,idx_ssb,idx_geo,dwgtc,METHOD)
                print('JNT matrix summation cycle '+CYC1+' with both collinear and crossover differences [sec]: '+str(time.time()-srt2))
                if ii == 0:
                    HtH_lut_ssb = np.add(HtH_ssbc,HtH_ssbx)
                    HtB_lut_ssb = np.add(HtB_ssbc,HtB_ssbx)
                    HtH_ones = np.add(HtH_1c,HtH_1x)   
                    if np.size(idx_geo)!=0:
                        HtH_lut = np.add(HtHc,HtHx)
                        HtB_lut = np.add(HtBc,HtBx)
                else:
                    HtH_lut_ssb = np.add(HtH_lut_ssb,np.add(HtH_ssbc,HtH_ssbx))
                    HtB_lut_ssb = np.add(HtB_lut_ssb,np.add(HtB_ssbc,HtB_ssbx))
                    HtH_ones = np.add(HtH_ones,np.add(HtH_1c,HtH_1x))
                    if np.size(idx_geo)!=0:
                        HtH_lut = np.add(HtH_lut,np.add(HtHc,HtHx))
                        HtB_lut = np.add(HtB_lut,np.add(HtBc,HtBx))
                nonan2kp = nonan2kpX+nonan2kpC
                ## Crossover only: since Collinear takes the difference between cycles it's one cycle short
            else:
                nonan2kp = nonan2kpX
                HtHx,HtBx,HtH_ssbx,HtB_ssbx,HtH_1x = bilinear_lut2ssb_analysis_geo(H_ssb_sparsex,H_geo_sparsex,B_obsx,idx_ssb,idx_geo,dwgtx,METHOD)
                print('JNT matrix summation cycle '+CYC1+' with only crossovers [sec]: '+str(time.time()-srt2))
                if ii == 0:
                    HtH_lut_ssb = np.copy(HtH_ssbx)
                    HtB_lut_ssb = np.copy(HtB_ssbx)
                    HtH_ones = np.copy(HtH_1x)
                    if np.size(idx_geo)!=0:
                        HtH_lut = np.copy(HtHx)
                        HtB_lut = np.copy(HtBx)
                else:
                    HtH_lut_ssb = np.add(HtH_lut_ssb,HtH_ssbx)
                    HtB_lut_ssb = np.add(HtB_lut_ssb,HtB_ssbx)
                    HtH_ones = np.add(HtH_ones,HtH_1x)
                    if np.size(idx_geo)!=0:
                        HtH_lut = np.add(HtH_lut,HtHx)
                        HtB_lut = np.add(HtB_lut,HtBx)
            Noutliers=Noutliers+nonan2kp 
    else:
        for ii in np.arange(N):
            srt2 = time.time()
            CYC1 = str(CYCLES[ii])
            CYC2 = CYC1
            H_ssb_sparse,H_geo_sparse,B_obs,wgts,nonan2kp,mseii,FN_temp_sv = obs_matrix(MISSIONNAME,METHOD,VAR,OMIT,SOURCE,CYC1,CYC2,BAND,DIMND,WEIGHT,WEIGHT_dT,ERR,SSBk,SSBc,Irel,B_raw,iteration,OPEN_OCEAN)
            if np.size(H_ssb_sparse)!=0:
                nonan2kp,mse[ii] = nonan2kp,mseii
                HtH,HtB,HtH_ssb,HtB_ssb,HtH_1 = bilinear_lut2ssb_analysis_geo(H_ssb_sparse,H_geo_sparse,B_obs,idx_ssb,idx_geo,wgts,METHOD)
                print('Analyzing cycle '+CYC1+' [sec]: '+str(time.time()-srt2))
                if ii == 0:
                    HtH_lut_ssb = HtH_ssb
                    HtB_lut_ssb = HtB_ssb
                    HtH_ones = HtH_1
                    if np.size(idx_geo)!=0:
                        HtH_lut = HtH
                        HtB_lut = HtB
                else:
                    HtH_lut_ssb = np.add(HtH_lut_ssb,HtH_ssb)
                    HtB_lut_ssb = np.add(HtB_lut_ssb,HtB_ssb)
                    HtH_ones = np.add(HtH_ones,HtH_1)
                    if np.size(idx_geo)!=0:
                        HtH_lut = np.add(HtH_lut,HtH)
                        HtB_lut = np.add(HtB_lut,HtB)
                Noutliers=Noutliers+nonan2kp
    if np.size(idx_geo)==0:
        print('COE not considered')
        HtH_lut,HtB_lut=[],[]
    print('Matrix summation for least squares [sec]: '+str(time.time()-srt))
    return HtH_lut,HtB_lut,HtH_lut_ssb,HtB_lut_ssb,HtH_ones,Noutliers,mse,FN_temp_sv
# --------------------------------------------------------------------------#
#-------------------Coestimated (COE) Solutuon -----------------------------#
'''
The following three functions are used for Block-wise least squares adjustment. The stps are outlined in 
Chapter 7: Adjustment and Filtering Methods from the 2016 book GPS: Theory, Algorithms and Applications by Xu, Guochang and Xu, Yan. 
The block-wise LS allows one to coestimate an SSB model (Z1) with a spatial solution (B_geo), and is only applied to the COE modeling method. 
The formal error of the SSB and spatial solution are indicated by FE and FE_geo, respectively.
'''
def cdssb_matrices(H_ssb,H_geo,z_obs,idx_ssb,min_obs_cnt):
    # see Chapter 7: Adjustment and Filtering Methods from the 2016 book GPS: Theory, Algorithms and Applications by Xu, Guochang and Xu, Yan.
    # Find the information matrices and information vectors
    H_ssb_col = H_ssb[:,idx_ssb]
    obs2keep0= lgrid.row2keep(H_ssb,idx_ssb)
    inn = np.where(~np.isnan(z_obs))[0]
    obs2keep = np.intersect1d(obs2keep0,inn)
    H_ssb_lut = H_ssb_col[obs2keep,:]
    z_lut = z_obs[obs2keep]
    H_geo_lut_row = H_geo[obs2keep,:]
    H_sm = lgrid.obsmat_count(H_geo_lut_row)
    idx_geo = np.where(H_sm>=4)[0]
    H_geo_lut = H_geo_lut_row[:,idx_geo]
    # Information matrices
    N11 = H_ssb_lut.T.dot(H_ssb_lut) # SSB info matrix
    N12 = H_ssb_lut.T.dot(H_geo_lut) # SSBxGeo info matrix
    N22 = H_geo_lut.T.dot(H_geo_lut) # Geo info matrix
    s1 = H_ssb_lut.T.dot(z_lut)      # SSB info vector
    s2 = H_geo_lut.T.dot(z_lut)      # Geo info vector
    return N11,N12,N22,s1,s2,z_lut,idx_geo,obs2keep


def cdssb_Z1(min_obs_cnt,MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,BAND,DIMND,OMIT,idx_ssb,col_cnt_ssb,col_cnt_geo,ERR,SSBk,SSBc,Irel,P_ssb,mN,OPEN_OCEAN):
    # see Chapter 7: Adjustment and Filtering Methods from the 2016 book GPS: Theory, Algorithms and Applications by Xu, Guochang and Xu, Yan.
    # Determine the SSB solution 
    srt1 = time.time()
    BAND1 = 'ku'
    if 'c' in BAND:
        BAND2 = 'c'
    elif 'ku' in BAND:
        BAND2 = 'ku'
    N = np.shape(CYCLES)[0]
    Nout_all=1
    iteration = 0
    B_raw=[]
    FN_temp_sv=[]
    col_ssb_update=np.zeros(np.shape(col_cnt_ssb))
    col_geo_update=np.zeros(np.shape(col_cnt_geo))
    while Nout_all!=0:
        mse = np.empty(N)*np.nan
        for ii in np.arange(N):
            srt2 = time.time()
            CYC1 = str(CYCLES[ii])
            CYC2 = CYC1
            FILE1,FILE2,FILE2_GEO = ldic.filenames_specific(MISSIONNAME,METHOD,SOURCE,CYC1,CYC2,BAND1,DIMND,ERR=ERR)
            ds = Dataset(FILE1).variables
            if VAR in ['usla','ssha_dfI','rng_mle']:
                if 'rms_' in ERR:
                    Nout_all = 0
                else:
                    if np.size(B_raw)!=0:
                        if ii==0:
                            Nout_all=0
                        ikp,Nout,mse[ii],FN_temp_sv = post_fit_outlier(ds,MISSIONNAME,METHOD,VAR,DIMND,OMIT,BAND,ERR,SSBk,SSBc,Irel,B_raw,iteration,CYC1,OPEN_OCEAN)
                        Nout_all +=Nout
            else:
                Nout_all = 0
            B_obs,inonani,inan = lflt.observables(METHOD,VAR,BAND2,ds,SSBk,SSBc,Irel,ERR,MISSIONNAME,OPEN_OCEAN)
            if np.size(B_raw)!=0 and VAR in ['usla','ssha_dfI','rng_mle']:
                if 'rms_r' in ERR:
                    inonan = np.copy(inonani)
                else:
                    inonan = np.intersect1d(inonani,ikp)
            else:
                inonan = np.copy(inonani)
            H_ssb_sparse_npz = np.load(FILE2,allow_pickle=True)
            H_ssb_sparse = H_ssb_sparse_npz['arr_0'].tolist()[inonan,:]
            H_geo_sparse_npz = np.load(FILE2_GEO,allow_pickle=True)
            H_geo_sparse = H_geo_sparse_npz['arr_0'].tolist()[inonan,:]
            col_ssb_update[ii,:] = lgrid.obsmat_count(H_ssb_sparse.copy())
            col_geo_update[ii,:] = lgrid.obsmat_count(H_geo_sparse.copy())

            n11i,n12i,n22i,s1i,s2i,z_luti,idx_geo,ikp_cd = cdssb_matrices(H_ssb_sparse,H_geo_sparse,B_obs[inonan],idx_ssb,min_obs_cnt)
            n11i,n12i,n22i = n11i.toarray(),n12i.toarray(),n22i.toarray()
            if ii == 0:
                HtH_ones = Hmat_ones(H_ssb_sparse[ikp_cd,:][:,idx_ssb])
                K1 = n12i.dot(npl.inv(n22i).dot(n12i.T))
                S1 = s1i
                K2 = n12i.dot(npl.inv(n22i).dot(s2i))
                N11 = n11i
            else:
                HtH_ones = np.add(HtH_ones,Hmat_ones(H_ssb_sparse[ikp_cd,:][:,idx_ssb]))
                K1 = np.add(K1,n12i.dot(npl.inv(n22i).dot(n12i.T)))
                S1 = np.add(S1,s1i)
                K2 = np.add(K2,n12i.dot(npl.inv(n22i).dot(s2i)))
                N11 = np.add(N11,n11i)
            print('Analyzing COE cdssb_Z1 for cycle '+CYC1+' [sec]: '+str(time.time()-srt2))
            #print('N11 shape: '+str(np.shape(N11)))
        print('Hssb shape: '+str(np.shape(H_ssb_sparse)))
        print('Hgeo shape: '+str(np.shape(H_geo_sparse)))
        ichk_s = np.where((np.diag(N11-K1)==0))[0]#&(np.diag(HtH_ones)<10))[0]
        if np.size(ichk_s)>0:
            print('not positive definite '+str(np.size(ichk_s))+' zero elements--> find new indices')
        cov_z1 = npl.inv(N11-K1)
        Z1 = cov_z1.dot(S1-K2)
        B_raw = lgrid.data2grid(Z1,P_ssb,mN,idx_ssb)
        FE = np.sqrt(np.diag(cov_z1))
        print('post fit residual outlier removal for iteration '+str(iteration)+': number of outliers = '+str(Nout_all))
        if iteration>5:#4:
            raise('too many iterations in post fit residual outlier removal')
        iteration+=1
    print('Time to create COE (cdssb_Z1) [sec]: '+str(time.time()-srt1))
    return Z1,FE,idx_geo,mse,FN_temp_sv,HtH_ones,idx_ssb,col_ssb_update,col_geo_update #Z1,FE_s,idx_geo,mse,FN_temp_sv

def cdssb_Z2(min_obs_cnt,MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,BAND,DIMND,idx_ssb,col_cnt_geo,Z1,ERR,SSBk,SSBc,Irel,FN_temp_sv,OPEN_OCEAN):
    # see Chapter 7: Adjustment and Filtering Methods from the 2016 book GPS: Theory, Algorithms and Applications by Xu, Guochang and Xu, Yan.
    # Determine the geo solution.
    PATHMISSIONS = ldic.save_path()
    srt1 = time.time()
    BAND1 = 'ku'
    if 'c' in BAND:
        BAND2 = 'c'
    elif 'ku' in BAND:
        BAND2 = 'ku'
    N = np.shape(CYCLES)[0]
    P_geo = np.shape(col_cnt_geo)[1]
    B_geo = np.empty((P_geo,N))
    FE_geo = np.empty((P_geo,N))
    for ii in np.arange(N):
        srt2 = time.time()
        CYC1 = str(CYCLES[ii])
        CYC2 = CYC1
        FILE1,FILE2,FILE2_GEO = ldic.filenames_specific(MISSIONNAME,METHOD,SOURCE,CYC1,CYC2,BAND1,DIMND,ERR=ERR)
        ds = Dataset(FILE1).variables
        B_obs,inonani,inan = lflt.observables(METHOD,VAR,BAND2,ds,SSBk,SSBc,Irel,ERR,MISSIONNAME,OPEN_OCEAN)
        if VAR in ['usla','ssha_dfI','rng_mle']:
            if 'rms_r' in ERR:
                inonan = np.copy(inonani)
            else:
                if np.size(FN_temp_sv)!=0:
                    dsi = Dataset(PATHMISSIONS+'postfit_outlier_idx_temp/'+MISSIONNAME+'_'+METHOD+'_'+BAND+'_c'+str(CYC1)+FN_temp_sv[-10:]).variables
                    inonan = np.intersect1d(inonani,dsi['obs_keep'][:].astype(int))
                else:
                    inonan = np.copy(inonani)
        else:
            inonan = np.copy(inonani)
        H_ssb_sparse_npz = np.load(FILE2,allow_pickle=True)
        H_ssb_sparse = H_ssb_sparse_npz['arr_0'].tolist()[inonan,:]
        H_geo_sparse_npz = np.load(FILE2_GEO,allow_pickle=True)
        H_geo_sparse = H_geo_sparse_npz['arr_0'].tolist()[inonan,:]
        n11i,n12i,n22i,s1i,s2i,z_lut,idx_geo,ikp_cd = cdssb_matrices(H_ssb_sparse,H_geo_sparse,B_obs[inonan],idx_ssb,min_obs_cnt)
        n11i,n12i,n22i = n11i.toarray(),n12i.toarray(),n22i.toarray()
        Z2 = npl.inv(n22i).dot(s2i-n12i.T.dot(Z1))
        B_geo[:,ii] = lgrid.data2grid(Z2,P_geo,P_geo,idx_geo)
        FE_geo[:,ii] = lgrid.data2grid(np.sqrt(np.diag(npl.inv(n22i))),P_geo,P_geo,idx_geo)
        print('Analyzing COE cdssb_Z2 for cycle '+CYC1+' [sec]: '+str(time.time()-srt2))
    print('Time to create COE (cdssb_Z2) [sec]: '+str(time.time()-srt1))
    return B_geo,FE_geo


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 08:31:17 2021

@author: alexaputnam

Author: Alexa Angelina Putnam Shilleh
Institution: University of Colorado Boulder
Software: sea state bias model generator (interpolation method)
Thesis: Sea State Bias Model Development and Error Analysis for Pulse-limited Altimetry
Semester/Year: Fall 2021
"""
import numpy as np
from netCDF4 import Dataset

import lib_dict_ext as ldic
import lib_geo_gridding_ext as lgrid
import lib_data_ext as ldata
import lib_bilinear_interpolation_ext as lbi
import lib_dynamic_thresholds_ext as ldyn
#####################################################
# Level-0 Filter
#####################################################
def filtered_dataset(METHOD,SOURCE,usla_ku,usla_c,ds,idx):
    '''
    Purpose: Create a new dataset (NC) that contains all of the filtered and updated variables for Step-2
    idx = indices of valid measurements provided by filter_data2obs (and filter_data)
    ds = product2data netCDF4 file (step-1)
    '''
    dsv = ds.variables
    dsk = list(dsv.keys())
    Nk = np.shape(dsk)[0]
    NC = {}
    NC['units']={}
    for jj in np.arange(Nk):
        var_i = dsk[jj]#.encode()
        print(dsk[jj])
        if 'units' not in var_i:
            NC[var_i]=[]
            NC['units'][var_i] = []
    for jj in np.arange(Nk):
        var_i = dsk[jj]#.encode()
        if 'units' not in var_i:
            NC[var_i]=np.take(dsv[var_i][:],idx,axis=0)
            if var_i not in ['pass_number','equator_longitude']:
                NC['units'][var_i] = 'none'#dsv[var_i].units.encode()#!!!
            else:
                NC['units'][var_i] = 'none'
    #reconstructions and new variables
    NC['ssha_freq_independent_OmTmG'] = np.take(dsv['ssha'][:],idx,axis=0)+np.take(dsv['sea_state_bias_ku'][:],idx,axis=0)+np.take(dsv['iono_corr_alt_ku'],idx,axis=0)+np.take(dsv['range_ku'],idx,axis=0)
    NC['units']['ssha_freq_independent_OmTmG'] = 'm'
    '''
    if SOURCE == 'rads':
        if 'xov' in METHOD or 'col' in METHOD:
            psi2 = np.take(dsv['off_nadir_angle_wf_ku'][:],idx,axis=0) # should be off_nadir_angle2_wf_ku
            sig0rads_ku = np.take(dsv['sig0_ku'][:],idx,axis=0)
            sig0A_ku = ldata.rads_sigma0_adj(sig0rads_ku[:,0],'j2',psi2[:,0],'ku')
            sig0D_ku = ldata.rads_sigma0_adj(sig0rads_ku[:,1],'j2',psi2[:,1],'ku')
            sig0_kuf = np.empty(np.shape(psi2))
            sig0_kuf[:,0],sig0_kuf[:,1] = sig0A_ku,sig0D_ku

            sig0rads_c = np.take(dsv['sig0_c'][:],idx,axis=0)
            sig0A_c = ldata.rads_sigma0_adj(sig0rads_c[:,0],'j2',psi2[:,0],'c')
            sig0D_c = ldata.rads_sigma0_adj(sig0rads_c[:,1],'j2',psi2[:,1],'c')
            sig0_cf = np.empty(np.shape(psi2))
            sig0_cf[:,0],sig0_cf[:,1] = sig0A_c,sig0D_c
        elif 'dir' in METHOD:
            psi2f = np.take(dsv['off_nadir_angle_wf_ku'][:],idx) # should be off_nadir_angle2_wf_ku
            sig0_kuf = ldata.rads_sigma0_adj(np.take(dsv['sig0_ku'][:],idx),'j2',psi2f,'ku')
            sig0_cf = ldata.rads_sigma0_adj(np.take(dsv['sig0_c'][:],idx),'j2',psi2f,'c')
        NC['sig0_ku']=sig0_kuf
        NC['sig0_c']=sig0_cf
    ''' 
    NC['usla_ku'] = np.take(usla_ku,idx,axis=0)
    NC['units']['usla_ku'] = 'm'
    NC['usla_c'] = np.take(usla_c,idx,axis=0)
    NC['units']['usla_c'] = 'm'
    return NC

def filter_data(dsv,usla_ku,usla_c,MISSIONNAME):
    '''
    Purpose: Filter-out masked values and latitude values greater than the inclination of the orbit, and provide the indices (idx) 
    of all the measurements that pass through the filter. See level-1 filter from dissertation.
    dsv = product2data netCDF4 file (step-1)
    '''
    lat = dsv['lat'][:]
    Ve,Vs,I_deg,alt = ldic.mission_specs(MISSIONNAME)
    if np.size(np.shape(lat)) == 1:
        iOcean = np.where(~np.isnan(usla_ku+usla_c))[0]
        idxLAT = np.where(np.abs(lat)<I_deg)[0]
        idx = np.intersect1d(iOcean,idxLAT)
    else:
        raise('This function only works for DIR data')
    return idx

def filter_data2obs(MISSIONNAME,METHOD,CYC1,ds):
    '''
    Purpose: compute usla_ku and usla_c and filter_data function to obtain the indices of valid measurements (idx).
    ds = product2data netCDF4 file (step-1)
    '''
    fku = 13.575e9
    fc = 5.3e9
    kc = (fku**2)/(fc**2)
    dsv = ds.variables
    cyc = int(CYC1)-1
    cS,cE = cyc,cyc+2
    # Find USLA: ALT or GIM, Ku or C
    # col - this structure only works for RADS
    if 'col' in METHOD:
        Rku=dsv['range_ku'][:,cS:cE]
        if MISSIONNAME=='s6Ah':
            Rc = Rku
        else:
            Rc = dsv['range_c'][:,cS:cE]
        if 'tx' in MISSIONNAME:
            if int(CYC1)<48:
                Iku = dsv['iono_corr_alt_ku'][:,cS:cE]
                print('using iono_corr_alt_ku for cycle '+str(int(CYC1)))
            else:
                Iku = dsv['iono_corr_gim_ku'][:,cS:cE]
                print('using iono_corr_gim_ku for cycle '+str(int(CYC1)))
        else:
            Iku = dsv['iono_corr_gim_ku'][:,cS:cE]
            print('using iono_corr_gim_ku for cycle '+str(int(CYC1)))
        P = dsv['ssha'][:,cS:cE]+dsv['sea_state_bias_ku'][:,cS:cE]+dsv['iono_corr_alt_ku'][:,cS:cE]+Rku
        usla_ku,usla_c = P-Rku-Iku,P-Rc-(kc*Iku)
        idx_pre = filter_data(dsv,usla_ku,usla_c,MISSIONNAME)
    else:
        t_xov = dsv['time'][:]
        Rku=dsv['range_ku'][:]
        if MISSIONNAME=='s6Ah':
            Rc = Rku
        else:
            Rc = dsv['range_c'][:]
        if 'tx' in MISSIONNAME:
            if int(CYC1)<48:
                Iku = dsv['iono_corr_alt_ku'][:]
                print('using iono_corr_alt_ku for cycle '+str(int(CYC1)))
            else:
                Iku = dsv['iono_corr_gim_ku'][:]
                print('using iono_corr_gim_ku for cycle '+str(int(CYC1)))
        else:
            Iku = dsv['iono_corr_gim_ku'][:]
            print('using iono_corr_gim_ku for cycle '+str(int(CYC1)))
        P = dsv['ssha'][:]+dsv['sea_state_bias_ku'][:]+dsv['iono_corr_alt_ku'][:]+Rku
        usla_ku,usla_c = P-Rku-Iku,P-Rc-(kc*Iku)
        idx_pre = filter_data(dsv,usla_ku,usla_c,MISSIONNAME)
    if 'xov' in METHOD:
        dTx = np.abs(t_xov[:,0]-t_xov[:,1])/(60.*60.*24.)
        ixov = np.where(dTx<10.0)[0]
        idx = np.intersect1d(idx_pre,ixov)
    else:
        idx = np.copy(idx_pre)
    return idx,usla_ku,usla_c


#####################################################
# Level-1/2 Filter
#####################################################
def filt_orig2new(ds,axis,MN,ERR,USLA_in,BAND,OPEN_OCEAN,MLE3=False):
    '''
    Purpose: Apply level-2 filter (see dissertation) and provide the indices (il2) of all the measurements that pass through the filter. 
    ds = data2matrix netCDF4 file (step-2)
    USLA_in = dependent variable vector
    axis = [], 0, or 1. Empty brackets for direct measurements, and 0 or 1 for collinear and crossover differences. 0 = ascending or pass i. 1 = descending or pass i+1. 
    '''
    if MN=='s3':
        if ERR=='plrm':
            MODE='plrm'
            MD='_plrm'
        else:
            MODE='sar'
            MD=''
    else:
        MODE=[]
        MD=''
    print(list(ds.keys()))
    if np.size(axis)==0:
        swh_ku = ds['swh_ku'+MD][:]
        usla = np.copy(USLA_in)
        Twr = ds['rad_wet_tropo_corr'][:]# radiometer filter is specific to ERA!
        Twm = ds['model_wet_tropo_corr'][:]# radiometer filter is specific to ERA!
        rng_rms_ku = ds['range_rms_ku'+MD][:]
        swh_rms_ku = ds['swh_rms_ku'+MD][:]
        sig0_rms_ku = ds['sig0_rms_ku'+MD][:]
        if MN not in ['s6Ah','sa']:
            swh_c = ds['swh_c'][:]
            rng_rms_c = ds['range_rms_c'][:]
            swh_rms_c = ds['swh_rms_c'][:]
            sig0_rms_c = ds['sig0_rms_c'][:]
            if MODE=='sar':
                psi2fi = ds['off_nadir_angle_wf_ku_sar'][:]
            elif MN=='sw':
                psi2fi = np.sqrt(ds['off_nadir_angle2_wf_rms_ku'][:])
            else:
                psi2fi = ds['off_nadir_angle_wf_ku'][:]
        if MN=='sw':
            rad_surf_type = ds['surface_type_rad1'][:]
        elif MN not in ['sa']:
            rad_surf_type = ds['rad_surf_type'][:]
        if MN=='sw':
            rad_d2l = ds['dist_coast'][:]
            bath = ds['topo'][:]
        else:
            rad_d2l = ds['rad_distance_to_land'][:]
            bath = ds['bathymetry'][:]
        if MLE3==True:
            swh_ku3 = ds['swh_ku_mle3'][:]
            #usla_ku3 = ds['usla_ku'][:]+ds['range_ku'][:]-ds['range_ku_mle3'][:]
            rng_rms_ku3 = ds['range_rms_ku_mle3'][:]
            swh_rms_ku3 = ds['swh_rms_ku_mle3'][:]
            sig0_rms_ku3 = ds['sig0_rms_ku_mle3'][:]
            rng_numval_ku3 = ds['range_numval_ku_mle3'][:]# = [10.,21.5]#[10.0,20.5] # number of measurements used in the 1Hz measurements
            sig0_ku3 = ds['sig0_ku_mle3'][:]# = [0,40]
            u10_ku3 = ds['wind_speed_alt_mle3'][:]# = [0,30]
            aMr = ds['alt'][:]-ds['range_ku_mle3'][:]
    else:
        swh_ku = ds['swh_ku'+MD][:,axis]
        usla = np.copy(USLA_in[:,axis])
        Twr = ds['rad_wet_tropo_corr'][:,axis]# radiometer filter is specific to ERA!
        Twm = ds['model_wet_tropo_corr'][:,axis]# radiometer filter is specific to ERA!
        rng_rms_ku = ds['range_rms_ku'+MD][:,axis]
        swh_rms_ku = ds['swh_rms_ku'+MD][:,axis]
        sig0_rms_ku = ds['sig0_rms_ku'+MD][:,axis]
        if MN not in ['s6Ah','sa']:
            swh_c = ds['swh_c'][:,axis]
            rng_rms_c = ds['range_rms_c'][:,axis]
            swh_rms_c = ds['swh_rms_c'][:,axis]
            sig0_rms_c = ds['sig0_rms_c'][:,axis]
            if MODE=='sar':
                psi2fi = ds['off_nadir_angle_wf_ku_sar'][:,axis]
            elif MN=='sw':
                psi2fi = np.sqrt(ds['off_nadir_angle2_wf_rms_ku'][:,axis])
            else:
                psi2fi = ds['off_nadir_angle_wf_ku'][:,axis]
        if MN=='sw':
            rad_surf_type = ds['surface_type_rad1'][:,axis]
        elif MN not in ['sa']:
            rad_surf_type = ds['rad_surf_type'][:,axis]
        if MN=='sw':
            rad_d2l = ds['dist_coast'][:,axis]
            bath = ds['topo'][:,axis]
        else:
            rad_d2l = ds['rad_distance_to_land'][:,axis]
            bath = ds['bathymetry'][:,axis]
        if MLE3==True:
            swh_ku3 = ds['swh_ku_mle3'][:,axis]
            #usla_ku3 = ds['usla_ku'][:,axis]+ds['range_ku'][:,axis]-ds['range_ku_mle3'][:,axis]
            rng_rms_ku3 = ds['range_rms_ku_mle3'][:,axis]
            swh_rms_ku3 = ds['swh_rms_ku_mle3'][:,axis]
            sig0_rms_ku3 = ds['sig0_rms_ku_mle3'][:,axis]
            rng_numval_ku3 = ds['range_numval_ku_mle3'][:,axis]# = [10.,21.5]#[10.0,20.5] # number of measurements used in the 1Hz measurements
            sig0_ku3 = ds['sig0_ku_mle3'][:,axis]# = [0,40]
            u10_ku3 = ds['wind_speed_alt_mle3'][:,axis]# = [0,30]
            aMr = ds['alt'][:,axis]-ds['range_ku_mle3'][:,axis]
    # Global limits
    dTw = Twr-Twm
    if OPEN_OCEAN==True:
        dist_mx = np.asarray([50.0,6000.0])*1000.#50000.0 #J2 looks like it has a default value at 3600000 (doesn't appear to do any harm though) 
        bath_mx = [-400000,-300.0]
    else:
        dist_mx = np.asarray([0,30.0])*1000.
        bath_mx = [-400000,0]
    swh_mx=[0.,12.5]#[1.5,4]
    #u10_mx=[3,7]#[0.,12.5]  (u10<u10_mx[1])&(u10>u10_mx[0])&
    usla_mx = [-1.5,1.5]
    dtropo_mx = [-0.075,0.065]# <--6-sigma | [-0.043,0.057]# <-- 4-sigma 
    if MN not in ['s6Ah','sa']:
        ikpHc = ldyn.dynamic_threshold(swh_rms_c,swh_ku,'swh_rms_c',MN,MODE)# dynamic_threshold
        print('filter swh_rms_c: '+str(np.size(ikpHc))+'/'+str(np.size(swh_ku)))
        ikpSc = ldyn.dynamic_threshold(sig0_rms_c,swh_ku,'sig0_rms_c',MN,MODE)# dynamic_threshold
        print('filter sig0_rms_c: '+str(np.size(ikpSc))+'/'+str(np.size(swh_ku)))
        ikpRc = ldyn.dynamic_threshold(rng_rms_c,swh_ku,'range_rms_c',MN,MODE)# dynamic_threshold
        print('filter range_rms_c: '+str(np.size(ikpRc))+'/'+str(np.size(swh_ku)))
    # MLE3-specific limits
    rng_numval_ku3_mx = [10.,21.5]#[10.0,20.5] # number of measurements used in the 1Hz measurements
    sig0_ku3_mx = [0,40]
    u10_ku3_mx = [0,30]
    aMr_mx = [-130.,100.]
    if MLE3==False:
        #print('MLE3 is FALSE')
        ikpHk = ldyn.dynamic_threshold(swh_rms_ku,swh_ku,'swh_rms_ku',MN,MODE)# dynamic_threshold
        print('filter swh_rms_ku: '+str(np.size(ikpHk))+'/'+str(np.size(swh_ku)))
        ikpSk = ldyn.dynamic_threshold(sig0_rms_ku,swh_ku,'sig0_rms_ku',MN,MODE)# dynamic_threshold #np.where(sig0_rms_ku<0.61)[0]#
        print('filter sig0_rms_ku: '+str(np.size(ikpSk))+'/'+str(np.size(swh_ku)))
        ikpRk = ldyn.dynamic_threshold(rng_rms_ku,swh_ku,'range_rms_ku',MN,MODE)# dynamic_threshold
        print('filter range_rms_ku: '+str(np.size(ikpRk))+'/'+str(np.size(swh_ku)))
        
        irad_d2l = np.where((rad_d2l>=dist_mx[0])&(rad_d2l<dist_mx[1]))[0]
        print('filter rad_d2l: '+str(np.size(irad_d2l))+'/'+str(np.size(swh_ku)))
        print('mean rad_d2l: '+str(np.nanmean(rad_d2l)))
        iuslai = np.where((usla<usla_mx[1])&(usla>usla_mx[0]))[0]
        print('filter usla: '+str(np.size(iuslai))+'/'+str(np.size(swh_ku)))
        iswh_kui = np.where((swh_ku<swh_mx[1])&(swh_ku>swh_mx[0]))[0]
        print('filter swh_ku: '+str(np.size(iswh_kui))+'/'+str(np.size(swh_ku)))
        iswh_ci = np.where((swh_c<swh_mx[1])&(swh_c>swh_mx[0]))[0]
        print('filter swh_c: '+str(np.size(iswh_ci))+'/'+str(np.size(swh_ku)))
        ibathi = np.where((bath>=bath_mx[0])&(bath<=bath_mx[1]))[0]
        print('filter bath: '+str(np.size(ibathi))+'/'+str(np.size(swh_ku)))
        idTwi = np.where((dTw>dtropo_mx[0])&(dTw<dtropo_mx[1]))[0]
        print('filter dTw: '+str(np.size(idTwi))+'/'+str(np.size(swh_ku)))
        isurfi = np.where((rad_surf_type==0))[0]
        print('filter rad surf: '+str(np.size(isurfi))+'/'+str(np.size(swh_ku)))

        if MN=='s6Ah':
            iOceani = np.where((rad_d2l>=dist_mx[0])&(rad_d2l<dist_mx[1])&(usla<usla_mx[1])&(usla>usla_mx[0])&(swh_ku<swh_mx[1])&(swh_ku>swh_mx[0])&(bath>=bath_mx[0])&(bath<=bath_mx[1])&(rad_surf_type==0)&(dTw>dtropo_mx[0])&(dTw<dtropo_mx[1]))[0]
            ikp = np.intersect1d(ikpHk,np.intersect1d(ikpRk,ikpSk))
        elif MN=='sa':
            iOceani = np.where((rad_d2l>=dist_mx[0])&(rad_d2l<dist_mx[1])&(usla<usla_mx[1])&(usla>usla_mx[0])&(swh_ku<swh_mx[1])&(swh_ku>swh_mx[0])&(bath>=bath_mx[0])&(bath<=bath_mx[1])&(dTw>dtropo_mx[0])&(dTw<dtropo_mx[1]))[0]
            ikp = np.intersect1d(ikpHk,np.intersect1d(ikpRk,ikpSk))
        elif MN=='sw':
            iOceani = np.where((usla<usla_mx[1])&(usla>usla_mx[0])&(swh_ku<swh_mx[1])&(swh_ku>swh_mx[0])&(swh_c<swh_mx[1])&(swh_c>swh_mx[0])&(bath>=bath_mx[0])&(bath<=bath_mx[1])&(rad_surf_type==0)&(dTw>dtropo_mx[0])&(dTw<dtropo_mx[1]))[0]
            ikpO = ldyn.dynamic_threshold(psi2fi,swh_ku,'off_nadir_angle_wf_ku',MN,MODE)
            print('filter off nadir angle: '+str(np.size(ikpO))+'/'+str(np.size(swh_ku)))
            ikp = np.intersect1d(ikpHk,np.intersect1d(ikpHc,np.intersect1d(ikpRk,np.intersect1d(ikpRc,np.intersect1d(ikpSk,np.intersect1d(ikpSc,ikpO))))))
            print('filter Oceani: '+str(np.size(iOceani))+'/'+str(np.size(swh_ku)))
        else:
            iOceani = np.where((rad_d2l>=dist_mx[0])&(rad_d2l<dist_mx[1])&(usla<usla_mx[1])&(usla>usla_mx[0])&(swh_ku<swh_mx[1])&(swh_ku>swh_mx[0])&(swh_c<swh_mx[1])&(swh_c>swh_mx[0])&(bath>=bath_mx[0])&(bath<=bath_mx[1])&(rad_surf_type==0)&(dTw>dtropo_mx[0])&(dTw<dtropo_mx[1]))[0]
            ikpO = ldyn.dynamic_threshold(psi2fi,swh_ku,'off_nadir_angle_wf_ku',MN,MODE)
            print('filter off nadir angle: '+str(np.size(ikpO))+'/'+str(np.size(swh_ku)))
            ikp = np.intersect1d(ikpHk,np.intersect1d(ikpHc,np.intersect1d(ikpRk,np.intersect1d(ikpRc,np.intersect1d(ikpSk,ikpSc)))))#np.intersect1d(ikpHk,np.intersect1d(ikpHc,np.intersect1d(ikpRk,np.intersect1d(ikpRc,np.intersect1d(ikpSk,np.intersect1d(ikpSc,ikpO))))))
        print('filter ikp: '+str(np.size(ikp))+'/'+str(np.size(swh_ku)))
        print('filter Oceani: '+str(np.size(iOceani))+'/'+str(np.size(swh_ku)))
        il2=np.intersect1d(iOceani,ikp)
    elif MLE3==True:
        #print('MLE3 is TRUE')
        iOceani3 = np.where((rad_d2l>=dist_mx[0])&(rad_d2l<dist_mx[1])&
              (usla<usla_mx[1])&(usla>usla_mx[0])&
              (swh_ku3<swh_mx[1])&(swh_ku3>swh_mx[0])&(swh_c<swh_mx[1])&(swh_c>swh_mx[0])&
              (bath>=bath_mx[0])&(bath<=bath_mx[1])&(rad_surf_type==0)&
              ((Twr-Twm)>dtropo_mx[0])&((Twr-Twm)<dtropo_mx[1]))[0]
        ikpHk3 = ldyn.dynamic_threshold(swh_rms_ku3,swh_ku3,'swh_rms_ku_mle3',MN,MODE)# dynamic_threshold
        ikpSk3 = ldyn.dynamic_threshold(sig0_rms_ku3,swh_ku3,'sig0_rms_ku_mle3',MN,MODE)# dynamic_threshold #np.where(sig0_rms_ku3<0.11)[0]# 
        ikpRk3 = ldyn.dynamic_threshold(rng_rms_ku3,swh_ku3,'range_rms_ku_mle3',MN,MODE)# dynamic_threshold
        ikp3 = np.intersect1d(ikpHk3,np.intersect1d(ikpHc,np.intersect1d(ikpRk3,np.intersect1d(ikpRc,np.intersect1d(ikpSk3,ikpSc)))))
        ilv0 = np.where((aMr>=aMr_mx[0])&(aMr<=aMr_mx[1])&(rng_numval_ku3>=rng_numval_ku3_mx[0])&(rng_numval_ku3<=rng_numval_ku3_mx[1])&
                        (sig0_ku3>=sig0_ku3_mx[0])&(sig0_ku3<=sig0_ku3_mx[1])&(u10_ku3>=u10_ku3_mx[0])&(u10_ku3<=u10_ku3_mx[1]))[0]
        il2=np.intersect1d(iOceani3,np.intersect1d(ikp3,ilv0))
    return il2

def pull_tdm_models(MN,MN2,ERR):
    '''
    Purpose: Pull models for tandem analysis. These models were created at the same time at the same location, but different satellites.
    MN,MN2 = mission name of older mission, mission name of younger mission (i.e. MN=j1 and MN2=j2)
    ERR = in this case must be equal to one of the following options: txj1, j1j2 or j2j3. The option j3s6A will be added soon.
    Bk1,Bc1,Irel1 = Ku-band SSB model, C-band SSB model and relative ionosphere correction for MN
    Bk2,Bc2,Irel2 = Ku-band SSB model, C-band SSB model and relative ionosphere correction for MN2
    '''
    PATHMISSIONS = ldic.save_path()
    if ERR == 'txj1':
        if MN == 'tx':
            dsK1 = Dataset(PATHMISSIONS+'tx/dir/matrix2model/tx_dir_ssb_matrix2model_jpl_344_c_364_ku_u10_2d_LatWgt.nc')
            dsC1 = Dataset(PATHMISSIONS+'tx/dir/matrix2model/tx_dir_ssb_matrix2model_jpl_344_c_364_c_u10_2d_LatWgt.nc')
            dsK2 = Dataset(PATHMISSIONS+'j1/dir/matrix2model/j1_dir_ssb_matrix2model_jpl_1_c_21_ku_u10_2d_LatWgt.nc')
            dsC2 = Dataset(PATHMISSIONS+'j1/dir/matrix2model/j1_dir_ssb_matrix2model_jpl_1_c_21_c_u10_2d_LatWgt.nc')

        elif MN2 == 'tx':
            dsK1 = Dataset(PATHMISSIONS+'j1/dir/matrix2model/j1_dir_ssb_matrix2model_jpl_1_c_21_ku_u10_2d_LatWgt.nc')
            dsC1 = Dataset(PATHMISSIONS+'j1/dir/matrix2model/j1_dir_ssb_matrix2model_jpl_1_c_21_c_u10_2d_LatWgt.nc')
            dsK2 = Dataset(PATHMISSIONS+'tx/dir/matrix2model/tx_dir_ssb_matrix2model_jpl_344_c_364_ku_u10_2d_LatWgt.nc')
            dsC2 = Dataset(PATHMISSIONS+'tx/dir/matrix2model/tx_dir_ssb_matrix2model_jpl_344_c_364_c_u10_2d_LatWgt.nc')
    elif ERR == 'j1j2':
        if MN == 'j1':
            dsK1 = Dataset(PATHMISSIONS+'j1/dir/matrix2model/j1_dir_ssb_matrix2model_jpl_240_c_259_ku_u10_2d_LatWgt.nc')
            dsC1 = Dataset(PATHMISSIONS+'j1/dir/matrix2model/j1_dir_ssb_matrix2model_jpl_240_c_259_c_u10_2d_LatWgt.nc')
            dsK2 = Dataset(PATHMISSIONS+'j2/dir/matrix2model/j2_dir_ssb_matrix2model_jpl_1_c_20_ku_u10_2d_LatWgt.nc')
            dsC2 = Dataset(PATHMISSIONS+'j2/dir/matrix2model/j2_dir_ssb_matrix2model_jpl_1_c_20_c_u10_2d_LatWgt.nc')

        elif MN2 == 'j1':
            dsK1 = Dataset(PATHMISSIONS+'j2/dir/matrix2model/j2_dir_ssb_matrix2model_jpl_1_c_20_ku_u10_2d_LatWgt.nc')
            dsC1 = Dataset(PATHMISSIONS+'j2/dir/matrix2model/j2_dir_ssb_matrix2model_jpl_1_c_20_c_u10_2d_LatWgt.nc')
            dsK2 = Dataset(PATHMISSIONS+'j1/dir/matrix2model/j1_dir_ssb_matrix2model_jpl_240_c_259_ku_u10_2d_LatWgt.nc')
            dsC2 = Dataset(PATHMISSIONS+'j1/dir/matrix2model/j1_dir_ssb_matrix2model_jpl_240_c_259_c_u10_2d_LatWgt.nc')
    elif ERR == 'j2j3':
        if MN == 'j2':
            dsK1 = Dataset(PATHMISSIONS+'j2/dir/matrix2model/j2_dir_ssb_matrix2model_jpl_281_c_303_ku_u10_2d_LatWgt.nc')
            dsC1 = Dataset(PATHMISSIONS+'j2/dir/matrix2model/j2_dir_ssb_matrix2model_jpl_281_c_303_c_u10_2d_LatWgt.nc')
            dsK2 = Dataset(PATHMISSIONS+'j3/dir/matrix2model/j3_dir_ssb_matrix2model_jpl_1_c_23_ku_u10_2d_LatWgt.nc')
            dsC2 = Dataset(PATHMISSIONS+'j3/dir/matrix2model/j3_dir_ssb_matrix2model_jpl_1_c_23_c_u10_2d_LatWgt.nc')

        elif MN2 == 'j2':
            dsK1 = Dataset(PATHMISSIONS+'j3/dir/matrix2model/j3_dir_ssb_matrix2model_jpl_1_c_23_ku_u10_2d_LatWgt.nc')
            dsC1 = Dataset(PATHMISSIONS+'j3/dir/matrix2model/j3_dir_ssb_matrix2model_jpl_1_c_23_c_u10_2d_LatWgt.nc')
            dsK2 = Dataset(PATHMISSIONS+'j2/dir/matrix2model/j2_dir_ssb_matrix2model_jpl_281_c_303_ku_u10_2d_LatWgt.nc')
            dsC2 = Dataset(PATHMISSIONS+'j2/dir/matrix2model/j2_dir_ssb_matrix2model_jpl_281_c_303_c_u10_2d_LatWgt.nc')
    Bk1 = dsK1.variables['ssb'][:]
    Bc1 = dsC1.variables['ssb'][:]
    Irel1 = dsK1.variables['iono_calib_bias'][:]*(-0.01216)
    Bk2 = dsK2.variables['ssb'][:]
    Bc2 = dsC2.variables['ssb'][:]
    Irel2 = dsK2.variables['iono_calib_bias'][:]*(-0.01216)
    return Bk1,Bc1,Irel1,Bk2,Bc2,Irel2

def tandem_mn(MN,ERR):
    '''
    Purpose: Provide the name of one mission (MN2) in a tandem analysis given the name of the other mission (MN) and ERR, where 
    in this case, ERR must be equal to one of the following options: txj1, j1j2 or j2j3. The option j3s6A will be added soon.
    '''
    if MN == 'tx':
        MN2='j1'
    elif MN =='j1':
        if 'j1j2' in ERR:
            MN2='j2'
        if 'txj1' in ERR:
            MN2='tx'
    elif MN =='j2':
        if 'j1j2' in ERR:
            MN2='j1'
        if 'j2j3' in ERR:
            MN2='j3'
    elif MN =='j3':
        if 'j2j3' in ERR:
            MN2='j2'
        if 'j3s6A' in ERR:
            MN2='s6A'
    elif MN =='s6A':
        if 'j3s6A' in ERR:
            MN2='j3'
    return MN2

def tdm_test(ds,MN2):
    '''
    Purpose: A tandem analysis test to make sure that all of the measurements between mission#1 (MN) and mission#2 (MN2) 
    have been properly aligned. If not properly aligned, then an error will be raised.
    ds = data2matrix netCDF4 file (step-2) that provides measurements for both MN and MN2, which can only be obtained 
    if the data2matrix file was created with ERR = txj1, j1j2 or j2j3. The option j3s6A will be added soon.
    '''
    t1 = ds['time'][:]
    t2 = ds['time_'+MN2][:]
    dt = abs(t2-t1)
    iH = np.where(dt>90)[0]
    dt_max = np.max(dt)
    print('~~~~~~~~~~~~DT_MAX= '+str(dt_max))
    if np.size(iH)>0:
        print('tandem not aligned')
    return iH

def ssb_solution(x_obs,y_obs,SSB):
    '''
    Purpose: Evaluate an SSB model (SSB) given the model parameters, or altimeter measurements, (x_obs = SWH and y_obs = U10 (default)) 
    to produce SSB estimates (B_est) for each point of measurement.
    '''
    x_grid,y_grid,z_grid = lgrid.grid_dimensions('u10',[])
    sz = np.size(np.shape(x_obs))
    if sz==1:
        B_est = lbi.bilinear_interpolation_sp_griddata(x_obs,y_obs,x_grid,y_grid,SSB)
    elif sz==2:
        B_est = np.empty(np.shape(x_obs))*np.nan
        B_est[:,0] = lbi.bilinear_interpolation_sp_griddata(x_obs[:,0],y_obs[:,0],x_grid,y_grid,SSB)
        B_est[:,1] = lbi.bilinear_interpolation_sp_griddata(x_obs[:,1],y_obs[:,1],x_grid,y_grid,SSB)
    return B_est

def get_iono_alt(ds,ERR,SSBk,SSBc,Irel,BAND):
    '''
    Purpose: Calculate the dual-frequency ionosphere correction given the Ku-band SSB model (SSBk) and C-band SSB model (SSBc), along with the relative ionosphere correction.
    ds = data2matrix netCDF4 file (step-2) 
    '''
    fku = 13.575e9
    fc = 5.3e9
    if BAND=='ku':
        k_iku = (fc**2)/((fku**2)-(fc**2))
    elif BAND=='c':
        k_iku = (fku**2)/((fku**2)-(fc**2))
    x_obs,y_obs = ldata.pull_observables(ds,'u10',ERR,[])
    B_ku = ssb_solution(x_obs,y_obs,SSBk)
    B_c = ssb_solution(x_obs,y_obs,SSBc)
    if ERR == 'mle3' or ERR=='rng_mle3':
        Rku = ds['range_ku_mle3'][:]
    else:
        Rku = ds['range_ku'][:]
    Rc = ds['range_c'][:]
    IkuA = k_iku*((Rku+B_ku)-(Rc+B_c)+Irel)
    return IkuA

def var2observables(METHOD,VAR,BAND2,ds,SSBk,SSBc,Irel,ERR,MN):
    '''
    Purpose: Extracts the dependent variable based on VAR and ERR
    ds = data2matrix netCDF4 file (step-2) 
    SSBk,SSBc,Irel = SSB Ku and C band solutions, and relative ionosphere correction, respectively. This is only provided if 
    the sought SSB solution is using previously estimated SSB models to calculate the dual-frequency ionosphere correction.
    '''
    # import lib_filterL12 as lf12
    fku = 13.575e9
    fc = 5.3e9
    k2c = (fku**2)/(fc**2)
    if np.size(ERR)==0:
        ERR='none'
    print('VAR: '+VAR+', ERR: '+ERR)
    ##### ALT REPORTED AND MEASUREMENT ANALYSIS #####
    if 'usla' in VAR:
        if ERR=='mle3':
            if BAND2=='ku':
                B_obsii = ds['usla_'+BAND2][:]+ds['range_ku'][:]-ds['range_ku_mle3'][:]#+ds['ocean_tide_sol2'][:]-ds['ocean_tide_sol1'][:]
            elif BAND2=='c':
                B_obsii = ds['usla_'+BAND2][:]#+ds['ocean_tide_sol2'][:]-ds['ocean_tide_sol1'][:]
        elif ERR=='plrm':
            if BAND2=='ku':
                B_obsii = ds['ssha_plrm'][:]+ds['sea_state_bias_ku_plrm'][:]
            elif BAND2=='c':
                B_obsii = ds['usla_'+BAND2][:]#+ds['ocean_tide_sol2'][:]-ds['ocean_tide_sol1'][:]
        elif ERR in ['rms_r','rms_wr','rms_rs','rms_rws']:
                if BAND2=='ku':
                    B_obsii = ds['usla_'+BAND2][:]+ldata.rand_err(ds['range_rms_ku'][:])
                elif BAND2=='c':
                    B_obsii = ds['usla_'+BAND2][:]+ldata.rand_err(ds['range_rms_c'][:])
        elif ERR in ['rms_w','rms_s','rms_ws','smth_swh']:
            if BAND2=='ku':
                B_obsii = ds['usla_'+BAND2][:]
            elif BAND2=='c':
                B_obsii = ds['usla_'+BAND2][:]
        elif ERR in ['txj1','j1j2','j2j3','j3s6A']:
            if MN =='tx':
                B_obsii = ds['usla_'+BAND2][:]+ds['ocean_tide_sol2'][:]-ds['ocean_tide_sol1'][:]
            else:
                B_obsii = ds['usla_'+BAND2][:]
        elif ERR=='none':
            print(VAR+' in usla')
            print('data shape: '+str(np.shape(ds['swh_ku'])))
            B_obsii = ds['usla_'+BAND2][:]
    ##### TANDEM ANALYSIS #####
    elif VAR in ['ORB_tdm','ORBin_tdm','ORM_tdm','ORMin_tdm','ORMB_tdm','ORMBin_tdm','I_tdm','I_tdm_uc','SSB_tdm','SLA_tdm','SLA_tdm_uc','USLA_tdm','USLA_tdm_uc','USLAin_tdm','SSH_tdm','SSH_tdm_uc','SLAin_tdm','SLA','SLA_uc','SLAin']:
        print(VAR+' in tdm')
        fku = 13.575e9
        fc = 5.3e9
        k_iku = (fc**2)/((fku**2)-(fc**2))
        if 'tdm' in VAR:
            MN2=tandem_mn(MN,ERR)
            inanH = tdm_test(ds,MN2)
        Rk1 = ds['range_ku'][:]
        Rc1 = ds['range_c'][:]
        B_ku1 = ssb_solution(ds['swh_ku'][:],ds['wind_speed_alt'][:],SSBk)
        B_c1 = ssb_solution(ds['swh_ku'][:],ds['wind_speed_alt'][:],SSBc)
        Iku1 = k_iku*((Rk1+B_ku1)-(Rc1+B_c1)+Irel)
        sIku1,ikp3 = ldata.iono_smooth(Iku1,ds['time'][:],21)
        Iku1uc = k_iku*((Rk1+B_ku1)-(Rc1+B_c1))
        sIku1uc,ikp3 = ldata.iono_smooth(Iku1uc,ds['time'][:],21)
        if VAR == 'I_tdm':
            B_obsii = sIku1
        elif VAR == 'I_tdm_uc':
            B_obsii = sIku1uc
        elif VAR == 'SSB_tdm':
            B_obsii = B_ku1
        A = ds['alt'][:]
        R = ds['range_ku'][:]
        Td = ds['model_dry_tropo_corr'][:]
        Tw = ds['rad_wet_tropo_corr'][:]
        ts = ds['solid_earth_tide'][:]
        to = ds['ocean_tide_fes'][:]
        tp = ds['pole_tide'][:]
        #print(MN)
        #print(ds.keys())
        mss = ds['mean_sea_surface'][:]
        #mss,mss15_err=lgrid.spline_interp_mss(MN,ds['lat'][:],ds['lon'][:],MSS='cnes15')
        '''
        if MN in ['j1','j2']: # and ERR in ['txj1','j2j3']:
        #    #dmss = np.nanmean
            mss = ds['mean_sea_surface_cnes15'][:]
        else:
             mss = ds['mean_sea_surface'][:]
        ''' 
        if MN in ['j1','j2']:
            ib = ds['inv_bar_corr'][:] #j1j2
            hf = ds['hf_fluctuations_corr'][:] #j1j2
            corr = ib+hf
        elif MN in ['tx','j3','s6A','sw']:
            kys = ds.keys()
            if 'internal_tide' in kys and 'internal_tide_hret_' in kys:
                ti = ds['internal_tide'][:] #txj3
            elif 'internal_tide' in kys and 'internal_tide_hret_' not in kys:
                ti = ds['internal_tide'][:] #txj3
            elif 'internal_tide_hret' in kys and 'internal_tide_' in kys:
                ti = ds['internal_tide_hret'][:] #txj3
            elif 'internal_tide_hret' in kys and 'internal_tide_' not in kys:
                ti = ds['internal_tide_hret'][:] #txj3
            dac = ds['dac'][:] #txj3
            te = ds['ocean_tide_non_eq'][:] #txj3
            corr = ti+dac+te
        if VAR == 'SLA_tdm': #in ['SLA_tdm','SLA']:
            B_obsii = A-R-Td-Tw-ts-to-tp-corr-B_ku1-sIku1-mss
        elif VAR == 'SLA': #in ['SLA_tdm','SLA']:
            B_obsii = A-R-Td-Tw-ts-to-tp-corr-B_ku1-sIku1-mss
        elif VAR == 'SLA_tdm_uc': #in ['SLA_tdm_uc','SLA_uc']:
            B_obsii = A-R-Td-Tw-ts-to-tp-corr-B_ku1-sIku1uc-mss
        elif VAR == 'SLA_uc': #in ['SLA_tdm_uc','SLA_uc']:
            B_obsii = A-R-Td-Tw-ts-to-tp-corr-B_ku1-sIku1uc-mss
        elif VAR == 'USLA_tdm':
            B_obsii = A-R-Td-Tw-ts-to-tp-corr-sIku1-mss
        elif VAR == 'USLA_tdm_uc':
            B_obsii = A-R-Td-Tw-ts-to-tp-corr-sIku1uc-mss
        elif VAR == 'SSH_tdm':
            B_obsii = A-R-Td-Tw-ts-to-tp-corr-B_ku1-sIku1
        elif VAR == 'SSH_tdm_uc':
            B_obsii = A-R-Td-Tw-ts-to-tp-corr-B_ku1-sIku1uc
        elif VAR == 'SLAin_tdm': #in ['SLAin_tdm','SLAin']:
            Iku = ds['iono_corr_alt_ku'][:]
            B_ku = ds['sea_state_bias_ku'][:]
            B_obsii = A-R-Td-Tw-ts-to-tp-corr-B_ku-Iku-mss
        elif VAR == 'SLAin': #in ['SLAin_tdm','SLAin']:
            Iku = ds['iono_corr_alt_ku'][:]
            sIku,ikp3 = ldata.iono_smooth(Iku,ds['time'][:],21)
            B_ku = ds['sea_state_bias_ku'][:]
            B_obsii = A-R-Td-Tw-ts-to-tp-corr-B_ku-sIku-mss
        elif VAR == 'USLAin_tdm':
            Iku = ds['iono_corr_alt_ku'][:]
            B_obsii = A-R-Td-Tw-ts-to-tp-corr-Iku-mss
        elif VAR == 'ORB_tdm':
            B_obsii = A-R-B_ku1
        elif VAR == 'ORBin_tdm':
            B_obsii = A-R-ds['sea_state_bias_ku'][:]
        elif VAR == 'ORM_tdm':
            B_obsii = A-R-mss
        elif VAR == 'ORMin_tdm':
            B_obsii = A-R-mss
        elif VAR == 'ORMB_tdm':
            B_obsii = A-R-mss-B_ku1
        elif VAR == 'ORMBin_tdm':
            B_obsii = A-R-mss-ds['sea_state_bias_ku'][:]
    if 'tdm' in VAR:
        B_obsii[inanH] = np.nan
    elif 'dSWH_tdm' in VAR:
        MN2=tandem_mn(MN,ERR)
        inanH = tdm_test(ds,MN2)
        swh1 = ds['swh_ku'][:]
        swh2 = ds['swh_ku_'+MN2][:]
        B_obsii = swh2-swh1
        B_obsii[inanH] = np.nan
    elif 'dWTC_tdm' in VAR:
        MN2=tandem_mn(MN,ERR)
        inanH = tdm_test(ds,MN2)
        orm1 = ds['rad_wet_tropo_corr'][:]
        orm2 = ds['rad_wet_tropo_corr_'+MN2][:]
        B_obsii = orm2-orm1
        B_obsii[inanH] = np.nan
    elif 'dORM_tdm' in VAR:
        MN2=tandem_mn(MN,ERR)
        inanH = tdm_test(ds,MN2)
        orm1 = ds['alt'][:]-ds['range_ku'][:]-ds['mean_sea_surface'][:]# j1,j2,j3 all use CNES-CLS 2011 for MSS
        orm2 = ds['alt_'+MN2][:]-ds['range_ku_'+MN2][:]-ds['mean_sea_surface_'+MN2][:]
        B_obsii = orm2-orm1
        B_obsii[inanH] = np.nan
        ##### VERIFICATION #####
    elif 'bm3' == VAR:
        # Gaspar, 1994: BM3(T) model
        swh = ds['swh_ku'][:]
        ws = ds['wind_speed_alt'][:]
        B_obsii = swh*(0.0036-(0.0045*ws)+(0.00019*ws**2))
        ##### RMS ANALYSIS #####
    elif 'ib_corr' in VAR:
        B_obsii = ds['usla_'+BAND2][:]+ds['hf_fluctuations_corr'][:]
    elif 'alt_cnes' in VAR:
        B_obsii = ds['usla_'+BAND2][:]+ds['alt'][:]-ds['altitude_cnes'][:]
    elif 'alt_jpl' in VAR:
        B_obsii = ds['usla_'+BAND2][:]+ds['alt'][:]-ds['alt_jpl'][:]#alt_nc==alt
    elif 'wet_tropo_model' in VAR:
        B_obsii = ds['usla_'+BAND2][:]+ds['rad_wet_tropo_corr'][:]-ds['model_wet_tropo_corr'][:]
    elif 'iono_tran' in VAR:
        if BAND2=='ku':
            B_obsii = ds['usla_'+BAND2][:]+ds['iono_corr_gim_ku'][:]-ds['iono_corr_alt_ku'][:]
        elif BAND2=='c':
            B_obsii = ds['usla_'+BAND2][:]+(k2c*ds['iono_corr_gim_ku'][:])-(k2c*ds['iono_corr_alt_ku'][:])
    elif 'iono_tran_smooth' in VAR:
        if BAND2=='ku':
            B_obsii = ds['usla_'+BAND2][:]+ds['iono_corr_gim_ku'][:]-ds['iono_corr_alt_ku_smooth'][:]
        elif BAND2=='c':
            B_obsii = ds['usla_'+BAND2][:]+(k2c*ds['iono_corr_gim_ku'][:])-(k2c*ds['iono_corr_alt_ku_smooth'][:])
    elif 'iono_corr_alt' in VAR: #!!!! 
        print(VAR+' is in IONO')
        if BAND2=='ku':
            I2rem = ds['iono_corr_gim_ku'][:]
        elif BAND2=='c':
            I2rem = k2c*ds['iono_corr_gim_ku'][:]
        Ialt = get_iono_alt(ds,ERR,SSBk,SSBc,Irel,BAND2)
        B_obsii = ds['usla_'+BAND2][:]+I2rem-Ialt
    elif 'ocean_fes' in VAR:
        B_obsii = ds['usla_'+BAND2][:]+ds['ocean_tide_got'][:]-ds['ocean_tide_fes'][:]
    elif 'mss_dtu' in VAR:
        B_obsii = ds['usla_'+BAND2][:]+ds['mean_sea_surface'][:]-ds['mean_sea_surface_dtu'][:]
    elif 'mss_cnes15' in VAR:
        B_obsii = ds['usla_'+BAND2][:]+ds['mean_sea_surface'][:]-ds['mean_sea_surface_cnes15'][:]
        # Variable testing
    elif 'mss' == VAR:
        B_obsii = ds['mean_sea_surface'][:]
    elif 'ssb' == VAR:
        B_obsii = ds['sea_state_bias_ku'][:]
    elif 'iono' == VAR:
        B_obsii = ds['iono_corr_alt_ku'][:]
    elif 'gim' == VAR:
        B_obsii = ds['iono_corr_gim_ku'][:]
    elif 'SO' == VAR:
        B_obsii = ds['usla_'+BAND2][:]
        lat = ds['lat'][:]
        B_obsii[lat>-40]=np.nan
    elif 'hemi_south' == VAR:
        B_obsii = ds['usla_'+BAND2][:]
        lat = ds['lat'][:]
        B_obsii[lat>0]=np.nan
    elif 'hemi_north' == VAR:
        B_obsii = ds['usla_'+BAND2][:]
        lat = ds['lat'][:]
        B_obsii[lat<0]=np.nan
    #else:
    #    print('variable (VAR) = '+VAR)
    #    print(list(ds.keys()))
    #    B_obsii = ds[VAR][:]
    print('mean '+VAR+': '+str(np.nanmean(B_obsii)))
    return B_obsii


def observables(METHOD,VAR,BAND2,ds,SSBk,SSBc,Irel,ERR,MN,OPEN_OCEAN,KEEP2d=False):
    '''
    Purpose: Runs var2observables to extract the dependent variable (B_obs) and also runs filt_orig2new to obtain 
    the indices for valid measurements (inonan) and outlying measurements (inan).
    ds = data2matrix netCDF4 file (step-2) 
    SSBk,SSBc,Irel = SSB Ku and C band solutions, and relative ionosphere correction, respectively. This is only 
    provided if the sought SSB solution is using previously estimated SSB models to calculate the dual-frequency ionosphere correction.
    KEEP2d = True or False as to whether to keep collinear and crossover pairs separately as an (Nx2) array. False = take the difference to provide the (Nx1) dependent variable. Default to False.
    '''
    il2=[]
    il2_bd = []
    Ve,Vs,I_deg,alt = ldic.mission_specs(MN)
    B_obsii = var2observables(METHOD,VAR,BAND2,ds,SSBk,SSBc,Irel,ERR,MN)
    if np.size(ERR)==0:
        ERR='none'
    if METHOD in ['dir','coe']:
        B_obs = np.copy(B_obsii)
        inonan = np.where((~np.isnan(B_obs))&(np.abs(ds['lat'][:])<I_deg))[0]#np.intersect1d(np.where(~np.isnan(B_obs))[0],ikp)
        inan = np.where((np.isnan(B_obs))&(np.abs(ds['lat'][:])>=I_deg))[0]#np.intersect1d(np.where(np.isnan(B_obs))[0],iout)
    else:
        print('Ascending-Descending for Observable')
        if KEEP2d is False:
            B_obs = B_obsii[:,0]-B_obsii[:,1]#Original (Asc-Des): B_obsii[:,0]-B_obsii[:,1]  !!! SWITCH FOR TRAN ASSUMPTION
            inonan = np.where((~np.isnan(B_obs))&(np.abs(ds['lat'][:,0])<I_deg))[0]#np.intersect1d(np.where(~np.isnan(B_obs))[0],ikp)
            inan = np.where((np.isnan(B_obs))&(np.abs(ds['lat'][:,0])>=I_deg))[0]#np.intersect1d(np.where(np.isnan(B_obs))[0],iout)
        else:
            B_obs = np.copy(B_obsii)
            inonan = np.where((~np.isnan(B_obs[:,0]-B_obs[:,1]))&(np.abs(ds['lat'][:,0])<I_deg))[0]#np.intersect1d(np.where(~np.isnan(B_obs))[0],ikp)
            inan = np.where((np.isnan(B_obs[:,0]-B_obs[:,1]))&(np.abs(ds['lat'][:,0])>=I_deg))[0]#np.intersect1d(np.where(np.isnan(B_obs))[0],iout)
    if ERR=='mle3':
        MLE3=True
    elif 'mle' in VAR:
        MLE3=True
    else:
        if VAR=='iono_mle':
            MLE3='Both'
        else:
            MLE3=False
    #if 'tdm' in VAR:
    #    B_in = []
    #else:
    if 'usla' in VAR:
    	B_in = np.copy(B_obsii)
    else:
        B_in = ds['usla_ku'][:]
    if np.size(np.shape(ds['swh_ku'][:]))==1:
        il2 = filt_orig2new(ds,[],MN,ERR,B_in,BAND2,OPEN_OCEAN,MLE3=MLE3)
    else:
        il20 = filt_orig2new(ds,0,MN,ERR,B_in,BAND2,OPEN_OCEAN,MLE3=MLE3)
        il21 = filt_orig2new(ds,1,MN,ERR,B_in,BAND2,OPEN_OCEAN,MLE3=MLE3)
        il2 = np.intersect1d(il20,il21)
    il2_bd = np.setdiff1d(np.arange(np.shape(ds['swh_ku'][:])[0]),il2)
    if np.size(il2_bd)!=0:
        print('filter level 0-2 #good/#bad: '+str(np.size(il2))+'/'+str(np.size(il2_bd)))
        inonan=np.intersect1d(inonan,il2)
        inan=np.intersect1d(inan,il2_bd)
    else:
        print('no additional filter')
    return B_obs,inonan,inan 







#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 08:49:58 2020

@author: alexaputnam

Author: Alexa Angelina Putnam Shilleh
Institution: University of Colorado Boulder
Software: sea state bias model generator (interpolation method)
Thesis: Sea State Bias Model Development and Error Analysis for Pulse-limited Altimetry
Semester/Year: Fall 2021
"""
import numpy as np
from netCDF4 import Dataset
import time
import scipy as sp
from os import path #pth2_exist = path.exists(FN1)

import lib_dict_ext as dic
import lib_data_ext as ldata
import lib_geo_gridding_ext as lgrid
import lib_trilinear_interpolation_ext as lti
import lib_bilinear_interpolation_ext as lbi
import lib_filter_ext as lflt
import lib_direct2difference_ext as ld2d
import lib_tandem_ext as ltdm


def crt_data2matrices(METHOD,NC,FILE1,FILE2,FILE2_GEO):
    ''' 
     Purpose: save observation vector file (data2vector) and design matrices (data2matrix)
     FILE1, FILE2, FILE2_GEO = data2vector and data2matrices filenames created in Step-2 (needed to save)
    '''
    root_grp = Dataset(FILE1,'w', format='NETCDF4')
    root_grp.description = "This file contains the "+METHOD+" observation and design matrices required for the interpolation method for modeling SSB"
    root_grp.history = "Author: Alexa Angelina Putnam Shilleh. Institute: University of Colorado Boulder" + time.ctime(time.time())
    dsk =  list(NC.keys())
    Nk = np.shape(dsk)[0]
    if 'info_ssb_cnt' in dsk:
	    iSSB = 'info_ssb_cnt'
	    iGEO = 'info_geo_cnt'
    elif 'col_cnt_ssb' in dsk:
        iSSB = 'col_cnt_ssb'
        iGEO = 'col_cnt_geo'
    dim = np.size(np.shape(NC['swh_ku']))
    ssbdim = np.shape(NC[iSSB])[0]
    root_grp.createDimension('para_ssb', ssbdim)
    y1 = root_grp.createVariable('col_cnt_ssb', 'f8', ('para_ssb',))
    y1[:] = NC[iSSB]
    y1.units = 'none'
    if dim == 2:
        Ndim = np.shape(NC['swh_ku'])[1]
        root_grp.createDimension('t0A_t1D', Ndim)
    Nsz = np.shape(NC['swh_ku'])[0]
    root_grp.createDimension('obs', Nsz)
    for ii in np.arange(Nk):
        if dsk[ii] not in ['info_ssb_cnt','info_geo_cnt','H_ssb_filt','H_geo_filt','units','col_cnt_ssb','col_cnt_geo']:
            if dim == 2:
                xii = root_grp.createVariable(dsk[ii], 'f8', ('obs','t0A_t1D'))
            else:
                xii = root_grp.createVariable(dsk[ii], 'f8', ('obs',))
            xii[:] = NC[dsk[ii]]
            xii.units = NC['units'][dsk[ii]]
    if np.size(FILE2)!=0:
        print('H_ssb_filt size: '+str(np.shape(NC['H_ssb_filt'])))
        np.savez_compressed(FILE2, NC['H_ssb_filt'])
        '''
	    if any(err in FILE1 for err in ['txj1','j1j2','j2j3','j3s6A']):
	        FILE2x = FILE2[:-4]+'_switchX'+FILE2[-4:]
            FILE2y = FILE2[:-4]+'_switchY'+FILE2[-4:]
	        np.savez_compressed(FILE2x, NC['H_ssb_filt_switchX'])
            np.savez_compressed(FILE2y, NC['H_ssb_filt_switchY'])
        '''
    if np.size(FILE2_GEO)!=0:
        geodim = np.shape(NC[iGEO])[0]
        root_grp.createDimension('para_geo', geodim)
        y2 = root_grp.createVariable('col_cnt_geo', 'f8', ('para_geo',))
        y2[:] = NC[iGEO]
        y2.units = 'none'
        print('H_geo_filt size: '+str(np.shape(NC['H_geo_filt'])))
        np.savez_compressed(FILE2_GEO, NC['H_geo_filt'])
    root_grp.close()
    #sp.sparse.save_npz(FILE2, NC['H_ssb_filt'], compressed=True) 
    #sp.sparse.save_npz(FILE2_GEO, NC['H_geo_filt'], compressed=True)
    
# --------------------------------------------------------------------------#
#-------------------sub-functions-------------------------------------------#

def observation_matrix(x_obs,y_obs,z_obs,x_grid,y_grid,z_grid):
    # Purpose: run function for bilinear_observation_matrix and trilinear_observation_matrix 
    # create observation matrices (design matrices) for SSB
    if np.size(z_grid) == 0:
        print('BILINEAR INTERPOLATION MODEL')
        H_sparse,x_scatter,y_scatter = lbi.bilinear_observation_matrix(x_obs,y_obs,x_grid,y_grid) 
        z_scatter = []
    else:
        print('TRILINEAR INTERPOLATION MODEL')
        H_sparse,x_scatter,y_scatter,z_scatter = lti.trilinear_observation_matrix(x_obs,y_obs,z_obs,x_grid,y_grid,z_grid) 
    return H_sparse,x_scatter,y_scatter,z_scatter
    
def geo_obs_mat(lon0,lat,size=3):
    '''
    Purpose: create spatial design matrix for COE model
    lon0,lat = longitude array and latitude array corresponding to measurements
    size = integer defining the degree difference between latitudinal nodes
    lon_grid,lat_grid = reduced Gaussian grid nodes
    geo_cnt = number of observations per node
    H_sparse = spatial design matrix for COE model
    '''
    T1 = time.time()
    lon = lgrid.lon180_to_lon360(lon0)
    lon180 = lgrid.lon360_to_lon180(lon0)
    lon_grid,lat_grid,lon_diff,lat_diff,area = lgrid.reduced_gauss_grid(size)
    lonG180 = lgrid.lon360_to_lon180(lon_grid)
    P = np.shape(lat_grid)[0]
    N = np.shape(lat)[0]
    geo_cnt = np.empty((P))*np.nan
    obs_mat = sp.sparse.lil_matrix((N,P))
    for ii in np.arange(P):
        latl = np.copy(lat_grid[ii]) - np.copy(lat_diff[ii]/2.)
        lath = np.copy(lat_grid[ii]) + np.copy(lat_diff[ii]/2.)
        lonl = np.copy(lonG180[ii]) - np.copy(lon_diff[ii]/2.)
        lonh = np.copy(lonG180[ii]) + np.copy(lon_diff[ii]/2.)
        if lonl<-180:
            lonl_n = lgrid.lon180_to_lon360(lonl+360.)
        else:
            lonl_n = lgrid.lon180_to_lon360(lonl)
        if lonh>180:
            lonh_n = lgrid.lon180_to_lon360(lonh-360.)
        else:
            lonh_n = lgrid.lon180_to_lon360(lonh)
        ij = np.where((lat>=latl)&(lat<lath)&(lon>=lonl_n)&(lon<lonh_n))[0]
        N_ij =  np.shape(ij)[0]
        if N_ij != 0:
            geo_cnt[ii] = N_ij
            obs_mat[ij,ii] = np.reshape(np.ones(N_ij),(N_ij,1)) #np.ones(N_ij)
    H_sparse = obs_mat.tocsr() 
    print('Time to create spatial observation matrix [s]: '+str(time.time()-T1))
    return lon_grid,lat_grid,geo_cnt,H_sparse

# --------------------------------------------------------------------------#
#-------------------data by interpolation model-----------------------------#

def data2matrices(MISSIONNAME,METHOD,NC_in,FILE1,FILE2,FILE2_GEO,DIM,OMIT,MSS,ERR=[]): 
    '''
    Purpose: Step-2: Organize filtered product2data direct measurement files (NC_in) into observation vector file (data2vector) and design matrices (data2matrix) – all within NC_save, and save.
    FILE1, FILE2, FILE2_GEO = data2vector and data2matrices filenames created in Step-2 (needed to save)
    '''
    x_grid,y_grid,z_grid = lgrid.grid_dimensions(DIM,OMIT)
    lon_obs,lat_obs=NC_in['lon'],NC_in['lat']
    if MISSIONNAME+'_dir' in FILE1:
        print('pulling DIRECT parameter measurements from lib_data.py/pull_observables')
        x_obs,y_obs = ldata.pull_observables(NC_in,DIM,ERR,MISSIONNAME)
    else:
        if np.size(ERR)!=0:
            if ERR not in ['txj1','j1j2','j2j3','j3s6A']:
                print('pull parameter measurments from save ERR '+'swh_'+ERR+', and '+DIM+'_'+ERR)
                x_obs = NC_in['swh_'+ERR]
                y_obs = NC_in[DIM+'_'+ERR]
            else:
                print('pulling DIFFERENCE parameter measurements from lib_data.py/pull_observables')
                x_obs,y_obs = ldata.pull_observables(NC_in,DIM,ERR,MISSIONNAME)
                if 'txj1' in ERR:
                   if MISSIONNAME=='tx':
                        MN2 = 'j1'
                   elif MISSIONNAME=='j1':
                        MN2 = 'tx'
                elif 'j1j2' in ERR:
                   if MISSIONNAME=='j1':
                        MN2 = 'j2'
                   elif MISSIONNAME=='j2':
                        MN2 = 'j1'
                elif 'j2j3' in ERR:
                   if MISSIONNAME=='j2':
                        MN2 = 'j3'
                   elif MISSIONNAME=='j3':
                        MN2 = 'j2'
                elif 'j3s6A' in ERR:
                   if MISSIONNAME=='j3':
                        MN2 = 's6A'
                   elif MISSIONNAME=='s6A':
                        MN2 = 'j3'
                x_obs2 = NC_in['swh_ku_'+MN2]
                y_obs2 = NC_in['wind_speed_alt_sig0_'+MN2]
        else:
            print('pulling DIFFERENCE parameter measurements from lib_data.py/pull_observables')
            x_obs,y_obs = ldata.pull_observables(NC_in,DIM,ERR,MISSIONNAME)
    if np.size(OMIT) == 0:
        z_obs = []
    else:
        z_obs = NC_in[OMIT]
    dim = np.size(np.shape(x_obs))
    print('SWH mean '+str(np.mean(x_obs)))
    print(DIM+' mean '+str(np.mean(y_obs)))
    H_ssb,x_scatter,y_scatter,z_scatter = observation_matrix(x_obs,y_obs,z_obs,x_grid,y_grid,z_grid)
    # Remove data not contributing to any of the nodes
    obs2keep= lgrid.row2keep(H_ssb,[])
    dsk = list( NC_in.keys())
    NC_save = {}
    NC_save['units']={}
    Nk = np.shape(dsk)[0]
    for ii in np.arange(Nk):
        if 'units' not in dsk[ii]:
            NC_save['units'][dsk[ii]] = NC_in['units'][dsk[ii]]
            if dim==2:
                print('CHECK HERE')
                print(dsk[ii])
                print(np.shape(NC_in[dsk[ii]]))
                print(np.shape(obs2keep))
                NC_save[dsk[ii]] = np.take(NC_in[dsk[ii]],obs2keep,axis=0)
            elif dim==1:
                NC_save[dsk[ii]] = np.take(NC_in[dsk[ii]],obs2keep)
    if METHOD=='dir':
        sIa,ikp3 = ldata.iono_smooth(NC_in['iono_corr_alt_ku'][:],NC_in['time'][:],21)
        NC_save['iono_corr_alt_ku_smooth']=np.take(sIa,obs2keep,axis=0)
        NC_save['units']['iono_corr_alt_ku_smooth']='m'
    NC_save['H_ssb_filt'] = H_ssb[obs2keep,:]   
    NC_save['info_ssb_cnt'] = lgrid.obsmat_count(NC_save['H_ssb_filt'])
    '''
    if ERR in ['txj1','j1j2','j2j3','j3s6A']:
	print('create mixed tandem matrices (i.e. j1 SWH with WS(swh j1, sig0 j2))')
        H_ssb_sx,x_scatter,y_scatter,z_scatter = observation_matrix(x_obs2,y_obs,z_obs,x_grid,y_grid,z_grid)
        H_ssb_sy,x_scatter,y_scatter,z_scatter = observation_matrix(x_obs,y_obs2,z_obs,x_grid,y_grid,z_grid)
        NC_save['H_ssb_filt_switchX'] = H_ssb_sx[obs2keep,:]
        NC_save['H_ssb_filt_switchY'] = H_ssb_sy[obs2keep,:]
	NC_save['info_ssb_cnt_switchX'] = lgrid.obsmat_count(NC_save['H_ssb_filt_switchX'])
        NC_save['info_ssb_cnt_switchY'] = lgrid.obsmat_count(NC_save['H_ssb_filt_switchY'])
    '''
    if np.size(FILE2_GEO)!=0:
        lon_grid,lat_grid,geo_cnt,H_geo = geo_obs_mat(lon_obs,lat_obs)
        NC_save['H_geo_filt'] = H_geo[obs2keep,:]
        NC_save['info_geo_cnt'] = geo_cnt
    if np.size(ERR)==0 or ERR=='none':
        if '_dir' in FILE1:
            if MSS is True:
		    #interpolate CNES 2015 MSS model: spline_interp_mss(MISSIONNAME,lat,lon,ERR=False,MSS='cnes15')
                mss15,mss15_err=lgrid.spline_interp_mss(MISSIONNAME,lat_obs[obs2keep],lon_obs[obs2keep],MSS='cnes15')
                NC_save['mean_sea_surface_cnes15'] = mss15
                NC_save['units']['mean_sea_surface_cnes15'] = 'm'
                mss11,mss11_err=lgrid.spline_interp_mss(MISSIONNAME,lat_obs[obs2keep],lon_obs[obs2keep],MSS='cnes11')
                NC_save['mean_sea_surface_cnes11'] = mss11
                NC_save['units']['mean_sea_surface_cnes11'] = 'm'
    crt_data2matrices(METHOD,NC_save,FILE1,FILE2,FILE2_GEO)
    return NC_save


def data2mat(MISSIONNAME,VAR,SOURCE,CYCLES,BAND,OMIT,DIM,GEO,MSS):
    # Purpose: filter product2data files and run function for data2matrices for each cycle within CYCLES array. Only used with Direct measurements.
    DIMND,OMITlab,MODELFILE = dic.file_labels(VAR,BAND,DIM,OMIT)
    Nc = np.shape(CYCLES)[0]
    for ii in np.arange(Nc):
        t1 = time.time()
        CYC= CYCLES[ii]
        # CALL PRODUCT2DATA OUTPUT FILE NAME
        FN = dic.create_filename(MISSIONNAME,'dir',[],'product2data',SOURCE,str(CYC),str(CYC),'na','na','nc') #create_filename(MISSIONNAME,METHOD,OMIT,APP,SOURCE,CYC1,CYC2,BAND,DIM,EXT,ERR=[],VERSION=[])
        print('Pulling: '+FN)
        ds = Dataset(FN)
        print('mean ssha_ku = '+str(np.round(np.nanmean(ds['ssha'][:]),4)))
        pth2_exist = True
        idx,usla_ku,usla_c = lflt.filter_data2obs(MISSIONNAME,'dir',CYC,ds)
        print(np.size(ds['lat'][:]))
        print(np.size(idx))
        print('mean usla_ku = '+str(np.round(np.nanmean(usla_ku),4)))
        fNC = lflt.filtered_dataset('dir',SOURCE,usla_ku,usla_c,ds,idx)
        if pth2_exist == True:
            FILE1,FILE2,FILE2_GEO = dic.filenames_specific(MISSIONNAME,'dir',SOURCE,str(CYC),str(CYC),BAND,DIMND)
            if GEO is False:
                FILE2_GEO=[]
            NC_save = data2matrices(MISSIONNAME,'dir',fNC,FILE1,FILE2,FILE2_GEO,DIM,OMIT,MSS,ERR='none')
        print('Time to create '+FILE1+' [sec]: '+str(time.time()-t1))
        if np.size(FILE2_GEO)!=0:
            print('Time to create '+FILE2_GEO+' [sec]: '+str(time.time()-t1))
    return NC_save

####################################################################################
# ERR != 0 or 'none'
####################################################################################
def poe(ds,DIM,ERR,MISSIONNAME,METHOD,VAR,OMIT,GEO,CYC):
    '''
    Purpose: Pull direct data2vector files (ds) and create new direct measurement observation vector files (data2vector) and design matrices (data2matrix) with added error. Put remade data2vector and data2matrix files into NC_save.
    '''
    dsv=ds.variables
    x_grid,y_grid,z_grid = lgrid.grid_dimensions(DIM,OMIT)
    lon_obs,lat_obs=dsv['lon'][:],dsv['lat'][:]
    if np.size(OMIT) == 0:
        z_obs = []
    else:
        z_obs = dsv[OMIT]
    x_obs,y_obs = ldata.pull_observables(dsv,DIM,ERR,MISSIONNAME,APP='data2matrix')
    if 'smth_swh' in ERR:
        inn_ss = np.where(~np.isnan(x_obs))[0]
        obs2keep_pre,usla_ku,usla_c = lflt.filter_data2obs(MISSIONNAME,'dir',CYC,ds)
        obs2keep = np.intersect1d(obs2keep_pre,inn_ss)
    else:
        obs2keep,usla_ku,usla_c = lflt.filter_data2obs(MISSIONNAME,'dir',CYC,ds)
    if np.size(OMIT) == 0:
        H_ssb,x_scatter,y_scatter,z_scatter = observation_matrix(np.take(x_obs,obs2keep,axis=0),np.take(y_obs,obs2keep,axis=0),z_obs,x_grid,y_grid,z_grid)
    else:
        H_ssb,x_scatter,y_scatter,z_scatter = observation_matrix(np.take(x_obs,obs2keep,axis=0),np.take(y_obs,obs2keep,axis=0),np.take(z_obs,obs2keep,axis=0),x_grid,y_grid,z_grid)
    if GEO is True:
        lon_grid,lat_grid,geo_cnt,H_geo = geo_obs_mat(np.take(lon_obs,obs2keep,axis=0),np.take(lat_obs,obs2keep,axis=0))
    dsk = dsv.keys()
    Nk = np.shape(dsk)[0]
    NC_save = {}
    NC_save['units']={}
    if np.size(ERR)!=0:
        NC_save['swh_'+ERR]=np.take(x_obs,obs2keep,axis=0)
        NC_save[DIM+'_'+ERR]=np.take(y_obs,obs2keep,axis=0)
        vEk,inonan,inan = lflt.observables(METHOD,VAR,'ku',dsv,[],[],[],ERR,MISSIONNAME)
        vEc,inonan,inan = lflt.observables(METHOD,VAR,'c',dsv,[],[],[],ERR,MISSIONNAME)
        NC_save[VAR+'_'+ERR+'_ku'] =np.take(vEk,obs2keep,axis=0)
        NC_save[VAR+'_'+ERR+'_c']= np.take(vEc,obs2keep,axis=0)
        NC_save['units']['swh_'+ERR]='m'
        NC_save['units'][DIM+'_'+ERR]='m/s'
        NC_save['units'][VAR+'_'+ERR+'_ku']= 'm'
        NC_save['units'][VAR+'_'+ERR+'_c']='m'
        if 'smth_swh' in ERR:
            Hr = dsv['swh_rms_ku'][:]
            T = dsv['time'][:]
            sHr,ikp2 = ldata.iono_smooth(Hr,T,11)
            NC_save['swh_rms_'+ERR+'_ku'] = np.take(sHr,obs2keep,axis=0)
            NC_save['units']['swh_rms_'+ERR+'_ku']='m'
            if np.size(inn_ss)!=np.size(np.where(~np.isnan(sHr))[0]):
                raise('smoothing not the same for SWH and SWH RMS')
    for ii in np.arange(Nk):
        if dsk[ii] not in ['info_ssb_cnt','info_geo_cnt','H_ssb_filt','H_geo_filt','units','col_cnt_ssb','col_cnt_geo']:
            NC_save[dsk[ii]] = np.take(dsv[dsk[ii]][:],obs2keep,axis=0)
            NC_save['units'][dsk[ii]] = dsv[dsk[ii]].units
    Ia = dsv['iono_corr_alt_ku'][:]
    T = dsv['time'][:]
    sIa,ikp3 = ldata.iono_smooth(Ia,T,21)
    NC_save['iono_corr_alt_ku_smooth']=np.take(sIa,obs2keep,axis=0)
    NC_save['units']['iono_corr_alt_ku_smooth']='m'
    NC_save['H_ssb_filt'] = H_ssb#[obs2keep,:]
    NC_save['info_ssb_cnt'] = lgrid.obsmat_count(NC_save['H_ssb_filt'])
    if GEO is True:
        NC_save['H_geo_filt'] = H_geo#[obs2keep,:]
        NC_save['info_geo_cnt'] = geo_cnt
    return NC_save

def mat2mat(MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,BAND,OMIT,DIM,ERR,GEO):
    # Purpose: run function for poe for each cycle within CYCLES array. Only used with Direct measurements.
    DIMND,OMITlab,MODELFILE = dic.file_labels(VAR,BAND,DIM,OMIT)
    Nc = np.shape(CYCLES)[0]
    tt = 0
    for ii in np.arange(Nc):
        t1 = time.time()
        CYC= CYCLES[ii]
        FN1,FN,FN2_GEO = dic.filenames_specific(MISSIONNAME,'dir',SOURCE,str(CYC),str(CYC),BAND,DIMND,ERR=[])
        FILE1,FILE2,FILE2_GEO = dic.filenames_specific(MISSIONNAME,'dir',SOURCE,str(CYC),str(CYC),BAND,DIMND,ERR=ERR)
        pth2_exist = path.exists(FN1)
        if pth2_exist == True:
            ds = Dataset(FN1)
            NC_save = poe(ds,DIM,ERR,MISSIONNAME,METHOD,VAR,OMIT,GEO,CYC)
            tt+=1
            if np.size(FILE1) != 0:
                if GEO is False:
                    FILE2_GEO=[]
                crt_data2matrices(METHOD,NC_save,FILE1,FILE2,FILE2_GEO)
        print('Time to create '+FILE1+' [sec]: '+str(time.time()-t1))
        if np.size(FILE2_GEO)!=0:
            print('Time to create '+FILE2_GEO+' [sec]: '+str(time.time()-t1))
    print('Number of files included = '+str(tt)+' out of '+str(Nc)+' cycles')
    if tt==0:
        raise('no files used')
    return NC_save


# --------------------------------------------------------------------------#
#-------------------run: data2obs to obs2lut--------------------------------#

def run_data2matrices(MISSIONNAME,METHOD,SOURCE,CYCLES_in,OMIT,DIM,ERR='none',GEO=False,MSS=False):
    '''
    Purpose: run function for data2mat (direct), mat2mat (direct with error), run_mat2tndm (direct and tandem) and dir2dif (collinear and crossover differences). All functions create step-2 data2vector and data2matrix files
    CYCLES_in = array containing cycle numbers
    '''
    strt = time.time()
    VAR = 'usla' # VAR (input variable) always equal to 'usla' for converting cycle stacks to matrices (data2matrices)
    BAND = 'ku' # BAND always equal to 'ku' for converting cycle stacks to matrices (data2matrices) [  'c' band automatically created ]
    if 'col' in METHOD:
        CYCLES = CYCLES_in[:-1]
    else:
        CYCLES = np.copy(CYCLES_in)
    if np.size(ERR)==0:
        if 'dir' in METHOD:
            NC_save = data2mat(MISSIONNAME,VAR,SOURCE,CYCLES,BAND,OMIT,DIM,GEO,MSS)
        else:
            NC_save = ld2d.dir2dif(MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,BAND,OMIT,DIM,ERR,GEO,MSS)
    else:
        if 'dir' in METHOD:
            if ERR in ['txj1','j1j2','j2j3','j3s6A']:
                print('TANDEM ANALYSIS')
                radius = 950.#1000.#1500.
                NC_save,NC_save2 = ltdm.run_mat2tndm(METHOD,VAR,SOURCE,BAND,OMIT,DIM,radius,ERR,GEO)
            else:
                NC_save = mat2mat(MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,BAND,OMIT,DIM,ERR,GEO)
        else:
            NC_save = ld2d.dir2dif(MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,BAND,OMIT,DIM,ERR,GEO,MSS)
    endt = time.time()
    print('#------Time to make data2matrices: '+str(endt-strt)+'------#')
    return NC_save



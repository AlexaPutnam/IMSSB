#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 19:23:30 2020

@author: alexaputnam

Author: Alexa Angelina Putnam Shilleh
Institution: University of Colorado Boulder
Software: sea state bias model generator (interpolation method)
Thesis: Sea State Bias Model Development and Error Analysis for Pulse-limited Altimetry
Semester/Year: Fall 2021
"""

import numpy as np
from netCDF4 import Dataset
from os import path
import time

import lib_dict_ext as dic
import lib_difference_measurement_generator_ext as ldif
import lib_data_ext as ldata
import lib_geo_gridding_ext as lgrid
import main_data2matrices_ext as md2m


####################################################################################
# Tandem
####################################################################################
def tndm_match(ds1,ds2,radius):
    '''
    Purpose: Run function for sub_tndm_match to find indices of measurements from tandem missions that match both temporally and spatially
    ds1 = netCDF4 altimeter dataset for one cycle (i.e. data2vector) from mission#1
    ds2 = netCDF4 altimeter dataset for one cycle (i.e. data2vector) from mission#2
    radius = search radius to find measurements at the same location from two different missions
    idx1,idx2 = indices of matching measurements for mission#1 and mission#2, respectively
    '''
    passes  = np.arange(1,255)#255)
    pn1 = ds1['pass_number'][:]
    pn2 = ds2['pass_number'][:]
    lat1=ds1['lat'][:]
    lon1=ds1['lon'][:]
    t1=ds1['time'][:]
    lat2=ds2['lat'][:]
    lon2=ds2['lon'][:]
    t2=ds2['time'][:]
    Ndim=np.size(np.shape(pn1))
    if Ndim==1:
        print('alongtrack: dim=1')
        idx1,idx2=sub_tndm_match(pn1,pn2,lat1,lat2,lon1,lon2,t1,t2,passes,radius)
    elif Ndim==2:
        print('difference: dim=2')
        idx1a,idx2a=sub_tndm_match(pn1[:,0],pn2[:,0],lat1[:,0],lat2[:,0],lon1[:,0],lon2[:,0],t1[:,0],t2[:,0],passes,radius)
        idx1b,idx2b=sub_tndm_match(pn1[:,1],pn2[:,1],lat1[:,1],lat2[:,1],lon1[:,1],lon2[:,1],t1[:,1],t2[:,1],passes,radius)
        idx1=np.intersect1d(idx1a,idx1b)
        idx2=np.intersect1d(idx2a,idx2b)
    return idx1,idx2

def sub_tndm_match(pn1,pn2,lat1,lat2,lon1,lon2,t1,t2,passes,radius):
    '''
    Purpose: Find indices of measurements from tandem missions that match both temporally and spatially
    pn1,pn2 = measurement pass numbers for mission#1 and mission#2, respectively
    lat1,lat2= measurement latitudes for mission#1 and mission#2, respectively
    lon1,lon2= measurement longitudes for mission#1 and mission#2, respectively
    t1,t2= measurement timestamps for mission#1 and mission#2, respectively
    passes = an array containing all pass numbers for TX, J1-3 and S6 (passes = [1:254]
    radius = search radius to find measurements at the same location from two different missions
    idx1,idx2 = indices of matching measurements for mission#1 and mission#2, respectively
    '''
    idx1,idx2,dist,dt = [],[],[],[]
    for ii in np.arange(np.shape(passes)[0]):
        ip1 = np.where(pn1==passes[ii])[0]
        ip2 = np.where(pn2==passes[ii])[0]
        if np.size(ip1)>0 and np.size(ip2)>0:
            idx1ii,idx2ii = ldif.tree_dist_col(lat1[ip1],lon1[ip1],lat2[ip2],lon2[ip2],t1[ip1],t2[ip2],radius,'col')
            if np.size(idx1ii)>0 and np.size(idx2ii)>0:
                idp1=ip1[idx1ii]
                idp2=ip2[idx2ii]
                x1,y1,z1,x2,y2,z2,neighbors = ldif.ecef2tree(lat1[idp1],lon1[idp1],lat2[idp2],lon2[idp2],radius)
                distii = lgrid.dist_func(x1,y1,z1,x2,y2,z2)
                dtii=t2[idp2]-t1[idp1]
                idx1 = np.append(idx1,idp1)
                idx2 = np.append(idx2,idp2)
                dist = np.append(dist,distii)
                dt = np.append(dt,dtii)
            else:
                print('no matching tandem measurements for pass '+str(passes[ii]))
        else:
            print('not available -  pass '+str(passes[ii]))
    idx1=idx1.astype(int)
    idx2=idx2.astype(int)
    return idx1,idx2

def tndm_ds(ds1,ds2,DIM,ERR,MN1,MN2,OMIT,idx1,idx2,GEO):
    '''
    Purpose: Create data2matrix library for each mission in a tandem analysis (MN1 and MN2) that contains information from the tandem mission (i.e. MN1=J2 file contains MN2=J3 measurements in addition to MN1 measurements during the tandem period). 
    ds1 = netCDF4 altimeter dataset for one cycle (i.e. data2vector) from mission#1
    ds2 = netCDF4 altimeter dataset for one cycle (i.e. data2vector) from mission#2
    idx1,idx2 = indices of matching measurements for mission#1 and mission#2, respectively
    NC_save1 = MN1 data2matrix library 
    NC_save2 = MN2 data2matrix library 
    '''
    x_grid,y_grid,z_grid = lgrid.grid_dimensions(DIM,OMIT)
    if np.size(OMIT) == 0:
        z_obs1 = []
        z_obs2 = []
    else:
        z_obs1 = np.take(ds1[OMIT][:],idx1,axis=0)
        z_obs2 = np.take(ds2[OMIT][:],idx2,axis=0)
    x_obs1,y_obs1 = ldata.pull_observables(ds1,DIM,ERR,MN1)
    x_obs2,y_obs2 = ldata.pull_observables(ds2,DIM,ERR,MN2)
    H_ssb1,x_scatter,y_scatter,z_scatter = md2m.observation_matrix(np.take(x_obs1,idx1,axis=0),np.take(y_obs1,idx1,axis=0),z_obs1,x_grid,y_grid,z_grid)
    H_ssb2,x_scatter,y_scatter,z_scatter = md2m.observation_matrix(np.take(x_obs2,idx2,axis=0),np.take(y_obs2,idx2,axis=0),z_obs2,x_grid,y_grid,z_grid)

    # Axis switch - old feature
    #y_obs1s2 = ldata.collard_jpl(np.take(ds2['sig0_ku'][:],idx2,axis=0),np.take(x_obs1,idx1,axis=0),MN2)
    #y_obs2s1 = ldata.collard_jpl(np.take(ds1['sig0_ku'][:],idx1,axis=0),np.take(x_obs2,idx2,axis=0),MN1)
    #H_ssb1_sX,x_scatter,y_scatter,z_scatter = md2m.observation_matrix(np.take(x_obs2,idx2,axis=0),np.take(y_obs1,idx1,axis=0),z_obs1,x_grid,y_grid,z_grid)
    #H_ssb2_sX,x_scatter,y_scatter,z_scatter = md2m.observation_matrix(np.take(x_obs1,idx1,axis=0),np.take(y_obs2,idx2,axis=0),z_obs2,x_grid,y_grid,z_grid)

    #H_ssb1_sY,x_scatter,y_scatter,z_scatter = md2m.observation_matrix(np.take(x_obs1,idx1,axis=0),y_obs1s2,z_obs1,x_grid,y_grid,z_grid)
    #H_ssb2_sY,x_scatter,y_scatter,z_scatter = md2m.observation_matrix(np.take(x_obs2,idx2,axis=0),y_obs2s1,z_obs2,x_grid,y_grid,z_grid)

    idVAR1 = [kk.encode() for kk in ds1.keys()]
    idVAR2 = [kk.encode() for kk in ds2.keys()]
    #dsk = list(np.intersect1d(idVAR1,idVAR2))
    dsk = list(np.unique(idVAR1+idVAR2))
    Nk = np.shape(dsk)[0]
    NC_save1 = {}
    NC_save1['units']={}
    NC_save2 = {}
    NC_save2['units']={}
    #NC_save1['wind_speed_alt_sig0_'+MN2]=y_obs1s2
    #NC_save1['units']['wind_speed_alt_sig0_'+MN2]='m/s'
    #NC_save2['wind_speed_alt_sig0_'+MN1]=y_obs2s1
    #NC_save2['units']['wind_speed_alt_sig0_'+MN1]='m/s'
    for ii in np.arange(Nk):
        if dsk[ii] not in ['info_ssb_cnt','info_geo_cnt','H_ssb_filt','H_geo_filt','units','col_cnt_ssb','col_cnt_geo']:
          if dsk[ii] in idVAR1:
            NC_save1[dsk[ii]] = np.take(ds1[dsk[ii]],idx1,axis=0)#ds[dsk[ii]][:][idx]
            NC_save1['units'][dsk[ii]] = ds1[dsk[ii]].units
          if dsk[ii] in idVAR2:
            NC_save1[dsk[ii]+'_'+MN2] = np.take(ds2[dsk[ii]],idx2,axis=0)#ds[dsk[ii]][:][idx]
            NC_save1['units'][dsk[ii]+'_'+MN2] = ds2[dsk[ii]].units
          if dsk[ii] in idVAR2:
            NC_save2[dsk[ii]] = np.take(ds2[dsk[ii]],idx2,axis=0)#ds[dsk[ii]][:][idx]
            NC_save2['units'][dsk[ii]] = ds2[dsk[ii]].units
          if dsk[ii] in idVAR1:
            NC_save2[dsk[ii]+'_'+MN1] = np.take(ds1[dsk[ii]],idx1,axis=0)#ds[dsk[ii]][:][idx]
            NC_save2['units'][dsk[ii]+'_'+MN1] = ds1[dsk[ii]].units
    NC_save1['H_ssb_filt'] = H_ssb1
    NC_save1['info_ssb_cnt'] = lgrid.obsmat_count(NC_save1['H_ssb_filt'])
    #NC_save1['H_ssb_filt_switchX'] = H_ssb1_sX
    #NC_save1['info_ssb_cnt_switchX'] = lgrid.obsmat_count(NC_save1['H_ssb_filt_switchX'])
    #NC_save1['H_ssb_filt_switchY'] = H_ssb1_sY
    #NC_save1['info_ssb_cnt_switchY'] = lgrid.obsmat_count(NC_save1['H_ssb_filt_switchY'])

    if GEO is True:
        lon_grid,lat_grid,geo_cnt1,H_geo1 = md2m.geo_obs_mat(np.take(ds1['lon'][:],idx1,axis=0),np.take(ds1['lat'][:],idx1,axis=0))
        NC_save1['H_geo_filt'] = H_geo1
        NC_save1['info_geo_cnt'] = geo_cnt1
        lon_grid,lat_grid,geo_cnt2,H_geo2 = md2m.geo_obs_mat(np.take(ds2['lon'][:],idx2,axis=0),np.take(ds2['lat'][:],idx2,axis=0))
        NC_save2['H_geo_filt'] = H_geo2
        NC_save2['info_geo_cnt'] = geo_cnt2

    NC_save2['H_ssb_filt'] = H_ssb2
    NC_save2['info_ssb_cnt'] = lgrid.obsmat_count(NC_save2['H_ssb_filt'])
    #NC_save2['H_ssb_filt_switchX'] = H_ssb2_sX
    #NC_save2['info_ssb_cnt_switchX'] = lgrid.obsmat_count(NC_save2['H_ssb_filt_switchX'])
    #NC_save2['H_ssb_filt_switchY'] = H_ssb2_sY
    #NC_save2['info_ssb_cnt_switchY'] = lgrid.obsmat_count(NC_save2['H_ssb_filt_switchY'])
    return NC_save1,NC_save2


def run_mat2tndm(METHOD,VAR,SOURCE,BAND,OMIT,DIM,radius,ERR,GEO):
    # Purpose: Run function for tndm_match and tndm_d that converts the MN1 and MN2 data2matrix libraries (NC1 and NC2) into data2matrix netCDF4 files from step-2.
    t1 = time.time()
    DIMND,OMITlab,MODELFILE = dic.file_labels(VAR,BAND,DIM,OMIT)
    MN1,CYCs1,MN2,CYCs2 = dic.tandem(ERR,METHOD)
    Nc = np.shape(CYCs1)[0]
    tt = 0
    for ii in np.arange(Nc):
        CYC1= CYCs1[ii]
        CYC2= CYCs2[ii]
        FN11,FN1,FN2_GEO1 = dic.filenames_specific(MN1,METHOD,SOURCE,str(CYC1),str(CYC1),BAND,DIMND,ERR=[])
        FILE11,FILE21,FILE2_GEO1 = dic.filenames_specific(MN1,METHOD,SOURCE,str(CYC1),str(CYC1),BAND,DIMND,ERR=ERR)
        FN12,FN2,FN2_GEO2 = dic.filenames_specific(MN2,METHOD,SOURCE,str(CYC2),str(CYC2),BAND,DIMND,ERR=[])
        FILE12,FILE22,FILE2_GEO2 = dic.filenames_specific(MN2,METHOD,SOURCE,str(CYC2),str(CYC2),BAND,DIMND,ERR=ERR)
        if GEO is False:
            FILE2_GEO1=[]
            FILE2_GEO2=[]
        pth2_exist1 = path.exists(FN11)
        pth2_exist2 = path.exists(FN12)
        print('radius = '+str(radius))
        if pth2_exist1==True and pth2_exist2==True:
            ds1 = Dataset(FN11).variables
            ds2 = Dataset(FN12).variables
            if 'dir' in METHOD:
                idx1,idx2 = tndm_match(ds1,ds2,radius)
            else:
                idx1,idx2 = np.arange(np.shape(ds1['lat'][:])[0]),np.arange(np.shape(ds2['lat'][:])[0])
            if np.size(idx1)!=np.size(idx2):
                raise('indices do not match in run_mat2tndm')
            NC1,NC2=tndm_ds(ds1,ds2,DIM,ERR,MN1,MN2,OMIT,idx1,idx2,GEO)
            tt+=1
            if np.size(FILE12) != 0:
                md2m.crt_data2matrices(METHOD,NC1,FILE11,FILE21,FILE2_GEO1)
                md2m.crt_data2matrices(METHOD,NC2,FILE12,FILE22,FILE2_GEO2)
        t2 = time.time()
        print('Time to create tandem file (i.e.'+FILE11+'): '+str(t2-t1))
    print('Number of files included = '+str(tt)+' out of '+str(Nc)+' cycles')
    if tt==0:
        raise('no files used')
    return NC1,NC2












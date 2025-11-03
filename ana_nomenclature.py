#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22-November-2022

@author: alexaputnam
"""

from netCDF4 import Dataset
import numpy as np
from scipy import interpolate
import os
import time


def files_in_dir(mypath):
    from os import walk
    files = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        files.extend(filenames)
        break
    return files


def rads2imssb_sa(dsin,idx):
    varG = ['time','pass_number','lat','lon','alt','rad_wet_tropo_corr','model_wet_tropo_corr',
    'model_dry_tropo_corr','pole_tide','ocean_tide_non_eq','internal_tide_hret','dac',
    'ocean_tide_fes','ocean_tide_got','load_tide_fes','load_tide_got','mean_sea_surface','mean_sea_surface_cnes15',
    'solid_earth_tide','wind_speed_alt']
    varE = [['dist_coast','topo_srtm30plus'],['rad_distance_to_land','bathymetry']]
    varK = [['range_ka','swh_ka','swh_rms_ka','sig0_ka','sig0_rms_ka','range_rms_ka','ssb_tran2019','iono_gim','iono_gim','ssha_ka'],['range_ku','swh_ku','swh_rms_ku','sig0_ku','sig0_rms_ku','range_rms_ku','sea_state_bias_ku','iono_corr_alt_ku','iono_corr_gim_ku','ssha']]
    varC = [['range_ka','swh_ka','swh_rms_ka','sig0_ka','sig0_rms_ka','range_rms_ka','ssb_tran2019'],['range_c','swh_c','swh_rms_c','sig0_c','sig0_rms_c','range_rms_c','sea_state_bias_c']]
    varIN = varG+varE[0]+varK[0]+varC[0]
    varOUT = varG+varE[1]+varK[1]+varC[1]
    Nv = np.shape(varIN)[0]
    LIB={}
    for ii in np.arange(Nv):
        LIB[varOUT[ii]] = {}
        if 'off_nadir_angle2' in varIN[ii]:
            LIB[varOUT[ii]] = np.sqrt(np.copy(dsin[varIN[ii]][:][idx]))
        elif 'dist_coast' in varIN[ii]:
            LIB[varOUT[ii]] = 1000.*(np.copy(dsin[varIN[ii]][:][idx]))
        else:
            LIB[varOUT[ii]] = np.copy(dsin[varIN[ii]][:][idx])
    return LIB    



def new_rads_files(MISSIONNAME):
    # MISSIONNAME,OpenOcean = 'sw',True
    fku = 13.575e9
    fc = 5.3e9
    k_iku = (fc**2)/((fku**2)-(fc**2))
    pth_old = '/home/aputnam/IMSSB/imssb_software/output/'+MISSIONNAME+'/product2data/rads/radsout/'
    pth_new = '/home/aputnam/IMSSB/imssb_software/output/'+MISSIONNAME+'/product2data/rads/' #'/home/aputnam/RADS_func/rads_data/tracks/'+MISSIONNAME+'/a/'
    FNCs = files_in_dir(pth_old)
    Nf = np.shape(FNCs)[0]
    for jj in np.arange(Nf):
        # check for altimetry data netcdf file
        if FNCs[jj][:2]==MISSIONNAME:
            #input file
            dsin = Dataset(pth_old+FNCs[jj])
            #print(dsin.variables.keys())
            if 'swh_ka' in dsin.variables.keys():
                swhK = dsin['swh_ka'][:]
                if np.size(swhK)>2:
                    print(FNCs[jj])
                    #output file
                    idx = np.where((swhK>=-1)&(swhK<=20)&(np.abs(dsin['ssha_ka'][:])<=2)&(np.abs(dsin['dist_coast'][:])<=50))[0]
                    if np.size(idx)>0:
                        LIB = rads2imssb_sa(dsin,idx)
        dim = np.size(LIB['time'])
        if dim>0:
            VARall = list(LIB.keys())
            Nv = np.shape(VARall)[0]
            print('size v. shape of data: '+str(dim)+' v. '+str(np.shape(LIB['time'])))
            FILESV =pth_new+FNCs[jj]
            root_grp = Dataset(FILESV, 'w', format='NETCDF4')
            root_grp.description = MISSIONNAME+" data product to save for product2data segment of IMSSB "
            root_grp.history = "Author: Alexa Angelina Putnam Shilleh. Institute: University of Colorado Boulder" + time.ctime(time.time())
            print('size of cycle file (time): '+str(dim))
            root_grp.createDimension('time', dim)
            for ii in np.arange(Nv):
                print('size of '+VARall[ii]+': '+str(np.size(LIB[VARall[ii]])))
                xi = root_grp.createVariable(VARall[ii], 'f8', ('time'))
                xi[:] = LIB[VARall[ii]]
            root_grp.close()
            itrk = 0





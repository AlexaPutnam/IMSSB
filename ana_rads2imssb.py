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

def lon360_to_lon180(lon_old):
    igt = np.where(lon_old>180)[0] #!!! used to be (prior to may 5 2021) lon_old>=180
    if np.size(igt)!=0:
        lon_new = np.mod((lon_old+180.),360.)-180.
    else:
        lon_new = np.copy(lon_old)
    return lon_new

def lon180_to_lon360(lon_old):
    igt = np.where(lon_old<0)[0]
    if np.size(igt)!=0:
        lon_new = np.mod(lon_old,360.)
    else:
        lon_new = np.copy(lon_old)
    return lon_new

def dir_in_dir(mypath):
    from os import walk
    dirs = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        dirs.extend(dirnames)
        break
    return dirs

def files_in_dir(mypath):
    from os import walk
    files = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        files.extend(filenames)
        break
    return files

def pull_SSB(MN,cyc=[]):
    pth = 'SSB_models/'
    if MN=='tx':
        if cyc<236:
            FNk = 'tx_jnt_ssb_matrix2model_jpl_48_c_100_ku_u10_2d_LatWgt.nc'
            FNc = 'tx_jnt_ssb_matrix2model_jpl_48_c_100_c_u10_2d_LatWgt.nc'
        else:
            FNk = 'tx_jnt_ssb_matrix2model_jpl_280_c_364_ku_u10_2d_LatWgt.nc'
            FNc = 'tx_jnt_ssb_matrix2model_jpl_280_c_364_c_u10_2d_LatWgt.nc'
    elif MN=='j1':
        FNk = 'j1_jnt_ssb_matrix2model_jpl_111_c_146_ku_u10_2d_LatWgt.nc'
        FNc = 'j1_jnt_ssb_matrix2model_jpl_111_c_146_c_u10_2d_LatWgt.nc'
    elif MN=='j2':
        FNk = 'j2_jnt_ssb_matrix2model_jpl_56_c_91_ku_u10_2d_LatWgt.nc'
        FNc = 'j2_jnt_ssb_matrix2model_jpl_56_c_91_c_u10_2d_LatWgt.nc'
    elif MN=='j3':
        FNk = 'j3_jnt_ssb_matrix2model_jpl_70_c_105_ku_u10_2d_LatWgt.nc'
        FNc = 'j3_jnt_ssb_matrix2model_jpl_70_c_105_c_u10_2d_LatWgt.nc'
    elif MN=='6a':
        FNk = 's6A_jnt_ssb_matrix2model_jpl_5_c_41_ku_u10_2d_LatWgt.nc'
        FNc = 's6A_jnt_ssb_matrix2model_jpl_5_c_41_c_u10_2d_LatWgt.nc'
    else:
        raise('NO SSB models for '+MN)
    return Dataset(pth+FNk),Dataset(pth+FNc)

def pull_SSB_ERR(MN):
    pth = 'SSB_error_models/'
    if MN=='tx':
        FN = 'tx_jnt_ssb_evaluation_error_ku_c.nc'
        fillK = 0.008
        fillC = 0.009
    else:
        FN = 'j2_jnt_ssb_evaluation_error_ku_c.nc'
        fillK = 0.022
        fillC = 0.024
    ds=Dataset(pth+FN)
    Bk_err = np.sqrt(ds['ssb_ku'][:])
    Bc_err = np.sqrt(ds['ssb_c'][:])
    #Bk_err[np.isnan(Bk_err)]=fillK
    #Bc_err[np.isnan(Bc_err)]=fillC
    return Bk_err,Bc_err


def parameter_grid(x_grid,y_grid,z_grid):
    '''
    Purpose: Return coordinate matrices (mx,my,mz) and collapsed coordinate matrices (fx,fy,fz) from coordinate vectors (x_grid,y_grid,z_grid).
    '''
    Nz = np.size(z_grid)
    if Nz !=0:
        my,mx,mz= np.meshgrid(y_grid,x_grid,z_grid)
        fx = mx.flatten('F')
        fy = my.flatten('F')
        fz = mz.flatten('F')
    elif Nz ==0:
        my,mx= np.meshgrid(y_grid,x_grid)
        fx = mx.flatten('F')
        fy = my.flatten('F')
        mz = []
        fz = []
    return mx,my,mz,fx,fy,fz
'''
def bilinear_interpolation_sp_griddata(x_obs,y_obs,x_grid,y_grid,B):
    # Purpose: uses least square to determine the value of all points within a grid/model
    mx,my,mz,x_scatter,y_scatter,z_scatter = parameter_grid(x_grid,y_grid,[])
    #print(np.size(np.isnan(B.flatten('C'))))
    if np.size(np.isnan(B.flatten('C')))==0:
       x_ma = mx
       y_ma = my
       B_ma = B
    else:
       B_mask = np.ma.masked_invalid(B)
       x_ma = mx[~B_mask.mask]
       y_ma = my[~B_mask.mask]
       B_ma = B_mask[~B_mask.mask]
    B_obs_gd =interpolate.griddata((y_ma, x_ma), np.ravel(B_ma,order='F'),(y_obs,x_obs),method='linear') #B_ma.ravel()
    return B_obs_gd
'''


def bilinear_interpolation_sp_griddata(x_obs,y_obs,x_grid,y_grid,B,ERR=False):
    # Purpose: uses least square to determine the value of all points within a grid/model
    mx,my,mz,x_scatter,y_scatter,z_scatter = parameter_grid(x_grid,y_grid,[])
    #print(np.size(np.isnan(B.flatten('C'))))
    if np.size(np.isnan(B.flatten('C')))==0:
       x_ma = mx
       y_ma = my
       B_ma = B
    else:
       B_mask = np.ma.masked_invalid(B)
       x_ma = mx[~B_mask.mask]
       y_ma = my[~B_mask.mask]
       B_ma = B_mask[~B_mask.mask]
    if ERR ==False:
        B_obs_gd =interpolate.griddata((y_ma, x_ma), np.ravel(B_ma,order='F'),(y_obs,x_obs),method='linear') #B_ma.ravel()
    else:
        B_obs_gd = np.empty(np.shape(x_obs))*np.nan
        B_obs_gd_int =interpolate.griddata((y_ma, x_ma), np.ravel(B_ma,order='F'),(y_obs,x_obs),method='linear') #B_ma.ravel()
        N = np.size(B_obs_gd_int)
        for ii in np.arange(N):
            B_obs_gd[ii] = np.random.normal(0, B_obs_gd_int[ii], 1)[0]
    return B_obs_gd


def imel_iono_avg(vI,vT,sW,CUT=True):
    #Purpose: ionosphere smoothing function (similar to Imel’s 1994 paper)
    #vI: array of ionosphere correction values
    #vT: array of time stamps corresponding to vI
    #sW: integer smoothing window size (i.e. 11)
    #CUT: True or False to remove datapoints that do not uphold to smoothing criteria
    #vIs_out: array of smoothed ionosphere corrections
    #iW = indices corresponding to data points that pass the smoothing criteria
    N = np.shape(vI)[0]
    res = np.round(np.median(np.diff(vT)),3)
    if res>1.2:
        print(res)
        print('resolution is too large (should be ~1-s), not '+str(res)+'-s')
        vIs_out = np.copy(vI)
        iW = np.arange(N)
    else:
        buffH = (res*(np.float(sW)-1.0))+0.5
        buffL = (res*(np.float(sW)-1.0))-0.5
        if sW>1:
            dT = np.subtract(vT[sW-1:], vT[:-sW+1]) #np.asarray(list(map(operator.sub, vT[sW-1:], vT[:-sW+1])))
            iW = np.where((dT[:]<buffH)&(dT[:]>=buffL))[0]
            vIs = np.convolve(vI, np.ones(sW), 'valid') / sW
        else:
            vIs = np.copy(vI)
            iW = np.arange(N)
        if CUT is True:
            vIs_out = np.take(vIs,iW)
        elif CUT is False:
            vIs_out=np.empty(np.shape(vI))*np.nan
            vIs_out[(sW-1)/2:(-sW+1)/2]=np.copy(vIs)
            iW = iW+(sW-1)/2
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



def list2arr(lDIR):
    N = np.shape(lDIR)[0]
    aDIR = np.empty(N)*np.nan
    FILL = 999999
    for ii in np.arange(N):
        # check for altimetry data directory
        if lDIR[ii][0]=='c':
            cyc = int(lDIR[ii][1:])
            aDIR[ii] = cyc
        else:
            aDIR[ii] = FILL
    isrt = np.argsort(aDIR)
    sDIR = np.copy(aDIR[isrt])
    fDIR = np.copy(sDIR[sDIR!=FILL])
    Nd = np.shape(fDIR)[0]
    nDIR = []
    for ii in np.arange(Nd):
        nDIR.append('c'+"{0:03}".format(int(fDIR[ii])))
    return nDIR

def random_error(RMS):
    N = np.size(RMS)
    random_error = np.empty(N)*np.nan
    for ii in np.arange(N):
        random_error[ii] = np.random.normal(0, RMS[ii], 1)[0]
    return random_error


def data_filter(dsin,VAR,MN):
    if VAR=='lat':
        vmin,vmax = -90.0,90.0
    elif VAR=='lon':
        vmin,vmax = -180.0,180.0
    elif VAR=='range_ku':
        if MN=='sw':
            vmin,vmax = 800e3, 900e3
        elif MN in ['6a','j3','j2','j1','tx','pn']:
            vmin,vmax = 1300e3, 1400e3
    elif VAR=='range_c':
        if MN=='sw':
            vmin,vmax = 800e3, 900e3
        elif MN in ['6a','j3','j2','j1','tx','pn']:
            vmin,vmax = 1300e3, 1400e3
    elif VAR=='wind_speed_alt':
        vmin,vmax = -1.0,30.0
    elif VAR=='wind_speed_rad':
        vmin,vmax = 0.0,30.0
    elif VAR=='swh_ku':
        vmin,vmax = 0.0,20.0 
    elif VAR=='swh_rms_ku':
        vmin,vmax = 0.0,2.0 #c-band: 0-3
    elif VAR=='sig0_ku':
        vmin,vmax = 0.0,40.0
    elif VAR=='sig0_rms_ku':
        vmin,vmax = 0.0,1.5
    elif VAR=='range_numval':
        vmin,vmax = 10.0, 21.5
    elif VAR=='range_numval_c':
        vmin,vmax = 10.0, 21.5
    elif VAR=='range_rms_ku':
        vmin,vmax = 0.00, 0.25
    elif VAR=='range_rms_c':
        vmin,vmax = 0.0, 0.5
    elif VAR=='wet_tropo_rad':
        vmin,vmax=-0.6, 0.0
    elif VAR=='dry_tropo_ecmwf':
        vmin,vmax=-2.5,-1.9
    elif VAR=='iono_gim':
        vmin,vmax=-0.4, 0.04
    elif VAR=='off_nadir_angle2_wf_ku':
        if MN=='sw':
            vmin,vmax = -0.15, 0.15
        elif MN in ['6a','j3','j2','j1','tx','pn']:
            vmin,vmax = -0.20, 0.64
    elif VAR=='tide_pole':
        vmin,vmax=-0.15, 0.15
    elif VAR=='tide_internal':
        vmin,vmax=-0.2, 0.2
    elif VAR=='tide_ocean_fes14':
        vmin,vmax=-5,5
    elif VAR=='tide_ocean_got410':
        vmin,vmax=-5,5
    elif VAR=='tide_load_fes14':
        vmin,vmax=-0.5,0.5
    elif VAR=='tide_load_got410':
        vmin,vmax=-0.5,0.5
    elif VAR=='tide_solid':
        vmin,vmax=-1.0,1.0
    elif 'inv_bar' in VAR:
        vmin,vmax=-2.0, 2.0
    elif 'topo_' in VAR:
        vmin,vmax=-200000,-300
    if VAR=='lon':
        lonvar=lon360_to_lon180(dsin[VAR][:])
        idx = np.where((lonvar>=vmin)&(lonvar<=vmax))[0]
    else:
        idx = np.where((dsin[VAR][:]>=vmin)&(dsin[VAR][:]<=vmax))[0]
    return idx

def loop_intersect1d(ds1,FLAGS):
    N = np.shape(FLAGS)[0]
    for ii in np.arange(N):
        if ii==0:
            idx = np.where(ds1[FLAGS[ii][0]][:]==FLAGS[ii][1])[0]
        else:
            idxi = np.where(ds1[FLAGS[ii][0]][:]==FLAGS[ii][1])[0]
            idx = np.intersect1d(idx,idxi)
    return idx

def swot_flags(dsin):
    FLAGS = [['surface_class',0], ['surface_type_rad1',0], ['surface_type_rad2',0],['flag_manoeuvre',0],
             ['qual_alt_rain_ice',0], ['qual_rad1_rain_ice',0], ['qual_rad2_rain_ice',0],['qual_attitude',0]]#, ['qual_range',0],['qual_swh',0],['qual_sig0',0], QUALITY FLAGS REMOVED FOR NOW
    idx = loop_intersect1d(dsin,FLAGS)
    return idx

def ra_flags(dsin):
    FLAGS = [['surface_class',0], ['surface_type_rad',0],['flag_manoeuvre',0],['qual_alt_rain_ice',0],['qual_attitude',0]]#, ['qual_range',0],['qual_swh',0],['qual_sig0',0], QUALITY FLAGS REMOVED FOR NOW
    idx = loop_intersect1d(dsin,FLAGS)
    return idx

def pull_rads(dsin,MN,OpenOcean):
    #OpenOcean=True
    #MN='sw'
    #dsin = Dataset('/Users/alexaputnam/SWOT/swp0008c200.nc')
    passN = dsin.pass_number #pass_number
    cycN = dsin.cycle_number #cycle_number
    eqLon = dsin.equator_longitude #equator_longitude
    if OpenOcean==True:
        filt_open_ocean = ['lat','lon','range_ku','wind_speed_alt','swh_ku','swh_rms_ku','sig0_ku','sig0_rms_ku',
                    'range_numval','range_numval_c','range_rms_ku','range_rms_c','wet_tropo_rad',
                    'dry_tropo_ecmwf','iono_gim','off_nadir_angle2_wf_ku','tide_pole','tide_ocean_fes14',
                    'tide_ocean_got410','tide_load_fes14','tide_load_got410','tide_solid','inv_bar_mog2d','topo_srtm30plus']
    else:
        filt_open_ocean = ['lat','lon','range_ku','wind_speed_alt','swh_ku','swh_rms_ku','sig0_ku','sig0_rms_ku',
                    'range_numval','range_numval_c','range_rms_ku','range_rms_c','wet_tropo_rad',
                    'dry_tropo_ecmwf','iono_gim','off_nadir_angle2_wf_ku','tide_pole','tide_solid','inv_bar_mog2d']
    
    var_info = ['time','lat','lon','dist_coast']
    var_meas = ['range_ku','range_c','range_ku_mle3','swh_ku','swh_c','swh_ku_mle3','sig0_ku','sig0_c','sig0_ku_mle3']
    var_rms = ['range_rms_ku','range_rms_c','range_rms_ku_mle3','swh_rms_ku','swh_rms_c','swh_rms_ku_mle3','sig0_rms_ku','sig0_rms_c','sig0_rms_ku_mle3']
    var_numv = ['range_numval_c','range_numval_ku_mle3']
    var_wind = ['wind_speed_alt','wind_speed_rad']
    varIN_dacIN = [['inv_bar_static','inv_bar_mog2d'],['inv_bar','dac']]#[['inv_bar_mog2d_mean','inv_bar_static','inv_bar_mog2d'],['dac_mean','inv_bar','dac']]
    varIN_meas = [['range_numval','off_nadir_angle2_wf_ku','off_nadir_angle2_wf_rms_ku'],['range_numval_ku','off_nadir_angle_wf_ku','off_nadir_angle_wf_rms_ku']]
    varIN_atm = [['wet_tropo_rad','wet_tropo_ecmwf','dry_tropo_ecmwf','iono_alt','iono_alt_mle3','iono_gim'],['rad_wet_tropo_corr','model_wet_tropo_corr','model_dry_tropo_corr','iono_corr_alt_ku','iono_corr_alt_ku_mle3','iono_corr_gim_ku']]
    varIN_tide = [['tide_pole','tide_non_equil','tide_internal','tide_ocean_fes14','tide_ocean_got410','tide_load_fes14','tide_load_got410','tide_solid'],['pole_tide','ocean_tide_non_eq','internal_tide_hret','ocean_tide_fes','ocean_tide_got','load_tide_fes','load_tide_got','solid_earth_tide']]
    varIN_ssb = [['ssb_cls','ssb_cls_c','ssb_cls_mle3'],['sea_state_bias_ku','sea_state_bias_c','sea_state_bias_ku_mle3']]
    if MN=='sw':
        varIN_geo = [['mss_dtu21','mss_cnescls15','rad1_dist_coast','rad2_dist_coast','topo_srtm30plus','surface_type_rad1','surface_type_rad2'],['mean_sea_surface','mean_sea_surface_cnes15','rad_distance_to_land','rad2_distance_to_land','bathymetry','rad_surf_type','rad2_surf_type']]
        varIN_MD = [['alt_gdrf'],['alt']]
    else:
        varIN_geo = [['mss_dtu21','mss_cnescls15','rad_dist_coast','topo_srtm30plus','surface_type_rad'],['mean_sea_surface','mean_sea_surface_cnes15','rad_distance_to_land','bathymetry','rad_surf_type']]
        varIN_MD = [['alt_gdrf'],['alt']]
    varIN = var_info+var_meas+var_rms+var_numv+var_wind+varIN_dacIN[0]+varIN_meas[0]+varIN_atm[0]+varIN_tide[0]+varIN_ssb[0]+varIN_geo[0]+varIN_MD[0]
    varOUT = var_info+var_meas+var_rms+var_numv+var_wind+varIN_dacIN[1]+varIN_meas[1]+varIN_atm[1]+varIN_tide[1]+varIN_ssb[1]+varIN_geo[1]+varIN_MD[1]
 
    Nf = np.shape(filt_open_ocean)[0]
    idx_var = np.arange(np.size(dsin['time'][:]))
    for ii in np.arange(Nf):
        idxi = data_filter(dsin,filt_open_ocean[ii],MN)
        #print('size of '+filt_open_ocean[ii]+' filter: '+str(np.size(idxi)))
        if np.size(idxi)>0:
            idx_var = np.intersect1d(idxi,idx_var)
    print('size of valid data array after variable filter: '+str(np.size(idx_var)))
    if MN=='sw':
        idx_flg = swot_flags(dsin)
    else:
        idx_flg = ra_flags(dsin)
    idx = np.intersect1d(idx_var,idx_flg)
    print('size of valid data array after variable and flag filter: '+str(np.size(idx)))
    LIB={}
    Nv = np.shape(varIN)[0] #,'pass_number'
    for ii in np.arange(Nv):
        LIB[varOUT[ii]] = {}
        if 'off_nadir_angle2' in varIN[ii]:
            LIB[varOUT[ii]] = np.sqrt(np.copy(dsin[varIN[ii]][idx]))
        elif 'dist_' in varIN[ii]:
            LIB[varOUT[ii]] = (np.copy(dsin[varIN[ii]][idx]))*1000.0
        else:
            LIB[varOUT[ii]] = np.copy(dsin[varIN[ii]][idx])
    LIB['pass_number'] = np.ones(np.size(idx))*passN
    LIB['cycle_number'] = np.ones(np.size(idx))*cycN
    LIB['equator_longitude'] = np.ones(np.size(idx))*eqLon
    LIB['ssha'] = LIB['alt']-LIB['range_ku']-LIB['model_dry_tropo_corr']-LIB['rad_wet_tropo_corr']-LIB['iono_corr_alt_ku']-LIB['dac']-LIB['solid_earth_tide']-LIB['ocean_tide_fes']-LIB['load_tide_fes']-LIB['pole_tide']-LIB['sea_state_bias_ku']-LIB['mean_sea_surface']
    LIB['ssha_mle3'] = LIB['alt']-LIB['range_ku_mle3']-LIB['model_dry_tropo_corr']-LIB['rad_wet_tropo_corr']-LIB['iono_corr_alt_ku_mle3']-LIB['dac']-LIB['solid_earth_tide']-LIB['ocean_tide_fes']-LIB['load_tide_fes']-LIB['pole_tide']-LIB['sea_state_bias_ku_mle3']-LIB['mean_sea_surface']
    return LIB
    



def new_rads_files(MISSIONNAME,OpenOcean,SEG='b'):
    # MISSIONNAME,OpenOcean = 'sw',True
    fku = 13.575e9
    fc = 5.3e9
    k_iku = (fc**2)/((fku**2)-(fc**2))
    pth_old = '/srv/data/rads/tracks/'+MISSIONNAME+'/'+SEG+'/'
    pth_new = '/home/aputnam/IMSSB/imssb_software/output/'+MISSIONNAME+'/product2data/rads/' #'/home/aputnam/RADS_func/rads_data/tracks/'+MISSIONNAME+'/a/'
    #input files
    DIRS = dir_in_dir(pth_old)
    DIRS = list2arr(DIRS)
    #DIRS = DIRS[63:]#DIRS[bad:]#J1=(bad=35, iono res > 1.2)# TX=(bad=19(c005), iono res > 1.2),6a=(bad=17 (c031), segment is too short)
    print('Old directories: ')
    print(DIRS)
    Nd = np.shape(DIRS)[0]
    for ii in np.arange(Nd):
        # check for altimetry data directory
        if DIRS[ii][0]=='c':
            itrk = 0
            cyc = int(DIRS[ii][1:])
            FNCs = files_in_dir(pth_old+DIRS[ii])
            Nf = np.shape(FNCs)[0]
            for jj in np.arange(Nf):
                # check for altimetry data netcdf file
                if FNCs[jj][:2]==MISSIONNAME:
                    #input file
                    dsin = Dataset(pth_old+DIRS[ii]+'/'+FNCs[jj])
                    #print(dsin.variables.keys())
                    if 'swh_ku' in dsin.variables.keys():
                        swhK = dsin['swh_ku'][:]
                        if np.size(swhK)>2:
                            print(FNCs[jj])
                            #output file
                            if itrk==0:
                                LIB = pull_rads(dsin,MISSIONNAME,OpenOcean)
                                itrk=1
                            else:
                                LIBi = pull_rads(dsin,MISSIONNAME,OpenOcean)
                                VAR = list(LIBi.keys())
                                Nl = np.shape(VAR)[0]
                                for vv in np.arange(Nl):
                                    LIB[VAR[vv]] = np.asarray(np.hstack((LIB[VAR[vv]],LIBi[VAR[vv]])))
            dim = np.size(LIB['time'])
            if dim>0:
                VARall = list(LIBi.keys())
                Nv = np.shape(VARall)[0]
                print('size v. shape of data: '+str(dim)+' v. '+str(np.shape(LIB['time'])))
                FILESV =pth_new+'/'+MISSIONNAME+'_dir_product2data_rads_'+str(cyc)+'_c_'+str(cyc)+'_na_na.nc'
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


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
import shutil

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

def new_rads_files(MISSIONNAME):
    fku = 13.575e9
    fc = 5.3e9
    k_iku = (fc**2)/((fku**2)-(fc**2))
    pth_old = '/srv/data/rads/tracks/'+MISSIONNAME+'/a/'
    #pth_old = '/home/aputnam/RADS_func/rads_data/'+MISSIONNAME+'/a_old/'
    pth_new = '/home/aputnam/RADS_func/rads_data/tracks/'+MISSIONNAME+'/a/'
    #input files
    DIRS = dir_in_dir(pth_old)
    DIRS = list2arr(DIRS)
    #DIRS = DIRS[63:]#DIRS[bad:]#J1=(bad=35, iono res > 1.2)# TX=(bad=19(c005), iono res > 1.2),6a=(bad=17 (c031), segment is too short)
    DIR_new = dir_in_dir(pth_new)
    DIR_new = list2arr(DIR_new)
    print('Old directories: ')
    print(DIRS)
    print('New directories: ')
    print(DIR_new)
    if np.shape(DIRS)[0]!=np.shape(DIR_new)[0]:
        DIRS =np.setxor1d(DIRS,DIR_new)
        print('Updated old directories: ')
        print(DIRS)
    Nd = np.shape(DIRS)[0]
    for ii in np.arange(Nd):
        # check for altimetry data directory
        if DIRS[ii][0]=='c':
          #if int(DIRS[ii][1:])>=236: # TEST ONLY FOR TOPEX
            cyc = int(DIRS[ii][1:])
            FNCs = files_in_dir(pth_old+DIRS[ii])
            Nf = np.shape(FNCs)[0]
            if DIRS[ii] not in DIR_new:
                os.mkdir(pth_new+DIRS[ii]+'/')
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
                            dsout = Dataset(pth_new+DIRS[ii]+'/'+FNCs[jj], "w", format="NETCDF4")
                            # Create new SSB
                            FBk,FBc = pull_SSB(MISSIONNAME,cyc=cyc)
                            ws_grid = FBk['u10'][:]
                            swh_grid = FBk['swh'][:]
                            Bk = FBk['ssb'][:]
                            Bc = FBc['ssb'][:]
                            Bk_err,Bc_err = pull_SSB_ERR(MISSIONNAME)
                            if MISSIONNAME=='tx' and cyc>=236:
                               Ical = FBk['rs_zs_bias'][:]*(-0.01216)
                            else:
                                Ical = FBk['iono_calib_bias'][:]*(-0.01216)
                            ws = dsin['wind_speed_alt'][:]
                            swhK = dsin['swh_ku'][:]
                            rK = dsin['range_ku'][:]
                            rC = dsin['range_c'][:]
                            t_obs = dsin['time'][:]
                            # evaluate SSB
                            ssb_new = bilinear_interpolation_sp_griddata(swhK,ws,swh_grid,ws_grid,Bk,ERR=False)
                            err_ssb_new = bilinear_interpolation_sp_griddata(swhK,ws,swh_grid,ws_grid,Bk_err,ERR=True)
                            ssb_c = bilinear_interpolation_sp_griddata(swhK,ws,swh_grid,ws_grid,Bc,ERR=False)
                            err_ssb_c = bilinear_interpolation_sp_griddata(swhK,ws,swh_grid,ws_grid,Bc_err,ERR=True)
                            # create new SSB with added error
                            ssb_newE = ssb_new+err_ssb_new# SSB with added random error
                            ssb_cE = ssb_c+err_ssb_c# SSB with added random error
                            # create new dual-frequency ionospehre corrections
                            IkuA = k_iku*((rK+ssb_new)-(rC+ssb_c)+Ical)
                            IkuAE = k_iku*((rK+ssb_newE)-(rC+ssb_cE)+Ical)# Ionosphere with random SSB error
                            if np.size(IkuA[~np.isnan(IkuA)])>100:
                                I_new,ikp3 = iono_smooth(IkuA,t_obs,21)
                                I_newE,ikp3E = iono_smooth(IkuAE,t_obs,21)# smoothed Ionosphere with random SSB error
                            else:
                                I_new = np.copy(IkuA)
                                I_newE = np.copy(IkuAE)
                            # add error to CLS model
                            ssb_cls = dsin['ssb_cls'][:]
                            ssb_clsKE = ssb_cls+err_ssb_new# SSB CLS with added random error
                            if 'ssb_cls_c' in dsin.variables.keys():
                                ssb_cls_c = dsin['ssb_cls_c'][:]
                                ssb_clsCE = ssb_cls_c+err_ssb_c# SSB CLS with added random error
                                IkuA_cls = k_iku*((rK+ssb_cls)-(rC+ssb_cls_c))
                                IkuAE_cls = k_iku*((rK+ssb_clsKE)-(rC+ssb_clsCE))# Ionosphere with random SSB error
                                if np.size(IkuA_cls[~np.isnan(IkuA_cls)])>100:
                                    I_new_cls,ikp3_cls = iono_smooth(IkuA_cls,t_obs,21)
                                    I_newE_cls,ikp3E_cls = iono_smooth(IkuAE_cls,t_obs,21)# smoothed Ionosphere with random SSB error
                                else:
                                    I_new_cls = np.copy(IkuA_cls)
                                    I_newE_cls = np.copy(IkuAE_cls)
                            else:
                                ssb_cls_c = np.zeros(np.size(rK))*np.nan
                                ssb_clsCE = np.zeros(np.size(rK))*np.nan
                                if 'iono_doris' in dsin.variables.keys():
                                    IkuA_cls = dsin['iono_doris'][:]
                                    IkuAE_cls = dsin['iono_doris'][:]#-(k_iku*err_ssb_c)
                                elif 'iono_alt' in dsin.variables.keys():
                                    IkuA_cls = dsin['iono_alt'][:]
                                    IkuAE_cls = dsin['iono_alt'][:]
                                else:
                                    IkuA_cls = np.zeros(np.size(rK))*np.nan
                                    IkuAE_cls = np.zeros(np.size(rK))*np.nan
                                I_new_cls = np.copy(IkuA_cls)
                                I_newE_cls = np.copy(IkuAE_cls)
                            # total added error to ionosphere from SSB and range
                            Rku_err = random_error(dsin['range_rms_ku'][:])
                            if 'range_rms_c' in dsin.variables.keys():
                                Rc_err = random_error(dsin['range_rms_c'][:])
                            else:
                                Rc_err = np.zeros(np.size(Rku_err))*np.nan
                            if 'ssb_cls_c' in dsin.variables.keys():
                                Iku_err = k_iku*((rK+Rku_err+ssb_clsKE)-(rC+Rc_err+ssb_clsCE))
                                if np.size(Iku_err[~np.isnan(Iku_err)])>100:
                                    Iku_err_s,ikp3E_cls = iono_smooth(Iku_err,t_obs,21)
                                else:
                                    Iku_err_s = np.copy(Iku_err)
                            else:
                                if 'iono_doris' in dsin.variables.keys():
                                    Iku_err = dsin['iono_doris'][:]#+(k_iku*Rku_err)-(k_iku*Rc_err)-(k_iku*err_ssb_c)
                                elif 'iono_alt' in dsin.variables.keys():
                                    Iku_err = dsin['iono_alt'][:]
                                else:
                                    Iku_err = np.zeros(np.size(rK))*np.nan
                                Iku_err_s = np.copy(Iku_err)
                            #Copy dimensions
                            for dname, the_dim in dsin.dimensions.iteritems():
                                #print(dname, len(the_dim))
                                dsout.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)
                            # Copy variables
                            ionS=0
                            for v_name, varin in dsin.variables.iteritems():
                                outVar = dsout.createVariable(v_name, varin.datatype, varin.dimensions)
                                #print(varin.datatype)
                                # Copy variable attributes
                                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                outVar[:] = varin[:]
                                if v_name=='ssb_cls':
                                    outVar = dsout.createVariable('ssb_alexa', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(ssb_new[:])#ssb_new[:]
                                    #
                                    outVar = dsout.createVariable('ssb_alexa_err', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(ssb_newE[:])
                                    #
                                    outVar = dsout.createVariable('err_ssb_ku', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(err_ssb_new[:])#ssb_new[:]
                                    #
                                    outVar = dsout.createVariable('ssb_cls_err', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(ssb_clsKE[:])
                                    #if v_name=='ssb_cls':
                                    outVar = dsout.createVariable('ssb_alexa_c', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(ssb_c[:])#ssb_c[:]
                                    #
                                    outVar = dsout.createVariable('ssb_alexa_c_err', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(ssb_cE[:])
                                    #
                                    outVar = dsout.createVariable('err_ssb_c', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(err_ssb_c[:])#ssb_new[:]
                                    #
                                    outVar = dsout.createVariable('ssb_cls_c_err', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(ssb_clsCE[:])
                                if v_name in ['iono_alt_smooth','iono_gim','iono_nic09'] and ionS==0:
                                    ionS=1
                                    outVar = dsout.createVariable('iono_alexa_smooth', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(I_new[:])#I_new[:]
                                    #
                                    outVar = dsout.createVariable('iono_alexa_smooth_err', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(I_newE[:])
                                    #
                                    outVar = dsout.createVariable('iono_cls_smooth', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(I_new_cls[:])#I_new[:]
                                    #
                                    outVar = dsout.createVariable('iono_cls_smooth_err', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(I_newE_cls[:])
                                    #
                                    outVar = dsout.createVariable('iono_cls_smooth_total_err', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(Iku_err_s[:])
                                    #
                                    outVar = dsout.createVariable('iono_alexa', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(IkuA[:])#IkuA[:]
                                    #
                                    outVar = dsout.createVariable('iono_alexa_err', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(IkuAE[:])
                                    #
                                    outVar = dsout.createVariable('iono_cls', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(IkuA_cls[:])#IkuA[:]
                                    #
                                    outVar = dsout.createVariable('iono_cls_err', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(IkuAE_cls[:])
                                    #
                                    outVar = dsout.createVariable('iono_cls_total_err', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(Iku_err[:])
                                if v_name=='range_rms_ku':
                                    outVar = dsout.createVariable('err_range_ku', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(Rku_err[:])
                                    #if v_name=='range_rms_c':
                                    #
                                    outVar = dsout.createVariable('err_range_c', varin.datatype, varin.dimensions)
                                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                                    outVar[:] = np.ma.masked_invalid(Rc_err[:])
                            # close the output file
                            dsout.close()
                    else:
                        print('DIRECT COPY: '+FNCs[jj])
                        shutil.copy(pth_old+DIRS[ii]+'/'+FNCs[jj],pth_new+DIRS[ii]+'/')

                    




#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:07:42 2020

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
from os import path

import lib_difference_measurement_generator_ext as ldif
import lib_dict_ext as dic
import main_data2matrices_ext as md2m


###############################################
###############################################
# indexing functions
def asc_des_passes(ds):
    '''
    Purpose: provide the indices of the ascending and descending measurements given a netCDF4 file of the altimeter data (ds)
    iA,iD =  indices of the ascending (A) and descending (D) measurements
    lat_A,lon_A,tA = latitude, longitude and time for ascending measurements
    lat_D,lon_D,tD = latitude, longitude and time for descending measurements
    '''
    if 'pass_direction_flag' in ds.keys():
        pn = ds['pass_direction_flag'][:].astype(int)
    else:
        pn = ds['pass_number'][:]
    PASSES01 = pn%2
    iA = np.where(PASSES01==1)[0] # ascending passes are odd numbered
    iD = np.where(PASSES01==0)[0] # descending passes are even numbered
    lat_A,lon_A = np.take(ds['lat'][:],iA),np.take(ds['lon'][:],iA)
    lat_D,lon_D = np.take(ds['lat'][:],iD),np.take(ds['lon'][:],iD)
    tA,tD = np.take(ds['time'][:],iA),np.take(ds['time'][:],iD)
    return iA,iD,lat_A,lon_A,lat_D,lon_D,tA,tD

def xov_indices_1cyc(ds1,radius,INTERP):
    '''
    Purpose: find crossover points within a single cycle
    ds1 = netCDF4 altimeter dataset for one cycle (i.e. data2vector)
    radius = search radius to find crossover points
    INTERP = True or False to interpolate to intersecting point
    iA_xov_c1,iD_xov_c1 =  indices of the ascending (A) and descending (D) measurements that match to provide crossover differences
    p_asc_c1,p_des_c1 =  partials for each point of intersection for each crossover difference between A and D
    lonX_c1,latX_c1 = geographic point of intersection for each crossover difference between A and D
    '''
    # separate data into ascending and descending passes
    iA_pass_c1,iD_pass_c1,lat_A_c1,lon_A_c1,lat_D_c1,lon_D_c1,t_asc_c1,t_des_c1 = asc_des_passes(ds1)
    # find crossovers within same cycle
    iA_sub_c1,iD_sub_c1,p_asc_c1,p_des_c1,lonX_c1,latX_c1 = ldif.tree_dist(lat_A_c1,lon_A_c1,lat_D_c1,lon_D_c1,t_asc_c1,t_des_c1,radius,'xov',INTERP)
    # save indices
    iA_xov_c1 = iA_pass_c1[iA_sub_c1]
    iD_xov_c1 = iD_pass_c1[iD_sub_c1]
    return iA_xov_c1,iD_xov_c1,p_asc_c1,p_des_c1,lonX_c1,latX_c1

def xov_indices_2cyc(ds1,ds2,radius,INTERP):
    '''
    Purpose: same as  xov_indices_1cyc, but finds crossover points between two cycles rather than within one
    * _c1a2d = crossover with ascending=cyc1 and descending=cyc2
    * _c2a1d = crossover with ascending=cyc2 and descending=cyc1
    '''
    # separate data into ascending and descending passes
    iA_pass_c1,iD_pass_c1,lat_A_c1,lon_A_c1,lat_D_c1,lon_D_c1,t_asc_c1,t_des_c1 = asc_des_passes(ds1)
    iA_pass_c2,iD_pass_c2,lat_A_c2,lon_A_c2,lat_D_c2,lon_D_c2,t_asc_c2,t_des_c2 = asc_des_passes(ds2)
    # find crossovers between two seperate cycles
    iA_sub_c1a2d,iD_sub_c1a2d,pA_c1a2d,pD_c1a2d,lonX_c1a2d,latX_c1a2d = ldif.tree_dist(lat_A_c1,lon_A_c1,lat_D_c2,lon_D_c2,t_asc_c1,t_des_c2,radius,'xov',INTERP)
    iA_sub_c2a1d,iD_sub_c2a1d,pA_c2a1d,pD_c2a1d,lonX_c2a1d,latX_c2a1d = ldif.tree_dist(lat_A_c2,lon_A_c2,lat_D_c1,lon_D_c1,t_asc_c2,t_des_c1,radius,'xov',INTERP)
    # save indices for crossver with ascending=cyc1 and descending-cyc2
    iA_xov_c1a2d = np.take(iA_pass_c1,iA_sub_c1a2d)
    iD_xov_c1a2d = np.take(iD_pass_c2,iD_sub_c1a2d)
    # save indices for crossver with ascending=cyc2 and descending-cyc1
    iA_xov_c2a1d = np.take(iA_pass_c2,iA_sub_c2a1d)
    iD_xov_c2a1d = np.take(iD_pass_c1,iD_sub_c2a1d)
    return iA_xov_c1a2d,iD_xov_c1a2d,iA_xov_c2a1d,iD_xov_c2a1d,pA_c1a2d,pD_c1a2d,lonX_c1a2d,latX_c1a2d,pA_c2a1d,pD_c2a1d,lonX_c2a1d,latX_c2a1d

def pairs2xov(varA,varD,pA,pD):
    '''
    Purpose: evaluates variable (var) at lonX and latX for ascending and descending measurements. Crossover values interpolated at intersecting point.
    varA = [Nx2] array containing ascending measurement pairs surrounding a crossover point
    varD = [Nx2] array containing descending measurement pairs surrounding a crossover point
    pA = [Nx2] array containing partials that weight ascending measurement pairs to a crossover point
    pD =[Nx2] array containing partials that weight descending measurement pairs to a crossover point
    varAx = interpolated value of var at crossover point for ascending measurements
    varDx = interpolated value of var at crossover point for descending measurements
    '''
    varAx = np.add(np.multiply(varA[:,0],pA[:,0]),np.multiply(varA[:,1],pA[:,1]))
    varDx = np.add(np.multiply(varD[:,0],pD[:,0]),np.multiply(varD[:,1],pD[:,1]))
    return varAx,varDx

#################################
# Main function
#################################
def xovgen(CYC1,CYC2,ds1,ds2,radius=7000.,INTERP=True):
    # Purpose: produce crossover data2vector and data2matrix files given two cycle files of direct measurements
    ''' # Check
    MISSIONNAME,CYCLES,TYPE,radius,INTERP = 'j2',np.arange(1,37),'model',7000,True # xovgen('j2',np.arange(5,37),'model',radius=7000.,INTERP=True)
    MISSIONNAME,CYCLES,TYPE,radius,INTERP = 'j2',np.arange(2,3),'model',2500,False # xovgen('j2',np.arange(5,37),'model',radius=1000.,INTERP=False)
    '''
    t1 = time.time()
    if np.size(ds2)==0:
        idVAR = [kk.encode() for kk in ds1.keys()]
    else:
        idVAR1 = [kk.encode() for kk in ds1.keys()]
        idVAR2 = [kk.encode() for kk in ds2.keys()] #.encode()
        idVAR = list(np.intersect1d(idVAR1,idVAR2))
    print(idVAR)
    Nv = np.shape(idVAR)[0]
    NC = {}
    NC['units']={}
    for jj in np.arange(Nv):
        var_i = idVAR[jj].decode("utf-8")
        if var_i not in ['info_ssb_cnt','info_geo_cnt','H_ssb_filt','H_geo_filt','units','col_cnt_ssb','col_cnt_geo']:
            print(var_i)
            NC[var_i]=[]
            NC[var_i+'_m0']=[]
            NC[var_i+'_m1']=[]
            NC['units'][var_i] = []
            NC['units'][var_i+'_m0'] = []
            NC['units'][var_i+'_m1'] = []
    if INTERP == True:
        NC['partials_m0']=[]
        NC['partials_m1']=[]
        NC['units']['partials_m0'] = 'none'
        NC['units']['partials_m1'] = 'non'
        NC['lat']=[]
        NC['lon']=[]
        #NC['mss_cls15']=[]
        #NC['mss_cls11']=[]
        NC['units']['lat'] = 'degrees_north'
        NC['units']['lon'] = 'degrees_east'
        #NC['units']['mss_cls15'] = 'm'
        #NC['units']['mss_cls11'] = 'm'
    # find crossover points within a single cycle
    iA_xov_c1,iD_xov_c1,pA_c1,pD_c1,lonX_c1,latX_c1 = xov_indices_1cyc(ds1,radius,INTERP)
    print('Time to find crossover indices for cycle '+str(CYC1)+': '+str(time.time()-t1))
    Nidx = np.size(iA_xov_c1)
    # find crossover points between two cycles
    if np.size(ds2)!=0:
        iA_xov_c1a2d,iD_xov_c1a2d,iA_xov_c2a1d,iD_xov_c2a1d,pA_c1a2d,pD_c1a2d,lonX_c1a2d,latX_c1a2d,pA_c2a1d,pD_c2a1d,lonX_c2a1d,latX_c2a1d = xov_indices_2cyc(ds1,ds2,radius,INTERP)
        print('Time to find crossover indices for cycle '+str(CYC2)+': '+str(time.time()-t1))
    '''
    iA_xov_c1 # c1
    iD_xov_c1 # c1
    iA_xov_c1a2d # c1
    iD_xov_c1a2d # c2
    iA_xov_c2a1d # c2
    iD_xov_c2a1d # c1
    '''
    # create crossover dataset for all variables of interest
    if Nidx!=0:
        if np.size(ds2)!=0:
            pvarA10 = np.hstack((pA_c1[:,0],pA_c1a2d[:,0]))
            pvarA0 = np.hstack((pvarA10,pA_c2a1d[:,0]))
            pvarD10 = np.hstack((pD_c1[:,0],pD_c1a2d[:,0]))
            pvarD0 = np.hstack((pvarD10,pD_c2a1d[:,0]))

            pvarA11 = np.hstack((pA_c1[:,1],pA_c1a2d[:,1]))
            pvarA1 = np.hstack((pvarA11,pA_c2a1d[:,1]))
            pvarD11 = np.hstack((pD_c1[:,1],pD_c1a2d[:,1]))
            pvarD1 = np.hstack((pvarD11,pD_c2a1d[:,1]))
            
            latX10 = np.hstack((latX_c1,latX_c1a2d))
            latX = np.hstack((latX10,latX_c2a1d))
            lonX10 = np.hstack((lonX_c1,lonX_c1a2d))
            lonX = np.hstack((lonX10,lonX_c2a1d))
        else:
            pvarA0 = pA_c1[:,0]
            pvarD0 = pD_c1[:,0]
            pvarA1 = pA_c1[:,1]
            pvarD1 = pD_c1[:,1]
            latX = latX_c1
            lonX = lonX_c1
        # save variables into dictionary
        NC['partials_m0'] = np.vstack((pvarA0,pvarD0)).T
        NC['partials_m1'] = np.vstack((pvarA1,pvarD1)).T
        NC['lat'] = np.vstack((latX,latX)).T
        NC['lon'] = np.vstack((lonX,lonX)).T
        for jj in np.arange(Nv):
            var_i = idVAR[jj].decode("utf-8")
            if var_i not in ['info_ssb_cnt','info_geo_cnt','H_ssb_filt','H_geo_filt','units','col_cnt_ssb','col_cnt_geo','partials']:
                # var[xov pts] within a single cycles
                vrA_c1jj = np.take(ds1[var_i][:],iA_xov_c1)
                vrD_c1jj = np.take(ds1[var_i][:],iD_xov_c1)
                # var[xov pts] within between two cycles
                if np.size(ds2)!=0:
                    # ascending passes from first cycle, descending passes from second cycle
                    vrA_c1a2djj = np.take(ds1[var_i][:],iA_xov_c1a2d)
                    vrD_c1a2djj = np.take(ds2[var_i][:],iD_xov_c1a2d)
                    # ascending passes from second cycle, descending passes from first cycle
                    vrA_c2a1djj = np.take(ds2[var_i][:],iA_xov_c2a1d)
                    vrD_c2a1djj = np.take(ds1[var_i][:],iD_xov_c2a1d)
                # specify crossover approach 
                if INTERP == True: # linear interpolation approach
                    # this approach requires creating new crossover points using the partials from the linear interpoaltion
                    vrA_c1i,vrD_c1i = pairs2xov(vrA_c1jj,vrD_c1jj,pA_c1,pD_c1) #jj=25, plt.plot(latX_c1-vrA_c1,'.'), jj=9, plt.hist(lonX_c1-vrD_c1,bins=30) 
                    if np.size(ds2)!=0:
                        vrA_c1a2di,vrD_c1a2di = pairs2xov(vrA_c1a2djj,vrD_c1a2djj,pA_c1a2d,pD_c1a2d) #plt.hist(lonX_c1a2d-vrA_c1a2d,bins=30) 
                        vrA_c2a1di,vrD_c2a1di = pairs2xov(vrA_c2a1djj,vrD_c2a1djj,pA_c2a1d,pD_c2a1d) #plt.hist(lonX_c2a1d-vrA_c2a1d,bins=30) 
                    vrA_c1,vrD_c1 = vrA_c1jj,vrD_c1jj
                    if np.size(ds2)!=0:
                        vrA_c1a2d,vrD_c1a2d = vrA_c1a2djj,vrD_c1a2djj
                        vrA_c2a1d,vrD_c2a1d = vrA_c2a1djj,vrD_c2a1djj
                else: # nearest neighbor approach
                    vrA_c1,vrD_c1 = vrA_c1jj,vrD_c1jj
                    if np.size(ds2)!=0:
                        vrA_c1a2d,vrD_c1a2d = vrA_c1a2djj,vrD_c1a2djj
                        vrA_c2a1d,vrD_c2a1d = vrA_c2a1djj,vrD_c2a1djj
                # filter variables
                if np.size(ds2)!=0:
                    if INTERP == True:
                        varA10 = np.hstack((vrA_c1[:,0],vrA_c1a2d[:,0]))
                        varA0 = np.hstack((varA10,vrA_c2a1d[:,0]))
                        varD10 = np.hstack((vrD_c1[:,0],vrD_c1a2d[:,0]))
                        varD0 = np.hstack((varD10,vrD_c2a1d[:,0]))
                        
                        varA11 = np.hstack((vrA_c1[:,1],vrA_c1a2d[:,1]))
                        varA1 = np.hstack((varA11,vrA_c2a1d[:,1]))
                        varD11 = np.hstack((vrD_c1[:,1],vrD_c1a2d[:,1]))
                        varD1 = np.hstack((varD11,vrD_c2a1d[:,1]))
                    
                        varA1i = np.hstack((vrA_c1i,vrA_c1a2di))
                        varAi = np.hstack((varA1i,vrA_c2a1di))
                        varD1i = np.hstack((vrD_c1i,vrD_c1a2di))
                        varDi = np.hstack((varD1i,vrD_c2a1di))
                    else:
                        varA1 = np.hstack((vrA_c1,vrA_c1a2d))
                        varA = np.hstack((varA1,vrA_c2a1d))
                        varD1 = np.hstack((vrD_c1,vrD_c1a2d))
                        varD = np.hstack((varD1,vrD_c2a1d))
                else:
                    if INTERP == True:
                        varA0 = vrA_c1[:,0]
                        varD0 = vrD_c1[:,0]
                        varA1 = vrA_c1[:,1]
                        varD1 = vrD_c1[:,1]
                    
                        varAi = vrA_c1i
                        varDi = vrD_c1i
                    else:
                        varA = vrA_c1
                        varD = vrD_c1
                # save variables into dictionary
                if INTERP == True:
                    vstk = np.vstack((varAi,varDi)).T
                    vstk0 = np.vstack((varA0,varD0)).T
                    vstk1 = np.vstack((varA1,varD1)).T
                    #if var_i not in ['lat','lon']:
                    NC[var_i] = vstk
                    NC[var_i+'_m0'] = vstk0
                    NC[var_i+'_m1'] = vstk1
                    if var_i not in ['pass_number','equator_longitude','partials']:
                        #if var_i not in ['lat','lon']:
                        NC['units'][var_i] = ds1[var_i].units.encode()
                        NC['units'][var_i+'_m0'] = ds1[var_i].units.encode()
                        NC['units'][var_i+'_m1'] = ds1[var_i].units.encode()
                    else:
                        NC['units'][var_i] = 'none'
                        NC['units'][var_i+'_m0'] = 'none'
                        NC['units'][var_i+'_m1'] = 'none'
                else:
                    vstk = np.vstack((varA,varD)).T
                    NC[var_i] = vstk
                    if var_i not in ['pass_number','equator_longitude','partials']:
                        NC['units'][var_i] = ds1[var_i].units.encode()
                    else:
                        NC['units'][var_i] = 'none'
    if np.size(ds2)!=0:
        if iA_xov_c1.shape[0]+iA_xov_c1a2d.shape[0]+iA_xov_c2a1d.shape[0] != vstk.shape[0]:
            raise('indexing problem in xogen function')
    else:
        if iA_xov_c1.shape[0]!= vstk.shape[0]:
            raise('indexing problem in xogen function')
    # save new crossover dataset
    print('Time to find crossover indices and assemble datasets for cycles '+str(CYC1)+' and '+str(CYC2)+': '+str(time.time()-t1))
    return NC
###############################################
###############################################
def colgen(CYC1,CYC2,ds1,ds2,radius=1000.,INTERP=False):
    # Purpose: produce collinear data2vector and data2matrix files given two cycle files of direct measurements
    # find crossover indices between two files of direct measurements
    tt1 = time.time()
    if 'pass_direction_flag' in ds1.keys():
        pn1 = ds1['pass_direction_flag'][:].astype(int)
        pn2 = ds2['pass_direction_flag'][:].astype(int)
    else:
        pn1 = ds1['pass_number'][:]
        pn2 = ds2['pass_number'][:]
    
    pn1_01 = pn1%2
    pn2_01 = pn2%2
    
    iasc1 = np.where(pn1_01==1)[0] # ascending passes are odd numbered
    ides1 = np.where(pn1_01==0)[0] # descending passes are even numbered
    iasc2 = np.where(pn2_01==1)[0] # ascending passes are odd numbered
    ides2 = np.where(pn2_01==0)[0] # descending passes are even numbered

    lat1,lon1 = ds1['lat'][:],ds1['lon'][:]
    lat2,lon2 = ds2['lat'][:],ds2['lon'][:]
    t1,t2 = ds1['time'][:],ds2['time'][:]
    idx1_12A,idx2_12A,tt,tt,tt,tt = ldif.tree_dist(np.take(lat1,iasc1),np.take(lon1,iasc1),np.take(lat2,iasc2),np.take(lon2,iasc2),np.take(t1,iasc1),np.take(t2,iasc2),radius,'col',INTERP)
    idx1_12D,idx2_12D,tt,tt,tt,tt = ldif.tree_dist(np.take(lat1,ides1),np.take(lon1,ides1),np.take(lat2,ides2),np.take(lon2,ides2),np.take(t1,ides1),np.take(t2,ides2),radius,'col',INTERP)

    print(np.size(idx1_12A))
    print(np.size(idx1_12D))
    print(np.size(idx2_12A))
    print(np.size(idx2_12D))
    if np.size(idx1_12A)+np.size(idx1_12D)+np.size(idx2_12A)+np.size(idx2_12D)!=0:
        idx1_12 = np.hstack((iasc1[np.asarray(idx1_12A)],ides1[np.asarray(idx1_12D)]))
        idx2_12 = np.hstack((iasc2[np.asarray(idx2_12A)],ides2[np.asarray(idx2_12D)]))
        print('mean swh ds1 '+str(np.mean(ds1['swh_ku'][:][idx1_12])))
        print('mean swh ds2 '+str(np.mean(ds2['swh_ku'][:][idx2_12])))
        print('Time to find colliniear '+str(np.shape(idx1_12))+' indices for cycles '+str(CYC1)+' and '+str(CYC2)+': '+str(time.time()-tt1))
        # create dataset
        idVAR1 = [kk for kk in ds1.keys()]
        idVAR2 = [kk for kk in ds2.keys()]
        idVAR = list(np.intersect1d(idVAR1,idVAR2))
        print(idVAR)
        Nv = np.shape(idVAR)[0]
        NC = {}
        NC['units']={}
        for jj in np.arange(Nv):
            var_i = idVAR[jj]
            if var_i not in ['info_ssb_cnt','info_geo_cnt','H_ssb_filt','H_geo_filt','units','col_cnt_ssb','col_cnt_geo']:
                NC[var_i]=[]
                NC['units'][var_i] = []
        for jj in np.arange(Nv):
            var_i = idVAR[jj]
            if var_i not in ['info_ssb_cnt','info_geo_cnt','H_ssb_filt','H_geo_filt','units','col_cnt_ssb','col_cnt_geo']:
                NC[var_i]=np.vstack((ds1[var_i][:][idx1_12],ds2[var_i][:][idx2_12])).T
                if var_i not in ['pass_number','equator_longitude']:
                    NC['units'][var_i] = ds1[var_i].units.encode()
                else:
                    NC['units'][var_i] = 'none'
    else:
        NC=[]
    return NC

###############################################
###############################################
def dir2dif(MISSIONNAME,METHOD,VAR,SOURCE,CYCLES,BAND,OMIT,DIM,ERR,GEO,MSS):
    # Purpose: final run function to run and save xogen and colgen
    t1 = time.time()
    DIMND,OMITlab,MODELFILE = dic.file_labels(VAR,BAND,DIM,OMIT)
    Nc = np.shape(CYCLES)[0]-1
    for ii in np.arange(Nc):
            CYC1= CYCLES[ii]
            CYC2 = int(CYC1)+1
            FN1,tt,tt = dic.filenames_specific(MISSIONNAME,'dir',SOURCE,str(CYC1),str(CYC1),BAND,DIMND,ERR=ERR)
            FN2,tt,tt = dic.filenames_specific(MISSIONNAME,'dir',SOURCE,str(CYC2),str(CYC2),BAND,DIMND,ERR=ERR)
            pth1_exist = path.exists(FN1)
            pth2_exist = path.exists(FN2)
            if pth1_exist == True:
                print('1st file exists: '+FN1)
                print('2nd file exists: '+FN2)
                ds1 = Dataset(FN1)
                dsv1 = ds1.variables
                # CALL PRODUCT2DATA OUTPUT FILE NAME
                if 'xov' in METHOD:
                    if pth2_exist == True:
                        ds2 = Dataset(FN2)
                        dsv2 = ds2.variables
                    else:
                        dsv2=[]
                        print(FN2+' does not exist, so only single cycle of crossovers used for '+str(CYC1))
                    NC = xovgen(CYC1,CYC2,dsv1,dsv2,radius=7000.,INTERP=True)
                elif 'col' in METHOD:
                    if pth2_exist == True:
                        ds2 = Dataset(FN2)
                        dsv2 = ds2.variables
                        if MISSIONNAME in ['sw','s6A','s6Ah']:
                            radius = 3000.#1000.
                        else:
                            radius = 1000.
                        NC = colgen(CYC1,CYC2,dsv1,dsv2,radius=radius,INTERP=False)
                    else:
                        print(FN2+' does not exist, so no collinear differences for '+str(CYC1)+' with '+str(int(CYC2)))
                else:
                    raise('METHOD must be xov or col')
                FILE1,FILE2,FILE2_GEO = dic.filenames_specific(MISSIONNAME,METHOD,SOURCE,str(CYC1),str(CYC1),BAND,DIMND,ERR=ERR)
                if GEO is False:
                    FILE2_GEO=[]
                if np.size(NC)!=0:
                    if 'xov' in METHOD:
                        NC_save = md2m.data2matrices(MISSIONNAME,METHOD,NC,FILE1,FILE2,FILE2_GEO,DIM,OMIT,MSS,ERR=ERR)
                    elif 'col' in METHOD:
                        if pth2_exist == True:
			                #np.save('/ALT1/usr/aputnam/jpl_topex_ssb/ssb_modeling/iSSB_functions/save_colinear.npy',NC)
                            NC_save = md2m.data2matrices(MISSIONNAME,METHOD,NC,FILE1,FILE2,FILE2_GEO,DIM,OMIT,MSS,ERR=ERR)
                        else:
                            print('No collinear measurements for '+str(CYC1)+' and '+str(CYC2))
                    t2 = time.time()
                    print('Time to create the along-track cycle '+str(CYC1)+' file from JPL/topex_gdrf: '+str(t2-t1))
                else:
                    print('No collinear measurements for '+str(CYC1)+' and '+str(CYC2))
            else:
                print('1st file DOES NOT exist: '+FN1)
                print('2nd file DOES NOT exist: '+FN2)
                print('No difference measurements for '+str(CYC1))
    return NC_save
###############################################
###############################################

 

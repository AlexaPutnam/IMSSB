#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:53:25 2020

@author: alexaputnam

Author: Alexa Angelina Putnam Shilleh
Institution: University of Colorado Boulder
Software: sea state bias model generator (interpolation method)
Thesis: Sea State Bias Model Development and Error Analysis for Pulse-limited Altimetry
Semester/Year: Fall 2021
"""
import numpy as np
import itertools
import time
from scipy import spatial

import lib_geo_gridding_ext as lgg #!!!!

###############################################
###############################################
# mathematical functions
    
def linear_fit(x,y):
    '''
    Purpose: linear fit given independent variable (x) and dependent variable (y)
    x = independent variable array
    y = dependent variable array
    ce = fitting coefficients
    y_est = estimated y values using the fitting coefficients and x
    '''
    Nl2 = np.shape(x)[0]
    H = np.ones((Nl2,2))
    H[:,1] = x
    ce = np.linalg.inv(H.T.dot(H)).dot(H.T.dot(y))
    y_est = ce[0]+ce[1]*x
    return ce,y_est

def linear_interpolation_dist(lat0,lon0,lat1,lon1,latX,lonX):
    '''
    Purpose: determine partials linear interpolation given geographic coordinates (lon [-180 to 180] and lat [-90 to 90]) 
    lat0,lon0 = geographic coordinates of measurement 0
    lat1,lon1 = geographic coordinates of measurement 1
    latX,lonX = geographic coordinates of intersecting point, X, between 0 and 1
    px = [2x1] array contains weights (partials) from X to 0 and X to 1.
    dX0,dX1,d01 = distance in [m] from X to 0, X to 1 and 0 to 1
    '''
    x0,y0,z0 = lgg.lla2ecef(lat0,lon0)
    x1,y1,z1 = lgg.lla2ecef(lat1,lon1)
    xX,yX,zX = lgg.lla2ecef(latX,lonX)
    dX0 = lgg.dist_func(x0,y0,z0,xX,yX,zX)
    dX1 = lgg.dist_func(xX,yX,zX,x1,y1,z1)
    d01 = lgg.dist_func(x0,y0,z0,x1,y1,z1)
    px_0 = np.divide(dX1,d01)
    px_1 = np.divide(dX0,d01)
    px = np.asarray([px_0,px_1])
    return px,dX0,dX1,d01

def lla_linear_interp(latA,lonA,latD,lonD):
    '''
    Purpose: determine the partials for linear interpolation given ascending and descending geographic coordinates
    latA,lonA = geographic coordinates for ascending measurements, A
    latD,lonD = geographic coordinates for descending measurements, D
    lonX,latX = geographic coordinates of intersecting point, X, between A and D
    pxA,pxD = weights (or partials) between A and X and D and X
    '''
    # lonA,latA,lonD,latD=lon_asc[iA_arr],lat_asc[iA_arr],lon_des[iD_arr],lat_des[iD_arr]
    # pxA,pxD,lonX,latX = lla_linear_interp(latA,lonA,latD,lonD)
    # Find line for ascending and descending pass around a single crossover point
    # y = mx+b
    lonD_new = lgg.lon360_to_lon180(lonD)
    lonA_new = lgg.lon360_to_lon180(lonA)

    ceD,latD_est = linear_fit(lonD_new,latD)
    ceA,latA_est = linear_fit(lonA_new,latA)
    # Determine the coordinates at the intersection point (assume a straight line since they're close enough)
    lonXew = (ceD[0]-ceA[0])/(ceA[1]-ceD[1])
    latX = ceA[0]+(ceA[1]*lonXew)
    #latXD = ceD[0]+(ceD[1]*lonXew)
    # Determine partials for linear interpolation
    pxA,dX0A,dX1A,d01A = linear_interpolation_dist(latA[0],lonA_new[0],latA[1],lonA_new[1],latX,lonXew)
    pxD,dX0D,dX1D,d01D = linear_interpolation_dist(latD[0],lonD_new[0],latD[1],lonD_new[1],latX,lonXew) 
    lonX = lgg.lon180_to_lon360(lonXew)
    return pxA,pxD,lonX,latX 

def ecef2tree(latA,lonA,latD,lonD,radius):
    '''
    Purpose: given geographic coordinates of two separate ground tracks, convert from lla to ecef and then determine the nearest neighbor of each ground track point with respect to the other ground track points given a predetermined radius. Use this to obtain collinear and crossover measurements.
    latA,lonA = geographic coordinates for ascending measurements, A (lon [-180 to 180] and lat [-90 to 90])
    latD,lonD = geographic coordinates for descending measurements, D (lon [-180 to 180] and lat [-90 to 90])
    radius = search radius to find crossover and collinear points
    xA,yA,zA = latA,lonA in ECEF
    xD,yD,zD = latD,lonD in ECEF
    neighbors = nearest neighbor points between A and D
    '''
    # convert lla to ecef
    xA,yA,zA = lgg.lla2ecef(latA,lonA)
    xD,yD,zD = lgg.lla2ecef(latD,lonD)
    dsA = np.asarray([xA,yA,zA]).T #zip(xA,yA,zA)
    dsD = np.asarray([xD,yD,zD]).T #zip(xD,yD,zD)
    #####################
    # create trees
    asc_tree = spatial.KDTree(dsA)
    des_tree = spatial.KDTree(dsD)
    # find indices of descending intersecting points wrt ascending points
    # For each element self.data[i] of this tree, results[i] is a list of the indices of its neighbors in other.data.
    neighbors = asc_tree.query_ball_tree(des_tree, r=radius,eps=0) #radius=12000
    return xA,yA,zA,xD,yD,zD,neighbors
###############################################
###############################################
# logic functions
def logic_days_asc(t_asc,iA1):
    # make sure ascending measurements are in the same day
    if iA1!=0 and iA1!=(np.shape(t_asc)[0]-1):
        tA = t_asc[[iA1-1,iA1,iA1+1]]
        tA_days = (tA/(60.*60.*24)).astype(int)
        logic_tAm1 = np.array_equal(tA_days[:-1],tA_days[:-1])
        logic_tAp1 = np.array_equal(tA_days[1:],tA_days[1:])
        logic_tA = np.array_equal(tA_days,tA_days)
    elif iA1==0 and iA1!=(np.shape(t_asc)[0]-1):
        tA = t_asc[[iA1,iA1+1]]
        tA_days = (tA/(60.*60.*24)).astype(int)
        logic_tAm1 = False
        logic_tAp1 = np.array_equal(tA_days[1:],tA_days[1:])
        logic_tA = np.array_equal(tA_days,tA_days)
    elif iA1!=0 and iA1==(np.shape(t_asc)[0]-1):
        tA = t_asc[[iA1-1,iA1]]
        tA_days = (tA/(60.*60.*24)).astype(int)
        logic_tAm1 = np.array_equal(tA_days[:-1],tA_days[:-1])
        logic_tAp1 = False
        logic_tA = np.array_equal(tA_days,tA_days)
    else: 
        logic_tAm1 = False
        logic_tAp1 = False
        logic_tA = False
    return logic_tA,logic_tAm1,logic_tAp1

def logic_dist_asc(logic_tA,logic_dtAm1,logic_dtAp1,distm1,distp1):
    # stack previous and subsequent ascending node differences together and index the stack [-1 = previous], [+1 = subsequent]
    if logic_tA is True:
        if (logic_dtAm1+logic_dtAp1) == 1:
            if logic_dtAm1 is False:
                distmp = np.copy(distp1)
                distmp_idx = np.ones(np.shape(distp1))
            elif logic_dtAp1 is False:
                distmp = np.copy(distm1)
                distmp_idx = -1.*np.ones(np.shape(distm1))
        elif (logic_dtAm1+logic_dtAp1)==2:
            distmp = np.hstack((distm1,distp1))
            distmp_idx = np.hstack((-1.*np.ones(np.shape(distm1)),np.ones(np.shape(distp1))))
        else:
            distmp = 10000000.
            distmp_idx=[]
    else:
        if logic_dtAm1 is True:
            distmp = distm1
            distmp_idx = -1.*np.ones(np.shape(distm1))
        elif logic_dtAp1 is True:
            distmp = distp1
            distmp_idx = np.ones(np.shape(distp1))
        else:
            distmp = 10000000.
            distmp_idx=[]
    return distmp,distmp_idx

def logic2logic(logicA,logicB):
    logAB = logicA+logicB
    if logAB==2:
        logicAB = True
    else:
        logicAB = False
    return logicAB

###############################################
###############################################
# crossover functions
def tree_dist_col(lat_asc,lon_asc,lat_des,lon_des,t_asc,t_des,radius,GEN):
  '''
  Purpose: find  collinear points between two groundtracks (lon [-180 to 180] and lat [-90 to 90])
  lat_asc,lon_asc,t_asc = geographic coordinates and time stamp of groundtrack (asc)
  lat_des,lon_des,t_des = geographic coordinates and time stamp of groundtrack (des)
  radius = search radius to find collinear points
  idx_asc,idx_des = indices of collinear points for asc and descending
  '''
  # intersecting points defined by the quad tree are kept the same if
  #   there is only one descending measurement for ever ascending measurement.
  #   Otherwise, only the measurements closest to each other are kept.
  ueq=1
  while ueq!=0:
    # find neighboring measurements
    xA,yA,zA,xD,yD,zD,neighbors = ecef2tree(lat_asc,lon_asc,lat_des,lon_des,radius)
    # number unique ascending measurements
    print(len(neighbors))
    Nr = len(neighbors)#np.shape(neighbors)[0]
    # number of intersecting descending points per ascending point
    sz = np.asarray([np.size(neighbors[kk]) for kk in np.arange(Nr)])
    idx= np.where(sz>0.)[0]
    # find where points actually interest (>0)
    idxA = [list((0*np.asarray(neighbors[ii]))+ii) for ii in idx]
    idxD = [neighbors[ii] for ii in idx]
    Nr2 = np.shape(idxA)[0]
    sz2 = np.asarray([np.size(idxD[kk]) for kk in np.arange(Nr2)])
    #####################
    idx_col= np.where(sz2>1.)[0]
    idxA_filt = idxA
    idxD_filt = idxD
    if np.size(idx_col)!=0:
        for ii in idx_col:
            dist = lgg.dist_func(xA[idxA[ii]],yA[idxA[ii]],zA[idxA[ii]],xD[idxD[ii]],yD[idxD[ii]],zD[idxD[ii]]) #np.sqrt((xA[idxA[ii]]-xD[idxD[ii]])**2+(yA[idxA[ii]]-yD[idxD[ii]])**2+(zA[idxA[ii]]-zD[idxD[ii]])**2)
            ir = np.where(dist==dist.min())[0][0]
            idxA_filt[ii] = [np.asarray(idxA[ii])[ir]]
            idxD_filt[ii] = [np.asarray(idxD[ii])[ir]]
    idx_asc_xov = list(itertools.chain(*idxA_filt))
    idx_des_xov = list(itertools.chain(*idxD_filt))
    if GEN == 'xov':
        dT = abs(t_asc[idx_asc_xov]-t_des[idx_des_xov])/(60.*60.*24.)
        idt10 = np.where(dT<10.)[0]
        idx_asc = np.take(idx_asc_xov,idt10)
        idx_des = np.take(idx_des_xov,idt10)
    else:
        idx_asc = np.copy(idx_asc_xov)
        idx_des = np.copy(idx_des_xov)
    Nd = np.shape(np.asarray(idx_des))[0]
    #print('Nd: ',Nd)
    Ndu = np.shape(np.unique(np.asarray(idx_des)))[0]
    #print('Ndu: ',Ndu)
    Na = np.shape(np.asarray(idx_asc))[0]
    #print('Na: ',Na)
    Nau = np.shape(np.unique(np.asarray(idx_asc)))[0]
    #print('Nau: ',Nau)
    ueq=0
    if GEN == 'col':
        if Nd!=Ndu:
            #iol_des = list_duplicates(idx_des)
            ueq = 1
            radius=radius-100
            print('Double-up on second set of coordinates: full/unique='+str(Nd)+'/'+str(Ndu)+' - change radius to '+str(radius))
        if Na!=Nau:
            #iol_asc = list_duplicates(idx_asc)
            if ueq!=1:
                ueq = 1
                radius=radius-100
                print('Double-up on first set of coordinates: full/unique='+str(Na)+'/'+str(Nau)+' - change radius to '+str(radius))
    return idx_asc,idx_des

def tree_dist_xov(lat_asc,lon_asc,lat_des,lon_des,t_asc,t_des,r1_max,radius):
    '''
    Purpose: find  crossover points between two groundtracks (lon [-180 to 180] and lat [-90 to 90])
    lat_asc,lon_asc,t_asc = geographic coordinates and time stamp of groundtrack (asc)
    lat_des,lon_des,t_des = geographic coordinates and time stamp of groundtrack (des)
    radius = search radius to find collinear points
    idx_Aarc,idx_Darc = indices of collinear points for asc and descending
    r1_max = variable no longer in use
    '''
    # http://www.deos.tudelft.nl/ers/operorbs/node11.html
    #####################
    #r_frac = int(radius/r1_max)
    # make sure longitude is in E-W
    # find neighboring measurements
    xA,yA,zA,xD,yD,zD,neighbors = ecef2tree(lat_asc,lon_asc,lat_des,lon_des,radius)
    # number unique ascending measurements
    print(len(neighbors))
    Nr_full = len(neighbors)#np.shape(neighbors)[0]
    # number of intersecting descending points per ascending point
    sz_full = np.asarray([np.size(neighbors[kk]) for kk in np.arange(Nr_full)])
    # Find points that have more than 1 descending measurement withon the radius of each ascending measurement
    #   I tested all single points to see whether they had previous or subsequent points that crossover, but they don't
    idx= np.where(sz_full>1.)[0]
    # find where points actually interest (>0)
    idxA = [list((0*np.asarray(neighbors[ii]))+ii) for ii in idx]
    idxD = [neighbors[ii] for ii in idx]
    Nr = len(idxA)#np.shape(idxA)[0]
    #sz = np.asarray([np.size(idxD[kk]) for kk in np.arange(Nr)])
    #####################
    # use points along the same ground track for interpolation
    idx_Aarc_stk = np.empty((2))*np.nan
    idx_Darc_stk = np.empty((2))*np.nan
    idx_arc = []
    for ii in np.arange(Nr):
        iA = np.asarray(idxA[ii])
        iD = np.asarray(idxD[ii])
        # Find distance between ascending node and neighboring descending nodes
        dist = lgg.dist_func(xA[iA],yA[iA],zA[iA],xD[iD],yD[iD],zD[iD]) # np.asarray([xA[iA[0]],yA[iA[0]],zA[iA[0]]])-np.asarray([xD[iD[0]],yD[iD[0]],zD[iD[0]]])
        # check
        idis = np.where(dist>radius)[0]
        if np.size(idis)!=0:
            print('number of valid xov > radius = ',np.size(idis))
            raise('distance function not working')

        # Convert time differences amongst descending measurements to days 
        #   Group all descending measurements within the same pass
        tD = t_des[iD]
        tD_days = (tD/(60.*60.*24)).astype(int)
        tD_days_uni,tD_days_idx,tD_days_inv = np.unique(tD_days,return_index=True,return_inverse=True)
        tD_days_cnt = lgg.unique_return_counts(tD_days,tD_days_uni)
        # Consider only crossovers where there are at least 2 desending measurements from the same pass surrounding a single ascending pass within radius 
        icnt = np.where(tD_days_cnt>1)[0]
        if np.size(icnt)!=0:
            for nn in np.arange(np.size(icnt)):
                iinv = np.where(tD_days_inv==tD_days_idx[icnt[nn]])[0]
                # Determine whether the descending points crossover the ascending pass
                latDi = lat_des[iD[iinv]]
                latAi = lat_asc[iA[0]]
                dlatDA = latDi-latAi
                logic = all(item >= 0 for item in dlatDA) or all(item < 0 for item in dlatDA)
                if logic == False:
                    # if there are more than 2 descending measurements then take the two closest to the ascending measurement
                    if np.size(iinv)>2:
                        idx_dist_srt = np.argsort(dist[iinv])
                        iinv2 = iinv[idx_dist_srt[:2]]
                    else:
                        iinv2 = np.copy(iinv)
                    # find second ascending node that crosses descending pass
                    if np.size(iinv2)==2:
                        iA1 = iA[0]
                        # distance from previous ascending node to each of the descending measurements
                        if iA1!=0:
                            pxAm1,pxDm1,lonXm1,latXm1 = lla_linear_interp(lat_asc[[iA1-1,iA1]],lon_asc[[iA1-1,iA1]],lat_des[iD[iinv2]],lon_des[iD[iinv2]])
                            px_sum_m1 = np.round(np.asarray([np.sum(pxAm1),np.sum(pxDm1)]),2)
                            id_m1 = np.where(px_sum_m1!=1.0)[0]
                            # Sum of partials == 1 indicates full crossover
                            if np.size(id_m1)==0:
                                distm1 = lgg.dist_func(xA[iA1-1],yA[iA1-1],zA[iA1-1],xD[iD[iinv2]],yD[iD[iinv2]],zD[iD[iinv2]]) 
                                logic_dAm1 = True
                            else:
                                logic_dAm1 = False
                                distm1 = []
                        else:
                            logic_dAm1 = False
                            distm1 = []
                        # distance from subsequent ascending node to each of the descending measurements
                        if iA1!=(np.shape(xA)[0]-1):
                            pxAp1,pxDp1,lonXp1,latXp1 = lla_linear_interp(lat_asc[[iA1,iA1+1]],lon_asc[[iA1,iA1+1]],lat_des[iD[iinv2]],lon_des[iD[iinv2]])
                            px_sum_p1 = np.round(np.asarray([np.sum(pxAp1),np.sum(pxDp1)]),2)
                            id_p1 = np.where(px_sum_p1!=1.0)[0]
                            # Sum of partials == 1 indicates full crossover
                            if np.size(id_p1)==0:
                                distp1 = lgg.dist_func(xA[iA1+1],yA[iA1+1],zA[iA1+1],xD[iD[iinv2]],yD[iD[iinv2]],zD[iD[iinv2]]) 
                                logic_dAp1 = True
                            else:
                                logic_dAp1 = False
                                distp1 = []
                        else:
                            logic_dAp1 = False
                            distp1 = []

                        # make sure ascending measurements are in the same day
                        logic_tA,logic_tAm1,logic_tAp1 = logic_days_asc(t_asc,iA1)
                        logic_dtAm1 = logic2logic(logic_dAm1,logic_tAm1)
                        logic_dtAp1 = logic2logic(logic_dAp1,logic_tAp1)
                        # stack previous and subsequent ascending node differences together and index the stack [-1 = previous], [+1 = subsequent]
                        distmp,distmp_idx = logic_dist_asc(logic_tA,logic_dtAm1,logic_dtAp1,distm1,distp1)
                        # make sure that the minimum distance within the stack is within reasonable distance to the next descending measurement
                        if np.nanmin(distmp)<=radius:
                            # find whether the minimum distance within the stack belongs to previous or subsequent ascending measurement
                            find_idx_2ndA = np.where(distmp==np.nanmin(distmp))[0]
                            # create ascending pass crossover pair
                            if distmp_idx[find_idx_2ndA[0]]>0:
                                iA_arr = np.asarray([iA1,iA1+1.]).astype(int)
                                #pxA,pxD = pxAp1,pxDp1
                            else:
                                iA_arr = np.asarray([iA1-1.,iA1]).astype(int)
                                #pxA,pxD = pxAm1,pxDm1
                            # create descending pass crossover pair
                            iD_arr = np.take(iD,iinv2)
                            dT = (abs(t_asc[iA_arr[0]]-t_des[iD_arr[0]])/(60.*60.*24.))
                            # only save crossover pairs that are less than 10 days apart
                            if dT>0 and dT<2: #if dT<10
                                # iteratively save all pairs into a single stack
                                idx_Aarc_stk = np.vstack((idx_Aarc_stk,iA_arr))
                                idx_Darc_stk = np.vstack((idx_Darc_stk,iD_arr))
                                # save all indices
                                idx_arc.append(np.copy(ii))
    # remove buffer row
    #print('shape of idx_Aarc_stk= '+str(np.shape(idx_Aarc_stk))+', shape of idx_Darc_stk= '+str(np.shape(idx_Darc_stk)))
    idx_Aarc_nn = np.copy(idx_Aarc_stk[1:,:]).astype(int)
    idx_Darc_nn = np.copy(idx_Darc_stk[1:,:]).astype(int)
    # remove all duplicate crossovers
    idx_ADarc = np.column_stack((idx_Aarc_nn,idx_Darc_nn)) # (8568, 4), expand allowable stretch: (8658, 4)
    #arc_uni,arc_idx,arc_inv = np.unique(idx_ADarc,return_index=True,return_inverse=True,axis=0)
    #print('shape/type idx_ADarc '+str(idx_ADarc.shape)+' / '+str(idx_ADarc.dtype))
    arc_uni = lgg.unique_rows(idx_ADarc)
    N = np.shape(arc_uni)[0]
    arc_idx = np.asarray([np.where((idx_ADarc == arc_uni[ii,:]).all(axis=1))[0][0] for ii in np.arange(N)])
    # final array of indices for all crossover pairs
    idx_Aarc = np.take(idx_Aarc_nn,arc_idx,axis=0) #(8341, 2), expand allowable stretch: (8431, 4)
    idx_Darc = np.take(idx_Darc_nn,arc_idx,axis=0)
    return idx_Aarc,idx_Darc

def xov_partials(idx_Aarc,lat_asc,lon_asc,idx_Darc,lat_des,lon_des):
    '''
    Purpose: extension of lla_linear_interp given multiple measurements and indices corresponding to neighboring measurements.
    idx_Aarc,idx_Darc = indices of collinear or crossover points between groundtrack A and groundtrack D
    lat_asc,lon_asc= geographic coordinates for ascending measurements, A
    lat_des,lon_des = geographic coordinates for descending measurements, D
    lonX,latX = geographic coordinates of intersecting point, X, between A and D
    partials_asc,partials_des= weights (or partials) between A and X and D and X
    '''
    #partials_ascEW,partials_desEW,lonXEW,latXEW = xov_partials(idxEW_Aarc,lat_asc,lonEW_asc,idxEW_Darc,lat_des,lonEW_des)
    Nm = np.shape(idx_Aarc)[0]
    partials_asc = np.empty((Nm,2))*np.nan #ittEW = np.where(partials_asc.sum(axis=1)>1000)[0]
    partials_des = np.empty((Nm,2))*np.nan
    lonX = np.empty((Nm))*np.nan
    latX = np.empty((Nm))*np.nan
    for ii in np.arange(Nm):
        itstA,itstD =  idx_Aarc[ii,:],idx_Darc[ii,:] 
        latA,lonA,latD,lonD = lat_asc[itstA],lon_asc[itstA],lat_des[itstD],lon_des[itstD]
        partials_asc[ii,:],partials_des[ii,:],lonX[ii],latX[ii] = lla_linear_interp(latA,lonA,latD,lonD) 
    return partials_asc,partials_des,lonX,latX

def tree_dist(lat_asc,lon_asc,lat_des,lon_des,t_asc,t_des,radius,GEN,INTERP):
    # Purpose: final run function to provide the indices of collinear or crossover points given two groundtracks using other functions in script
    # (lon [-180 to 180] and lat [-90 to 90])
    st1=time.time()
    r1_max = 3000.
    #####################
    # filter data for col
    if GEN == 'col':
        idx_asc,idx_des = tree_dist_col(lat_asc,lon_asc,lat_des,lon_des,t_asc,t_des,radius,GEN)
        partials_asc,partials_des,lonX,latX = [],[],[],[]
    #####################
    # filter data for xov
    # make sure that intersecting points that don't need interpolation, are within 3000 km of each other
    if GEN == 'xov':
        if INTERP == True:
            idx_asc,idx_des = tree_dist_xov(lat_asc,lon_asc,lat_des,lon_des,t_asc,t_des,r1_max,radius)
            partials_asc,partials_des,lonX,latX = xov_partials(idx_asc,lat_asc,lon_asc,idx_des,lat_des,lon_des)
        elif INTERP == False:
            idx_asc_pre,idx_des_pre = tree_dist_col(lat_asc,lon_asc,lat_des,lon_des,t_asc,t_des,radius,GEN)
            idx_vstk = np.vstack((idx_asc_pre,idx_des_pre)).T
            partials_asc,partials_des,lonX,latX = [],[],[],[]
            nn_uni,nn_idx,nn_inv = np.unique(idx_vstk,return_index=True,return_inverse=True,axis=0)
            idx_asc,idx_des = nn_uni[:,0],nn_uni[:,1]
    #####################
    print('time to find nearest neighbors (radius= '+str(radius)+' m): '+str(time.time()-st1))
    return idx_asc,idx_des,partials_asc,partials_des,lonX,latX


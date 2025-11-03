#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:44:39 2020

@author: alexaputnam

Author: Alexa Angelina Putnam Shilleh
Institution: University of Colorado Boulder
Software: sea state bias model generator (interpolation method)
Thesis: Sea State Bias Model Development and Error Analysis for Pulse-limited Altimetry
Semester/Year: Fall 2021
"""
import numpy as np
from netCDF4 import Dataset
from scipy import interpolate
import time

###############################################
###############################################
# mss

def spline_interp_mss(MISSIONNAME,lat,lon,ERR=False,MSS='cnes15'):
    '''
    Purpose: Interpolate a new MSS solution (mss) and corresponding error (mss_err) if Grid_0002 available given geographic coordinates (lat,lon)
    MSS = in this case, MSS defines the MSS model. As of now there are only the CNES15 and CNES11 models.
    CLS15: only for Jason-1 and -2
    CLS11: only for Jason-1
    '''
    #from scipy import interpolate
    t1 = time.time()
    N = np.shape(lat)[0]
    if MISSIONNAME == 'j1':
        if MSS=='cnes15':
            ds_mss = Dataset('/OSTM1/data/model/CLS15/mss_cnes_cls2015.nc').variables
        elif MSS=='cnes11':
            ds_mss = Dataset('/OSTM1/data/model/CLS11/mss_cnes_cls2011.nc').variables
    elif MISSIONNAME == 'j2':
        if MSS=='cnes15':
            ds_mss = Dataset('/OSTM1/data/model/CLS15/mss_cnes_cls2015.nc').variables
    #else:
    #    raise('no MSS '+MSS+' solution provided for '+MISSIONNAME+'. Set MSS=False')
    if MSS=='cnes15':
            ds_mss = Dataset('/OSTM1/data/model/CLS15/mss_cnes_cls2015.nc').variables
    lat_grid = ds_mss['NbLatitudes'][:]#x
    lon_grid = ds_mss['NbLongitudes'][:]#y
    mss_grid = ds_mss['Grid_0001'][:]
    f_mss = interpolate.interp2d(lat_grid, lon_grid, mss_grid, kind='linear')
    print('Time to apply linear spline interpolation for MSS for 1 cycle '+str(time.time()-t1))
    if np.size(np.shape(lat))==1:
	#mss = f_mss(lat,lon)
	#'''
        mss = np.empty(N)*np.nan
        for ii in np.arange(N):
            mss[ii] = f_mss(lat[ii],lon[ii])
	#'''
    else:
	#mss = np.empty((N,2))*np.nan
	#mss[:,0] = f_mss(lat[:,0],lon[:,0])
	#mss[:,1] = f_mss(lat[:,1],lon[:,1])
	#'''
        mss = np.empty((N,2))*np.nan
        for ii in np.arange(N):
            mss[ii,0] = f_mss(lat[ii,0],lon[ii,0])
            mss[ii,1] = f_mss(lat[ii,1],lon[ii,1])
	#'''
    if ERR is True:
        mss_err_grid = ds_mss['Grid_0002'][:]
        f_mss_err = interpolate.interp2d(lat_grid, lon_grid, mss_err_grid, kind='linear')
        if np.size(np.shape(lat))==1:
            mss_err = np.empty(N)*np.nan
            for ii in np.arange(N):
                mss_err[ii] = f_mss_err(lat[ii],lon[ii])
        else:
            mss_err = np.empty((N,2))*np.nan
            for ii in np.arange(N):
                mss_err[ii,0] = f_mss_err(lat[ii,0],lon[ii,0])
                mss_err[ii,1] = f_mss_err(lat[ii,1],lon[ii,1])
    else:
        mss_err = []

    return mss,mss_err


###############################################
###############################################
# gridding
def unique_rows(A, return_index=False, return_inverse=False):
    """
    Input: A, return_index=False, return_inverse=False
    Output: B
    Purpose: Similar to MATLAB's unique(A, 'rows'), where B is the unique rows of A.

    Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
    where B is the unique rows of A and I and J satisfy
    A = B[J,:] and B = A[I,:]

    Returns I if return_index is True
    Returns J if return_inverse is True
    """
    A = np.require(A, requirements='C')
    assert A.ndim == 2, "array must be 2-dim'l"

    B = np.unique(A.view([('', A.dtype)]*A.shape[1]),return_index=return_index,return_inverse=return_inverse)

    if return_index or return_inverse:
        return (B[0].view(A.dtype).reshape((-1, A.shape[1]), order='C'),) \
            + B[1:]
    else:
        return B.view(A.dtype).reshape((-1, A.shape[1]), order='C')

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

def reduced_gauss_grid(size=3):
    '''
    Purpose: Create a reduced Gaussian grid
    size = integer defining the degree difference between latitudinal nodes
    lon,lat = coordinates for reduced Gaussian grid nodes (lon [0 to 360], lat [-90 to 90])
    lon_diff = degree difference between longitudinal nodes
    lat_diff = degree difference between latitudinal nodes
    area = area of reduced Gaussian grid segments
    '''
    lon = []
    lat = []
    lon_diff = []
    lat_diff = []
    lat0 = np.arange(-90, 90+size, size)
    dlat = size
    pts_per_lat = np.empty(np.shape(lat0))*np.nan
    for i in range(len(lat0)):
        nlon = int(np.floor(360./size*np.cos(lat0[i]*np.pi/180)))
        if np.abs(lat0[i])==90:
            nlon =1
            lon_row = [0]
            lon_diff += [360]*len(lon_row)
        else:
            lon_row = np.arange(360/nlon, 360+360/nlon, 360./nlon)
            pts_per_lat[i] = np.shape(lon_row)[0]
            lon_diff += [np.diff(lon_row)[0]]*len(lon_row)
        lon += list(lon_row)
        lat += [lat0[i]]*len(lon_row)
        lat_diff += [dlat]*len(lon_row)
    lon,lat,lon_diff,lat_diff = np.array(lon), np.array(lat),np.array(lon_diff), np.array(lat_diff)
    area = lon_diff*lat_diff*np.cos(lat*(np.pi/180.))*(np.pi/180.)**2 #(np.sum(area))/(4.*np.pi)
    area[[0,-1]] = 2.*np.pi*(1.-np.cos(size/2.*(np.pi/180.))) #4551*((np.sum(area))/(4.*np.pi)-1.)
    # accuracy of covering the sphere: ((np.sum(area))/(4.*np.pi)-1.) = 0.01 %
    return lon,lat,lon_diff,lat_diff,area

def unique_return_counts(arr,arr_unique):
    # Purpose: count (cnt) the number of times a certain value occurs in an array given array (arr) and the unique array of arr (arr_unique = arr with all duplicates removed)
    cnt = np.asarray([list(arr).count(ii) for ii in arr_unique])
    return cnt

def row2keep(H,idx_ssb):
    '''
    Purpose: provide the indices (kp_row) for all observations that observe at least one node.
    H =  matrix containing all partials. H size is [M,N], where M = number of measurements, and N = number of nodes
    idx_ssb =  indices of all observed nodes
    '''
    H1=H.copy()
    H1[H1!=0]=1
    mx=0#4#8 #5
    rH1=np.squeeze(np.asarray(H1.sum(axis=1)))
    if np.size(idx_ssb)!=0:
        H1_ssb=H[:,idx_ssb].copy()
        H1_ssb[H1_ssb!=0]=1
        rH1_ssb=np.squeeze(np.asarray(H1_ssb.sum(axis=1)))
        kp_row = np.where(rH1_ssb>mx)[0]
    else:
        kp_row = np.where(rH1>=mx)[0]
    return kp_row


def data2grid(fSSB,fN,mN,idx):
    '''
    Purpose: Reshape least squares output, or estimated model (fSSB), into a collapsed full model or a 2D full model. This function injects fSSB (which does not observe each node) into the predefined grid of the full model. All unobserved nodes are left as NaN.
    fN = size of collapsed full model
    mN = shape of full model
    idx = indices of observed nodes (must be same size as fSSB)
    '''
    fgrid_fill = np.empty((fN))*np.nan
    fgrid_fill[idx] = np.copy(fSSB)
    if np.size(mN) >= 2:
        grid = np.reshape(fgrid_fill,mN,order='F')
    elif np.size(mN) < 2:
        grid = fgrid_fill
    else:
        raise ValueError('lib_bilinear/data2grid: incorrect dimensions for mN array')
    return grid

def obsmat_count(H):
    # Purpose: count the number of observations per node (col_cnt) given the matrix containing all partials for the least squares (H)
    H_cnt = H.copy()
    H_cnt[H_cnt!=0.] = 1.
    col_cnt = (H_cnt.T.dot(H_cnt)).diagonal()
    return col_cnt

def grid_dimensions(DIM,OMIT):
    '''
    Purpose: provides the arrays corresponding to the grid axis dimensions of the SSB model
    x_grid = SWH node points
    y_grid =  u10 or sig0 node points
    z_grid = node points of 3rd parameter (for 3D model only)
    '''
    if OMIT == 'mean_wave_period_t02':
        dz = 0.25 #3.0
        z_grid = np.arange(1.0,15.+dz,dz)
    elif np.size(OMIT)==0:
        z_grid = []
    else:
        raise('Dimensions undefined')
    if DIM == 'u10': #Tran: 11.75 m, 20.75 m/s
        dx = 0.25
        dy = 0.25
        x_grid = np.arange(0.,12.00+dx,dx) #np.arange(1.5,3.5+dx,dx)#
        y_grid = np.arange(0.,20.75+dy,dy) #np.arange(6.25,9.75+dy,dy)#
    elif DIM == 'sig0':
        dx = 0.25
        dy = 0.25
        x_grid = np.arange(0.,15.0+dx,dx)
        y_grid = np.arange(10.,20.0+dy,dy) 
    return x_grid,y_grid,z_grid

###############################################
###############################################
# geographic and distance functions
def lla2ecef(lat_deg,lon_deg):
    # Purpose: convert geodetic coordinates (lat_deg, lon_deg) to Earth-centered Earth-fixed (ECEF) coordinates (x,y,z) assuming altitude = 0
    # (lon [-180 to 180] and lat [-90 to 90])
    # WGS84 ellipsoid constants:
    alt = 0.
    a = 6378137. #height above WGS84 ellipsoid (m)
    e = 8.1819190842622e-2
    d2r = np.pi/180.
    N = a/np.sqrt(1.-(e**2)*(np.sin(lat_deg*d2r)**2))
    x = (N+alt)*np.cos(lat_deg*d2r)*np.cos(lon_deg*d2r)
    y = (N+alt)*np.cos(lat_deg*d2r)*np.sin(lon_deg*d2r)
    z = ((1.-e**2)*N+alt)*np.sin(lat_deg*d2r)
    return x,y,z

def lon360_to_lon180(lon_old):
    # Purpose: convert longitude from lon_old (0 to 360) to lon_new (-180 to 180)
    igt = np.where(lon_old>180)[0] #!!! used to be (prior to may 5 2021) lon_old>=180
    if np.size(igt)!=0:
        lon_new = np.mod((lon_old+180.),360.)-180.
    else:
        lon_new = np.copy(lon_old)
    return lon_new

def lon180_to_lon360(lon_old):
    # Purpose: convert longitude from lon_old (-180 to 180) to lon_new (0 to 360)
    igt = np.where(lon_old<0)[0]
    if np.size(igt)!=0:
        lon_new = np.mod(lon_old,360.)
    else:
        lon_new = np.copy(lon_old)
    return lon_new


def dist_func(xA,yA,zA,xD,yD,zD):
    '''
    Purpose: Distance (dist) between two points in xyz-space. Mainly used for determining the distance between collinear and crossover difference measurements.
    xA,yA,zA = ECEF coordinates of ascending (or pass i) measurements
    xD,yD,zD = ECEF coordinates of descending (or pass i+1) measurements
    '''
    dist = np.sqrt((np.subtract(xA,xD)**2)+(np.subtract(yA,yD)**2)+(np.subtract(zA,zD)**2))
    return dist 


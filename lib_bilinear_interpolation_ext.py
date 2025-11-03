#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 19:29:03 2019

@author: alexaputnam

Author: Alexa Angelina Putnam Shilleh
Institution: University of Colorado Boulder
Software: sea state bias model generator (interpolation method)
Thesis: Sea State Bias Model Development and Error Analysis for Pulse-limited Altimetry
Semester/Year: Fall 2021
"""

# IN-HOUSE FUNCTIONS
import time
import numpy as np
import scipy as sp
from scipy import interpolate

# PUTNAM FUNCTIONS
import lib_geo_gridding_ext as lgg #!!!!
# --------------------------------------------------------------------------#
#-------------------Bilinear Interpolation----------------------------------#
# --------------------------------------------------------------------------#
'''
VARIABLE REFERENCE
x_obs = x-axis parameter observations (N)
y_obs = y-axis parameter observations (N)
x_grid = x-axis grid nodes (X)
y_grid = y-axis grid nodes (Y)
H_sparse = observation matrix (N,[XxY])
x_scatter = flattened copy of x-axis value grid points (XxY)
y_scatter = flattened copy of y-axis value grid points (XxY)
p00,p10,p01,p11 = weighted distance (partials) with the last two number indicating the nodes associated with the partials
B = node values
z = estimated value at point [x,y]
B_obs_gd = estimated value for each observation
'''
def bilinear_observation_matrix(x_obs,y_obs,x_grid,y_grid): 
    '''
    Purpose: create observation matrix for the interpolation method
    Necessary shape to match 'bilinear_partials.py'
        Bins are observed row by row
        Nodes = [(x0,y0), (x1,y0), (x0,y1), (x1,y1)]
    x_obs,y_obs,z_obs = x_obsA,y_obsA,z_obsA
    '''    
    # Create grid in 2d
    mx,my,mz,x_scatter,y_scatter,fz = lgg.parameter_grid(x_grid,y_grid,[])
    # Find sizes
    Nsz = np.size(x_obs)
    Nsp = np.shape(x_obs)[0]
    if Nsz == Nsp:
        Nobs = np.shape(x_obs)[0]  
        Nd = 1
    elif Nsz > Nsp:
        Nobs,Nd = np.shape(x_obs)
    Nx_grid,Ny_grid = np.shape(mx)
    Nx_bin,Ny_bin = Nx_grid-1,Ny_grid-1
    Nnodes = Nx_grid*Ny_grid
    # Initialize indicies
    ibinX = np.asarray([0,1,0,1])
    ibinY = np.asarray([0,0,1,1])
    # Node numbering
    fNodes = np.arange(Nnodes)
    mNodes = np.reshape(fNodes,(Nx_grid,Ny_grid),order='F')
    mNodes.astype(int)
    # Observation matrix
    if Nd == 1:
        H_sparse = sp.sparse.lil_matrix((Nobs,Nnodes))
    elif Nd == 2:
        H_sparseA = sp.sparse.lil_matrix((Nobs,Nnodes))
        H_sparseD = sp.sparse.lil_matrix((Nobs,Nnodes))
    #fNodes_order_chk = []
    T1 = time.time()
    for yy in np.arange(Ny_bin):
        iY = yy+ibinY
        for xx in np.arange(Nx_bin):
            iX = xx+ibinX
            observed_nodes = mNodes[iX,iY]
            #fNodes_order_chk.extend(observed_nodes)
            x_val = mx[iX,iY]
            y_val = my[iX,iY]
            # Indices of data present within ibserved bin
            if Nd == 1:
                iXY = np.where((x_obs>x_val[0]) & (x_obs<=x_val[-1]) & (y_obs>y_val[0]) & (y_obs<=y_val[-1]))[0]
                # The four partials corresponding to four nodes for each data point
                if np.size(iXY) != 0:
                    p00,p10,p01,p11 = bilinear_partials(x_obs[iXY],x_val[0],x_val[-1],y_obs[iXY],y_val[0],y_val[-1])
                    # Observation matrix
                    H_sparse[iXY,observed_nodes[0]]=np.reshape(p00,(len(p00),1))
                    H_sparse[iXY,observed_nodes[1]]=np.reshape(p10,(len(p10),1))
                    H_sparse[iXY,observed_nodes[2]]=np.reshape(p01,(len(p01),1))
                    H_sparse[iXY,observed_nodes[3]]=np.reshape(p11,(len(p11),1))
            elif Nd == 2:
                iXYa = np.where((x_obs[:,0]>x_val[0]) & (x_obs[:,0]<=x_val[-1]) & (y_obs[:,0]>y_val[0]) & (y_obs[:,0]<=y_val[-1]))[0]
                iXYd = np.where((x_obs[:,1]>x_val[0]) & (x_obs[:,1]<=x_val[-1]) & (y_obs[:,1]>y_val[0]) & (y_obs[:,1]<=y_val[-1]))[0]
                # The four partials corresponding to four nodes for each data point
                if np.size(iXYa) != 0:
                    p00a,p10a,p01a,p11a = bilinear_partials(x_obs[iXYa,0],x_val[0],x_val[-1],y_obs[iXYa,0],y_val[0],y_val[-1])
                    # Observation matrix
                    H_sparseA[iXYa,observed_nodes[0]]=np.reshape(p00a,(len(p00a),1))
                    H_sparseA[iXYa,observed_nodes[1]]=np.reshape(p10a,(len(p10a),1))
                    H_sparseA[iXYa,observed_nodes[2]]=np.reshape(p01a,(len(p01a),1))
                    H_sparseA[iXYa,observed_nodes[3]]=np.reshape(p11a,(len(p11a),1))
                if np.size(iXYd) != 0:
                    p00d,p10d,p01d,p11d = bilinear_partials(x_obs[iXYd,1],x_val[0],x_val[-1],y_obs[iXYd,1],y_val[0],y_val[-1])
                    # Observation matrix
                    H_sparseD[iXYd,observed_nodes[0]]=np.reshape(p00d,(len(p00d),1))
                    H_sparseD[iXYd,observed_nodes[1]]=np.reshape(p10d,(len(p10d),1))
                    H_sparseD[iXYd,observed_nodes[2]]=np.reshape(p01d,(len(p01d),1))
                    H_sparseD[iXYd,observed_nodes[3]]=np.reshape(p11d,(len(p11d),1))
    print('Time to create bilinear observation_matrix for a single cycle [s]: '+str(time.time()-T1))
    # Convert to CSR
    if Nd == 1:
        H_sparse = H_sparse.tocsr()   
    elif Nd == 2:
        H_sparse = (H_sparseA-H_sparseD).tocsr()# original: (H_sparseA-H_sparseD).tocsr(). !!!! SWITCH TO CHECK TRAN ASSUMPTION   
  
    return H_sparse,x_scatter,y_scatter



def bilinear_partials(x,x0,x1,y,y0,y1):
    ''' 
    Purpose: computes the partials for each of the 4 nodes surrounding
        a single observation point. Multiple observations may be analyzed.
    Bins are observed row by row.
    Bin nodes coordinates of [p00,p10,p01,p11] =
    [(x0,y0), (x1,y0), (x0,y1), (x1,y1)]
    '''
    wx = np.divide(np.subtract(x,x0),np.subtract(x1,x0))
    wy = np.divide(np.subtract(y,y0),np.subtract(y1,y0))
    p00 = np.multiply((1.-wx),(1.-wy))
    p10 = np.multiply((wx),(1.-wy))
    p01 = np.multiply((1.-wx),(wy))
    p11 = np.multiply((wx),(wy))
    return p00,p10,p01,p11

def bilinear_forward(x,x0,x1,y,y0,y1,B):
    '''
    Purpose: uses least square to determine the value of a point within a grid
    '''
    B = np.asarray(B)
    N = np.shape(x)[0]
    z = np.empty((N))
    for ii in np.arange(N):
        xi = x[ii]
        yi = y[ii]
        p1234 = bilinear_partials(xi,x0,x1,yi,y0,y1)
        z[ii] = np.dot(B,p1234)
    return z

def bilinear_interpolation(x_obs,y_obs,x_grid,y_grid,B):
    # Purpose: uses least square to determine the value of all points within a grid/model
    t0 = time.time()
    Nr,Nc = np.shape(B)
    Nx = np.shape(x_grid)[0]
    Ny = np.shape(y_grid)[0]
    ibinX = np.asarray([0,1,0,1])
    ibinY = np.asarray([0,0,1,1])
    if (Nx,Ny) != (Nr,Nc):
        B = B.T
    B_obs = np.empty((np.shape(x_obs)[0]))*np.nan
    for yy in np.arange(Nc-1):
        y0 = y_grid[yy]
        y1 = y_grid[yy+1]
        iY = yy+ibinY
        for xx in np.arange(Nr-1):
            x0 = x_grid[xx]
            x1 = x_grid[xx+1]
            iX = xx+ibinX
            B10 = np.copy(B[iX,iY])
            ixy = np.where((x_obs>=x0) & (x_obs<x1) & (y_obs>=y0) & (y_obs<y1))[0]
            xi = x_obs[ixy]
            yi = y_obs[ixy]
            B_obs[ixy] = bilinear_forward(xi,x0,x1,yi,y0,y1,B10)
    print('time bilineary interpolate '+str(np.shape(x_obs)[0])+' measurements: '+str(np.round(time.time()-t0,3)))
    return B_obs
            
def bilinear_interpolation_sp_griddata(x_obs,y_obs,x_grid,y_grid,B):
    # Purpose: uses least square to determine the value of all points within a grid/model
    mx,my,mz,x_scatter,y_scatter,z_scatter = lgg.parameter_grid(x_grid,y_grid,[])
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



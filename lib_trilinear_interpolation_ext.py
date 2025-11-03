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
#-------------------Trilinear Interpolation----------------------------------#
# --------------------------------------------------------------------------#

def trilinear_observation_matrix(x_obs,y_obs,z_obs,x_grid,y_grid,z_grid): 
    '''
    Purpose: creates observation matrix for the interpolation method of a 3D model
    x_obs = x-axis parameter observations (N)
    y_obs = y-axis parameter observations (N)
    z_obs = z-axis parameter observations (N)
    x_grid = x-axis grid nodes (X)
    y_grid = y-axis grid nodes (Y)
    z_grid = z-axis grid nodes (Z)
    H_sparse = least squares partials matrix (N,[XxY])
    x_scatter = collapsed x coordinate matrices (XxYxZ)
    y_scatter = collapsed y coordinate matrices (XxYxZ)
    z_scatter = collapsed z coordinate matrices(XxYxZ)
    '''    
    # Create grid in 2d
    mx,my,mz,x_scatter,y_scatter,z_scatter = lgg.parameter_grid(x_grid,y_grid,z_grid)
    # Find sizes
    Nsz = np.size(x_obs)
    Nsp = np.shape(x_obs)[0]
    if Nsz == Nsp:
        Nobs = np.shape(x_obs)[0]  
        Nd = 1
    elif Nsz > Nsp:
        Nobs,Nd = np.shape(x_obs)
    Nx_grid,Ny_grid,Nz_grid = np.shape(mx)
    Nx_bin,Ny_bin,Nz_bin = Nx_grid-1,Ny_grid-1,Nz_grid-1
    Nnodes = Nx_grid*Ny_grid*Nz_grid
    # Initialize indicies
    ibinX = np.asarray([0,1,0,1,0,1,0,1])
    ibinY = np.asarray([0,0,1,1,0,0,1,1])
    ibinZ = np.asarray([0,0,0,0,1,1,1,1])
    # Node numbering
    fNodes = np.arange(Nnodes)
    mNodes = np.reshape(fNodes,(Nx_grid,Ny_grid,Nz_grid),order='F')
    mNodes.astype(int)
    # Observation matrix
    if Nd == 1:
        H_sparse = sp.sparse.lil_matrix((Nobs,Nnodes))
    elif Nd == 2:
        H_sparseA = sp.sparse.lil_matrix((Nobs,Nnodes))
        H_sparseD = sp.sparse.lil_matrix((Nobs,Nnodes))
    #fNodes_order_chk = []
    T1 = time.time()
    for zz in np.arange(Nz_bin):
        iZ = zz+ibinZ
        for yy in np.arange(Ny_bin):
            iY = yy+ibinY
            for xx in np.arange(Nx_bin):
                iX = xx+ibinX
                observed_nodes = mNodes[iX,iY,iZ]
                #fNodes_order_chk.extend(observed_nodes)
                x_val = mx[iX,iY,iZ]
                y_val = my[iX,iY,iZ]
                z_val = mz[iX,iY,iZ]
                # Indices of data present within ibserved bin
                if Nd == 1:
                    iXYZ = np.where((x_obs>x_val[0]) & (x_obs<=x_val[-1]) & (y_obs>y_val[0]) & (y_obs<=y_val[-1]) & (z_obs>z_val[0]) & (z_obs<=z_val[-1]))[0]
                    # The four partials corresponding to four nodes for each data point
                    if np.size(iXYZ) != 0:
                        p000,p100,p010,p110,p001,p101,p011,p111 = trilinear_partials(x_obs[iXYZ],x_val[0],x_val[-1],y_obs[iXYZ],y_val[0],y_val[-1],z_obs[iXYZ],z_val[0],z_val[-1])
                        # Observation matrix
                        H_sparse[iXYZ,observed_nodes[0]]=np.reshape(p000,(len(p000),1))
                        H_sparse[iXYZ,observed_nodes[1]]=np.reshape(p100,(len(p100),1))
                        H_sparse[iXYZ,observed_nodes[2]]=np.reshape(p010,(len(p010),1))
                        H_sparse[iXYZ,observed_nodes[3]]=np.reshape(p110,(len(p110),1))
                        H_sparse[iXYZ,observed_nodes[4]]=np.reshape(p001,(len(p001),1))
                        H_sparse[iXYZ,observed_nodes[5]]=np.reshape(p101,(len(p101),1))
                        H_sparse[iXYZ,observed_nodes[6]]=np.reshape(p011,(len(p011),1))
                        H_sparse[iXYZ,observed_nodes[7]]=np.reshape(p111,(len(p111),1))
                elif Nd == 2:
                    iXYZa = np.where((x_obs[:,0]>x_val[0]) & (x_obs[:,0]<=x_val[-1]) & (y_obs[:,0]>y_val[0]) & (y_obs[:,0]<=y_val[-1]) & (z_obs[:,0]>z_val[0]) & (z_obs[:,0]<=z_val[-1]))[0]
                    iXYZd = np.where((x_obs[:,1]>x_val[0]) & (x_obs[:,1]<=x_val[-1]) & (y_obs[:,1]>y_val[0]) & (y_obs[:,1]<=y_val[-1]) & (z_obs[:,1]>z_val[0]) & (z_obs[:,1]<=z_val[-1]))[0]
                    # The four partials corresponding to four nodes for each data point
                    if np.size(iXYZa) != 0:
                        p000a,p100a,p010a,p110a,p001a,p101a,p011a,p111a = trilinear_partials(x_obs[iXYZa,0],x_val[0],x_val[-1],y_obs[iXYZa,0],y_val[0],y_val[-1],z_obs[iXYZa,0],z_val[0],z_val[-1])
                        # Observation matrix
                        H_sparseA[iXYZa,observed_nodes[0]]=np.reshape(p000a,(len(p000a),1))
                        H_sparseA[iXYZa,observed_nodes[1]]=np.reshape(p100a,(len(p100a),1))
                        H_sparseA[iXYZa,observed_nodes[2]]=np.reshape(p010a,(len(p010a),1))
                        H_sparseA[iXYZa,observed_nodes[3]]=np.reshape(p110a,(len(p110a),1))
                        H_sparseA[iXYZa,observed_nodes[4]]=np.reshape(p001a,(len(p001a),1))
                        H_sparseA[iXYZa,observed_nodes[5]]=np.reshape(p101a,(len(p101a),1))
                        H_sparseA[iXYZa,observed_nodes[6]]=np.reshape(p011a,(len(p011a),1))
                        H_sparseA[iXYZa,observed_nodes[7]]=np.reshape(p111a,(len(p111a),1))
                    if np.size(iXYZd) != 0:
                        p000d,p100d,p010d,p110d,p001d,p101d,p011d,p111d = trilinear_partials(x_obs[iXYZd,1],x_val[0],x_val[-1],y_obs[iXYZd,1],y_val[0],y_val[-1],z_obs[iXYZd,1],z_val[0],z_val[-1])
                        # Observation matrix
                        H_sparseD[iXYZd,observed_nodes[0]]=np.reshape(p000d,(len(p000d),1))
                        H_sparseD[iXYZd,observed_nodes[1]]=np.reshape(p100d,(len(p100d),1))
                        H_sparseD[iXYZd,observed_nodes[2]]=np.reshape(p010d,(len(p010d),1))
                        H_sparseD[iXYZd,observed_nodes[3]]=np.reshape(p110d,(len(p110d),1))
                        H_sparseD[iXYZd,observed_nodes[4]]=np.reshape(p001d,(len(p001d),1))
                        H_sparseD[iXYZd,observed_nodes[5]]=np.reshape(p101d,(len(p101d),1))
                        H_sparseD[iXYZd,observed_nodes[6]]=np.reshape(p011d,(len(p011d),1))
                        H_sparseD[iXYZd,observed_nodes[7]]=np.reshape(p111d,(len(p111d),1))
    print('Time to create trilinear observation_matrix for a single cycle [s]: '+str(time.time()-T1))
    # Convert to CSR
    if Nd == 1:
        H_sparse = H_sparse.tocsr()   
    elif Nd == 2:
        H_sparse = (H_sparseA-H_sparseD).tocsr()   
    return H_sparse,x_scatter,y_scatter,z_scatter


def trilinear_partials(x,x0,x1,y,y0,y1,z,z0,z1):
    '''
    Purpose: computes the partials for each of the 8 nodes surrounding a single observation point in a 3D model. Multiple observations may be analyzed.
    [x,y,z] = observation point
    [x0,y0,x1,y1,z0,z1] = 8 surrounding nodes
    p000,p100,p010,p110,p001,p101,p011,p111 = weighted distance (partials) with the last three numbers indicating the nodes associated with the partials
 
        http://paulbourke.net/miscellaneous/interpolation/
    '''
    xh = np.divide(np.subtract(x,x0),np.subtract(x1,x0)) #(xi-x0)/(x1-x0)
    yh = np.divide(np.subtract(y,y0),np.subtract(y1,y0)) #(yi-y0)/(y1-y0)
    zh = np.divide(np.subtract(z,z0),np.subtract(z1,z0)) #(zi-z0)/(z1-z0)
    xl = 1.0-xh
    yl = 1.0-yh
    zl = 1.0-zh
    p000 = xl*yl*zl 
    p100 = xh*yl*zl
    p010 = xl*yh*zl
    p110 = xh*yh*zl
    p001 = xl*yl*zh
    p101 = xh*yl*zh
    p011 = xl*yh*zh
    p111 = xh*yh*zh
    return p000,p100,p010,p110,p001,p101,p011,p111


def trilinear_forward(x,x0,x1,y,y0,y1,z,z0,z1,B):
    '''
    Purpose: uses least square to determine the value of a point within a 3D grid
    [x,x0,x1,y,y0,y1,z,z0,z1]: same definition as trlinear_partials
    B = node values
    Best = estimated value at point [x,y,z]
    '''
    B = np.asarray(B)
    N = np.shape(x)[0]
    Best = np.empty((N))
    for ii in np.arange(N):
        xi = x[ii]
        yi = y[ii]
        zi = z[ii]
        p3D = trilinear_partials(xi,x0,x1,yi,y0,y1,zi,z0,z1)
        Best[ii] = np.dot(B,p3D)
    return Best


def trilinear_interpolation(x_obs,y_obs,z_obs,x_grid,y_grid,z_grid,B):
    '''
    Purpose: same as  trilinear_interpolation_sp_griddata
    unused, but kept as alternative method
    '''
    Nr,Nc,Nd = np.shape(B)
    B_obs = np.empty((np.shape(x_obs)[0]))*np.nan
    ibinX = np.asarray([0,1,0,1,0,1,0,1])
    ibinY = np.asarray([0,0,1,1,0,0,1,1])
    ibinZ = np.asarray([0,0,0,0,1,1,1,1])
    for zz in np.arange(Nd-1):
        z0 = z_grid[zz]
        z1 = z_grid[zz+1]
        iZ = zz+ibinZ
        for yy in np.arange(Nc-1):
            y0 = y_grid[yy]
            y1 = y_grid[yy+1]
            iY = yy+ibinY
            for xx in np.arange(Nr-1):
                x0 = x_grid[xx]
                x1 = x_grid[xx+1] 
                iX = xx+ibinX
                ixyz = np.where((x_obs>x0) & (x_obs<=x1) & (y_obs>y0) & (y_obs<=y1) & (z_obs>z0) & (z_obs<=z1))[0]
                if np.size(ixyz)!=0:
                    B10 = np.copy(B[iX,iY,iZ])
                    xi = x_obs[ixyz]
                    yi = y_obs[ixyz]
                    zi = z_obs[ixyz]
                    B_obs[ixyz] = trilinear_forward(xi,x0,x1,yi,y0,y1,zi,z0,z1,B10)
    return B_obs


def trilinear_interpolation_sp_griddata(x_obs,y_obs,z_obs,x_grid,y_grid,z_grid,B):
    '''
    Purpose: uses least square to determine the value of all points within a 3D grid/model
    [x_obs,y_obs,z_obs,x_grid,y_grid,z_grid]: same definition as  trilinear_observation_matrix
    B = grid/model containing all node values
    B_obs_gd = estimated value for each observation
    '''
    #t0 = time.time()
    B_mask = np.ma.masked_invalid(B)
    mx,my,mz,x_scatter,y_scatter,z_scatter = lgg.parameter_grid(x_grid,y_grid,z_grid)
    x_ma = mx[~B_mask.mask]
    y_ma = my[~B_mask.mask]
    z_ma = mz[~B_mask.mask]
    B_ma = B_mask[~B_mask.mask]
    # B_obs_gd =interpolate.griddata((y_ma, x_ma), np.ravel(B_ma,order='F'),(y_obs,x_obs),method='linear')
    B_obs_gd =interpolate.griddata((y_ma, x_ma, z_ma), np.ravel(B_ma,order='F'),(y_obs,x_obs,z_obs),method='linear') #B_ma.ravel()
    #print('time bilineary interpolate '+str(np.shape(x_obs)[0])+' measurements: '+str(np.round(time.time()-t0,3)))
    return B_obs_gd








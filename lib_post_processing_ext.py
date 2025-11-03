#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 13:22:18 2020

@author: alexaputnam

Author: Alexa Angelina Putnam Shilleh
Institution: University of Colorado Boulder
Software: sea state bias model generator (interpolation method)
Thesis: Sea State Bias Model Development and Error Analysis for Pulse-limited Altimetry
Semester/Year: Fall 2021
"""
import time
import numpy as np
from netCDF4 import Dataset

import lib_dict_ext as ldic
import lib_geo_gridding_ext as lgrid
import lib_data_ext as ldata
import lib_trilinear_interpolation_ext as lti
import lib_bilinear_interpolation_ext as lbi
import lib_estimation_ext as lest
import lib_filter_ext as lflt
# --------------------------------------------------------------------------#
#-------------------zero_significance---------------------------------------#
# --------------------------------------------------------------------------#
def funcfit(x,y,deg):
    '''
    Purpose: Fit a polynomial to the (x,y) dataset 
    x = independent vairable array
    y = dependent variable array
    deg = degree of polynomial fit (deg can equal 1,2,3 or 4)
    ce = fitting coefficients
    y_est = estimated value for y
    '''
    Nl2 = np.shape(x)[0]
    H = np.ones((Nl2,deg+1))
    if deg>=1:
        H[:,1] = x
    if deg>=2:
        H[:,2] = x**2
    if deg>=3:
        H[:,3] = x**3
    if deg>=4:
        H[:,4] = x**4
    ce = np.linalg.inv(H.T.dot(H)).dot(H.T.dot(y))
    if deg>=1:
        y_est=ce[0]+ce[1]*x
    if deg>=2:
        y_est+=ce[2]*x**2
    if deg>=3:
        y_est+=ce[3]*x**3
    if deg>=4:
        y_est+=ce[4]*x**4
    return ce,y_est

def hist_res():
    '''
    Purpose: resolution of the histogram bins for determining the zero-signicance
    dtec,dmet,dssb = resolution for a TECu histogram, ionosphere correction histogram and SSB histogram, respectively.
    '''
    dIku = 0.0001
    dtec = np.round((dIku*10.)/0.002187,4) #=0.0457272*10.
    dmet = dtec*0.01216
    dssb = dIku*10.
    return dtec,dmet,dssb

def zerosig_hist(x,VAR):
    # Purpose: derive the histogram (hist), CDF (cdf) and corresponding bin edges (bin_edges) and bin midpoints (bin_mid) from an array of data (x) corresponding to variable (VAR)
    dtec,dmet,dssb = hist_res()
    if VAR == 'iono':
        bins_array = np.arange(-0.2,0.2,dmet) 
    elif VAR in ['usla','ssha_dfI','rng_mle','iono_corr_alt', 'mss_dtu','mss_cnes15', 'ocean_got', 'wet_tropo_model','sum_err','ib_corr','alt_jpl','alt_cnes','iono_tran','iono_tran_smooth']:
        bins_array = np.arange(-0.5,0.5,dssb) 
    elif VAR == 'tec':
        bins_array = np.arange(-100.,100.,dtec) 
    else:
        raise('no defined bins for '+VAR)
    hist,bin_edges = np.histogram(x,bins=bins_array)
    db = np.diff(bin_edges)[0]
    bin_mid = bin_edges[:-1]+db/2
    cdf = np.cumsum(hist)# plt.plot(bin_mid,hist)
    return hist,cdf,bin_edges,bin_mid



def zero_xing(rts,bins,VARLEV):
    '''
    Purpose: Determine the zero-significance value (rt) given the roots of a polynomial (rts) and CDF bins that correspond to the cutoff region (bins). rt is the zero percentile crossing.
    VARLEV = leveling variable (‘tec’ for TECu)
    '''
    if VARLEV == 'tec':
        bin_i = bins[0]
    else:
        bin_i = bins[-1]
    img = np.imag(rts)
    rl = np.real(rts)
    img_i = np.where(img==0)[0]
    if np.size(img_i) == 1:
        if np.abs(rl[img_i]-bin_i)<=0.01:
            idx = img_i
        else:
            rts_min = np.abs(rl-bin_i)
            idxp = np.where(rts_min==np.min(rts_min))[0]
            if np.size(idxp)>1:
                idx = idxp[0]
            else:
                idx=idxp 
    else:
        rts_min = np.abs(rl-bin_i)
        idxp = np.where(rts_min==np.min(rts_min))[0]
        if np.size(idxp)>1:
            idx = idxp[0]
        else:
            idx=idxp
    rt = rl[idx]
    return rt 



def find_zerosig(hist_x,bin_mid,cutoff,deg_in,XYflip,VARLEV='met',LINEAR=False,MP=False):
    '''
    Purpose: find the zero-significance value of an SSB model
    cutoff = percentile segment for polynomial fit (see dissertation)
    deg_in = degree of polynomial fit (automatically adjusted to 1 if LINEAR = True)
    XYflip = True or False whether the x-axis of the CDF is used as the dependent variable for the polynomial fit (True for SSB)
    VARLEV = leveling variable (‘met’ for SSB)
    LINEAR = True or False to apply a linear fit to the cutoff region (False for SSB)
    MP = True or False to extrapolate to midpoint rather than zero-significance (Default to False)
    zero_sig = zero-significance value 
    cdf_out = CDF of TECu cutoff region
    poly_fit = polynomisl fit to TECu CDF cutoff region
    bin_out = CDF independent variable bins corresponding to poly_fit
    Vres = mean square error of the fit 
    '''
    # Determine zero-significance value
    # CDF in %
    if LINEAR is False:
        deg = deg_in
    elif LINEAR is True:
        deg = 1
    min_pnt = deg+1
    cdf = np.cumsum(hist_x)
    cdf_nrm = 100.*(cdf/float(np.nanmax(cdf)))
    # Orient CDF so that the region of interest is in the lower percentile
    if VARLEV == 'tec':
        cdf_rev = np.copy(cdf_nrm)
    else:
        cdf_rev = 100.-cdf_nrm
    # PAR1: Constrict CDF to region of interest (determined by 'cutoff')
    preidx = np.where((cdf_rev>=cutoff[0]) & (cdf_rev<=cutoff[1]))[0]     
    cdf_chop = np.take(cdf_rev,preidx)
    cdf_diff = np.diff(cdf_chop)
    # PAR2: make sure that there is a change in percentile from bin-to-bin
    idx_no0p = np.where(abs(cdf_diff)>0.0)[0]
    # Call indices that meet the PAR1 and PAR2 requirements
    if VARLEV == 'tec':
        idx = np.take(preidx,idx_no0p+1) 
    else:
        idx = np.take(preidx,idx_no0p)
    print('number of points in hist to fit zero-sig for model bias '+str(np.size(idx)))
    # Iterative fit (4-sigma)
    if np.size(idx) != 0:
        Nsig = np.shape(idx)[0]
        isig = np.arange(Nsig)
        Nres = -1
        res=100
        iout = 0
        while Nres!=Nsig:
            if iout>0:
                print('iterative fit. iter # = '+str(iout))
                # Only iterate until there are at least half the amount of data points that 
                #   were started with. If more points need to be removed then the 
                #   CDF slope is poorly conditioned within the cutoff.
            if Nsig >= min_pnt: #np.shape(idx)[0]/2.:
                if XYflip == False:
                    poly_coef,poly_fit = funcfit(bin_mid[idx[isig]],cdf_rev[idx[isig]],deg)
                    res = cdf_rev[idx[isig]]-poly_fit
                elif XYflip ==True:
                    poly_coef,poly_fit = funcfit(cdf_rev[idx[isig]],bin_mid[idx[isig]],deg)
                    res = bin_mid[idx[isig]]-poly_fit
                Nres = np.size(res)
                sig = np.std(res)
                sig4 = sig*4.
                mn_res = np.mean(res[~np.isnan(res)])
                isig = np.where((res<mn_res+sig4)&(res>mn_res-sig4))[0]
                Nsig = np.size(isig)
                iout+=1
                if LINEAR is False:
                    if XYflip == False:
                        poly_coef_rev = poly_coef[::-1]
                        rts = np.roots(poly_coef_rev)
                        zero_sig = zero_xing(rts,bin_mid[idx[isig]],VARLEV) #rts[0]
                    elif XYflip == True:
                        zero_sig = poly_coef[0] 
                    cdf_out = cdf_rev[idx[isig]]
                    bin_out = bin_mid[idx[isig]]
                elif LINEAR is True:
                    if MP is False:
                        midCO = 0.0
                    elif MP is True:
                    	midCO = np.sum(cutoff)/2.
                    if XYflip == False:
                        y_est=(midCO-poly_coef[0])/poly_coef[1]
                    elif XYflip == True:
                        y_est=poly_coef[0]+poly_coef[1]*midCO
                        #print('y_est: '+str(y_est))
                        #print('midCO: '+str(midCO))
                        #print('poly_coef '+str(poly_coef))
                    zero_sig = np.copy(y_est)
                    cdf_out = cdf_rev[idx[isig]]
                    bin_out = bin_mid[idx[isig]]
                # This might require one less indentation
                if Nres == 0:
                    print('WARNING: Nres==0')
                Vres = np.nansum(np.square(res))/float(Nres)
            else:
                print('Poorly conditioned histogram')
                Nres = 0
                Nsig = 0
                zero_sig = np.nan
                cdf_out = np.nan
                bin_out = np.nan
                poly_fit=np.nan
                Vres = np.nan
    else:
        zero_sig = np.nan
        cdf_out = np.nan
        bin_out = np.nan
        poly_fit=np.nan
        Vres = np.nan
    '''
    plt.figure()
    plt.plot(bin_mid[idx[isig]],cdf_rev[idx[isig]])
    plt.plot(poly_fit,cdf_rev[idx[isig]])
    '''
    return zero_sig,cdf_out,poly_fit,bin_out,Vres

def model_level(x_obs,y_obs,z_obs,x_grid,y_grid,z_grid,B,VAR,cutoff,deg):
    '''
    Purpose: evaluate the raw SSB solution and provide the zero-significance value for a single cycle (zsi)
    x_obs,y_obs,z_obs = independent variable observations
    x_grid,y_grid,z_grid = model parameters
    B = raw SSB modeling
    cutoff = percentile segment for polynomial fit (see dissertation)
    deg = degree of polynomial fit
    '''
    if np.size(z_grid) == 0:
        ssb_est = lbi.bilinear_interpolation_sp_griddata(x_obs,y_obs,x_grid,y_grid,B)
    else:
        ssb_est = lti.trilinear_interpolation_sp_griddata(x_obs,y_obs,z_obs,x_grid,y_grid,z_grid,B)
    zval = 4
    ssb_new,iout,ikp = ldata.iterative_outlier_removal(ssb_est,zval)
    hist_x,cdf_ssb,bin_edges,bin_mid = zerosig_hist(ssb_new,VAR)
    XYflip = True
    print('cutoff / deg / XYflip :'+str(cutoff)+' / '+str(deg)+' / '+str(XYflip))
    zsi,cdf_out,poly_fit,bin_out,Vres = find_zerosig(hist_x,bin_mid,cutoff,deg,XYflip)
    return zsi


def run_model_level(MISSIONNAME,METHOD,VAR,SOURCE,BAND,CYCLES,DIM,OMIT,B,FE,cutoff,deg,BAND2,ERR,SSBk,SSBc,Irel,WEIGHT,OPEN_OCEAN):
    '''
    Purpose: run function for model_level to obtain all zero-significant values for each cycle and calculate the model bias
    B = raw SSB modeling
    FE = SSB model formal error
    cutoff = percentile segment for polynomial fit (see dissertation)
    deg = degree of polynomial fit
    zerosig = array containing the zero-significance values for each cycle
    zerosig_mean = SSB model bias (= mean of zerosig)
    model_bias =  mean usla over all cycles
    mean_ssb_hdp_cyc = array containing the mean usla for each cycle
    '''
    srt1 = time.time()
    DIMND,OMITlab,MODELFILE = ldic.file_labels(VAR,BAND,DIM,OMIT)
    Ncyc = np.shape(CYCLES)[0]
    x_grid,y_grid,z_grid = lgrid.grid_dimensions(DIM,OMIT)
    mx,my,mz,x_scatter,y_scatter,z_scatter = lgrid.parameter_grid(x_grid,y_grid,z_grid)
    zerosig= np.empty(Ncyc)*np.nan
    mean_ssb_hdp_cyc = np.empty(Ncyc)*np.nan
    for ii in np.arange(Ncyc):
        srt2 = time.time()
        CYC1 = str(CYCLES[ii])
        CYC2 = str(CYCLES[ii])
        FILE1,FILE2,FILE2_GEO = ldic.filenames_specific(MISSIONNAME,'dir',SOURCE,CYC1,CYC2,BAND,DIMND,ERR=ERR)
        ds = Dataset(FILE1).variables
        usla,inonan,inan = lflt.observables(METHOD,VAR,BAND2,ds,SSBk,SSBc,Irel,ERR,MISSIONNAME,OPEN_OCEAN)
        x_obsi,y_obsi = ldata.pull_observables(ds,DIM,ERR,MISSIONNAME)
        x_obs = np.take(x_obsi,inonan)
        y_obs = np.take(y_obsi,inonan)
        lat_obs= ds['lat'][:][inonan]
        if DIM=='u10_square':
            xi,yi=x_grid[3],y_grid[4]
        else:
            xi,yi=x_grid[7],y_grid[22]
        idx = np.where((x_obs>=xi-0.125)&(x_obs<xi+0.125)&(y_obs>=yi-0.125)&(y_obs<yi+0.125))[0]
        if np.size(idx)!=0:
            uslaidx = np.take(usla[inonan],idx)
            latidx=np.take(lat_obs,idx)
            if WEIGHT == True:
                wgts = lest.latitude_weight(latidx,MISSIONNAME)
            else:
                wgts = np.ones(np.shape(uslaidx))
            mean_ssb_hdp_cyc[ii] = np.sum(uslaidx*wgts)/np.sum(wgts)
    model_bias = np.mean(mean_ssb_hdp_cyc[~np.isnan(mean_ssb_hdp_cyc)])
    print('model bias = '+str(model_bias))
    if VAR in ['usla','ssha_dfI','rng_mle','iono_corr_alt', 'mss_dtu','mss_cnes15', 'ocean_got', 'wet_tropo_model','sum_err','ib_corr','alt_jpl','alt_cnes','iono_tran','iono_tran_smooth']:
        for ii in np.arange(Ncyc):
            srt2 = time.time()
            CYC1 = str(CYCLES[ii])
            CYC2 = str(CYCLES[ii])
            FILE1,FILE2,FILE2_GEO = ldic.filenames_specific(MISSIONNAME,METHOD,SOURCE,CYC1,CYC2,BAND,DIMND,ERR=ERR)
            ds = Dataset(FILE1).variables
            x_obsi,y_obsi = ldata.pull_observables(ds,DIM,ERR,MISSIONNAME)
            if METHOD == 'dir':
                x_obs = np.copy(x_obsi)
                y_obs = np.copy(y_obsi)
                if np.size(OMIT) == 0:
                    z_obs = []
                else:
                    z_obs = ds[OMIT][:]
            else:
                x_obs = x_obsi[:,0]
                y_obs = y_obsi[:,0]
                if np.size(OMIT) == 0:
                    z_obs = []
                else:
                    z_obs = ds[OMIT][:][:,0]
            Npts = np.shape(x_obs)[0]
            if np.size(OMIT) == 0:
                zerosig[ii] =model_level(x_obs,y_obs,[],x_grid,y_grid,z_grid,B,VAR,cutoff,deg) #model_level(x_obs[idx],y_obs[idx],[],x_grid,y_grid,z_grid,B,OMIT,cutoff,deg)
            else:
                zerosig[ii] =model_level(x_obs,y_obs,z_obs,x_grid,y_grid,z_grid,B,VAR,cutoff,deg)#model_level(x_obs[idx],y_obs[idx],z_obs[idx],x_grid,y_grid,z_grid,B,OMIT,cutoff,deg)
            print('zero-significance from cycle '+CYC1+' = '+str(zerosig[ii]))
            print('Time to estimate zero-significance [using '+str(Npts)+' random points] from cycle '+CYC1+' [s]: '+str(time.time()-srt2))
        zs_new,iout,ikp = ldata.iterative_outlier_removal(zerosig[~np.isnan(zerosig)],4)
        zerosig_mean = np.mean(zerosig[~np.isnan(zerosig)][ikp])
        print('zerosig model bias= '+str(np.round(zerosig_mean,4))+' ,std = '+str(np.round(np.std(zerosig[~np.isnan(zerosig)][ikp]),4)))
    else:
        zerosig_mean = model_bias
    print('Time to estimate mean zero-significance [s]: '+str(time.time()-srt1))
    #zerosig_mean = B[7,22]-(-0.05)
    return zerosig,zerosig_mean,model_bias,mean_ssb_hdp_cyc

# --------------------------------------------------------------------------#
#-------------------smoothing---------------------------------------#
# --------------------------------------------------------------------------#
def cube_nd(fx,fy,fz,fB,Nd):
    '''
    Purpose: cubic fit parametric SSB model solution given the leveled, raw SSB model
    fx,fy,fz  = collapsed coordinate matrices
    fB = collapsed leveled, raw SSB model
    Nd = model dimensions (nominal: Nd=2 for 2D)
    ce = fitting coefficients of parametric modeling
    fB_fit = collapsed parametric SSB model (fills in all grid nodes)
    fB_comb = combination of leveled, raw SSB model and parametric SSB model. All observed nodes are equal to the leveled, raw SSB solution, while all unobserved nodes are equal to the parametric solution.
    '''
    inn = np.where(~np.isnan(fB))[0]
    N = np.shape(inn)[0]
    H = np.ones((N,(Nd*3)+1))
    H[:,0] = fx[inn]
    H[:,1] = np.square(fx[inn])
    H[:,2] = np.power(fx[inn],3)
    H[:,3] = fy[inn]
    H[:,4] = np.square(fy[inn])
    H[:,5] = np.power(fy[inn],3)
    H[:,6] = fx[inn]*fy[inn]
    if Nd == 3:
        H[:,7] = fz[inn]
        H[:,8] = np.square(fz[inn])
        H[:,9] = np.power(fz[inn],3)
    ce = np.linalg.inv(H.T.dot(H)).dot(H.T.dot(fB[inn]))
    fB_fit = (ce[0]*fx)+(ce[1]*fx**2)+(ce[2]*fx**3)+(ce[3]*fy)+(ce[4]*fy**2)+(ce[5]*fy**3)+(ce[6]*(fx*fy))
    if Nd == 3:
        fB_fit = fB_fit+(ce[6]*fz)+(ce[7]*fz**2)+(ce[8]*fz**3)
    fB_comb = np.copy(fB_fit)
    fB_comb[inn] = fB[inn]
    return ce,fB_fit,fB_comb


def putnam_smoothing(B,FE,mx,my,mz): 
    '''
    Purpose: The name is a bit obnoxious (I used it when I was testing multiple methods). This function produces the leveled and smoothed final SSB model solution (B_smooth). See dissertation for more details on this step.
    B = leveled, raw SSB model 
    FE = formal error of the raw SSB model
    mx,my,mz = coordinate matrices
    ce = fitting coefficients of parametric modeling
    '''
    order = 'F'
    Nd = np.size(B.shape)
    fx,fy,fB,fFE = mx.flatten(order),my.flatten(order),B.flatten(order),FE.flatten(order)
    if Nd == 3:
        fz = mz.flatten(order)
    elif Nd == 2:
        fz=[]
    else:
        raise('putnam_smoothing/lib_BI_smooth only made for 2-D or 3-D, this has '+str(Nd)+' dimensions')
    # Find deltas
    dx = np.diff(np.unique(mx))[0]
    dy = np.diff(np.unique(my))[0]
    
    # Fit: linear (9 cm rms worse than quad), quad (1 cm worse than cubic)
    # Cubic 3D fit
    ce,fB_fit,fB_comb=cube_nd(fx,fy,fz,fB,Nd)
    fB_smooth = np.copy(fB_comb)
    fB_smooth_raw = np.copy(fB)#test variable
    inn = np.where(~np.isnan(fB))[0]
    
    fFEnn2 = 1./np.square(np.copy(fFE[inn]))
    fBnn = np.copy(fB[inn])
    fB_fit_nn = np.copy(fB_fit[inn])
    fFEnn = np.copy(fFE[inn])
    fB_smooth_nn = np.copy(fB[inn])*np.nan
    fFE_smooth_nn = np.copy(fFE[inn])*np.nan
    fB_smooth_raw_nn = np.copy(fFE[inn])*np.nan#test variable
    fxnn = np.copy(fx[inn])
    fynn = np.copy(fy[inn])
    sxx = np.subtract.outer(fxnn,fxnn)/dx
    syy = np.subtract.outer(fynn,fynn)/dy
    if Nd == 3:
        dz = np.diff(np.unique(mz))[0]
        fznn = np.copy(fz[inn])
        szz = np.subtract.outer(fznn,fznn)/dz
        r = np.sqrt(np.square(sxx)+np.square(syy)+np.square(szz))
    elif Nd ==2:
        r = np.sqrt(np.square(sxx)+np.square(syy))
    wgt_dst = np.exp(-r**2) #1./r # np.exp(-r/wgt_err) #
    N = np.shape(inn)[0]
    for ii in np.arange(N):
        wgt_data = np.multiply(fFEnn2,wgt_dst[ii,:]) #wgt_dst[ii,:] #
        fFE_smooth_nn[ii] = np.sum(np.multiply(wgt_data,fFEnn))/np.sum(wgt_data) # tells the story of the data
    for ii in np.arange(N):
        wgt_data = np.multiply(fFEnn2,wgt_dst[ii,:])
        wgt_fit = fFE_smooth_nn[ii]/np.nanmax(fFE_smooth_nn)
        fB_smooth_data = np.sum(np.multiply(wgt_data,fBnn))/np.sum(wgt_data)
        fB_smooth_nn[ii] = ((1.-wgt_fit)*fB_smooth_data)+(wgt_fit*fB_fit_nn[ii])
        fB_smooth_raw_nn[ii] = fB_smooth_data#test variable
    fB_smooth[inn] = fB_smooth_nn
    B_smooth = np.reshape(fB_smooth,np.shape(B),order=order)
    fB_smooth_raw[inn] = fB_smooth_raw_nn#test variable
    #B_intermed = np.reshape(fB_smooth_raw,np.shape(B),order=order)#test variable
    #crt_lut2ssb_intermediate(mx,my,B_intermed)
    return B_smooth,ce



#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
@author: alexaputnam

Author: Alexa Angelina Putnam Shilleh
Institution: University of Colorado Boulder
Software: sea state bias model generator (interpolation method)
Thesis: Sea State Bias Model Development and Error Analysis for Pulse-limited Altimetry
Semester/Year: Fall 2021
"""


import numpy as np

def derive_fit(x,swh):
    db=0.25
    swh_bins = np.arange(0,12+db,db)
    N = np.shape(swh_bins)[0]
    mn = np.empty(N)*np.nan
    sd3 = np.empty(N)*np.nan
    for ii in np.arange(N):
        idx = np.where((swh>=swh_bins[ii]-(db/2.0))&(swh<swh_bins[ii]+(db/2.0)))[0]
        if np.size(idx)>50:
            mn[ii] = np.nanmean(x[idx])
            sd3[ii] = 3.0*np.nanstd(x[idx])
    i0 = np.where(~np.isnan(mn+sd3))[0]
    i1 = np.where(~np.isnan(mn+sd3)&(swh_bins<=2))[0]
    i2 = np.where(~np.isnan(mn+sd3)&(swh_bins>2))[0]
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    ce0 = np.polyfit(swh_bins[i0],(mn+sd3)[i0], 2)[::-1]
    print('ce0 :'+str(ce0))
    ce1 = np.polyfit(swh_bins[i1],(mn+sd3)[i1], 3)[::-1]
    print('ce1 :'+str(ce1))
    ce2 = np.polyfit(swh_bins[i2],(mn+sd3)[i2], 1)[::-1]
    print('ce2 :'+str(ce2))
    ce0n = np.polyfit(swh_bins[i0],(mn-sd3)[i0], 2)[::-1]
    print('ce0 negative:'+str(ce0n))
    ce1n = np.polyfit(swh_bins[i1],(mn-sd3)[i1], 3)[::-1]
    print('ce1 negative:'+str(ce1n))
    ce2n = np.polyfit(swh_bins[i2],(mn-sd3)[i2], 1)[::-1]
    print('ce2 negative:'+str(ce2n))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    return swh_bins,mn,sd3

def ce2est(ce,x,EXP):
    # Purpose: evaluate dependent variable (y) given the fitting coefficients (ce), dependent variable (x) and indication for exponential or non-exponential fit (EXP = True or False)
    Nce = np.shape(ce)[0]
    if EXP is False:
        if Nce == 2:
            y = ce[0]+(ce[1]*x)
        elif Nce == 3:
            y = ce[0]+(ce[1]*x)+(ce[2]*(x**2))
        elif Nce == 4:
            y = ce[0]+(ce[1]*x)+(ce[2]*(x**2))+(ce[3]*(x**3))
        elif Nce == 5:
            y = ce[0]+(ce[1]*x)+(ce[2]*(x**2))+(ce[3]*(x**3))+(ce[4]*(x**4))
    else:
        y = ce[0]+(ce[1]*np.exp(-x/ce[2]))
    return y


def plyfit(ce1,ce2,x,y_in,celim,EXP):
    '''
    Purpose: use ce2est to evaluate y (y_est), take the difference to the true value of y (y_in) to y_est (dy), create threshold envelope (var_bin) 
       and corresponding independent variable (x) bins (par_bin). This is used for creating the dynamic thresholds. 
    ce1 = coefficients for first fitted portion of par_bin or entire par_bin
    ce2 = coefficients for second fitted portion of par_bin or may be empty if ce1 is used for entire fit.
    celim = cutoff point along var_bin that separate first portion fit (ce1) from second portion fit (ce2). May be empty if ce1 provides entire fit.
    par_bin =  independent variable (x) bins
    '''
    par_bin = np.arange(0,12.75,0.25)
    var_bin = np.empty(np.shape(par_bin))*np.nan
    if np.size(ce2)!=0:
        y_est = np.empty(np.shape(x))*np.nan
        idce1= np.where(x<=celim)[0]
        idce2= np.where(x>celim)[0]
        y_est[idce1] = ce2est(ce1,x[idce1],EXP)
        y_est[idce2] = ce2est(ce2,x[idce2],EXP)
        ibin1= np.where(par_bin<=celim)[0]
        ibin2= np.where(par_bin>celim)[0]
        var_bin[ibin1]=ce2est(ce1,par_bin[ibin1],EXP)
        var_bin[ibin2]=ce2est(ce2,par_bin[ibin2],EXP)
    else:
        y_est = ce2est(ce1,x,EXP)
        var_bin = ce2est(ce1,par_bin,EXP)
    dy = y_est-y_in
    return y_est,dy,var_bin,par_bin


def dynamic_threshold_sa(rms,swh,TH):
    # Purpose: provides indices for S6 swh rms, range rms, sig0 rms and off-nadir angle that fall within the dynamic threshold envelope.
    Nd = np.size(np.shape(swh))
    EXP=False
    CBAND=False
    if TH == 'swh_rms_ku':
        ce1 = np.asarray([0.4203, 0.0483, 0.0024]) #[0,2],[2,12.5]
        ce2 = []
        celim = []
    elif TH == 'range_rms_ku':
        ce1 = np.asarray([ 0.0445,  0.0217, -0.0001])#np.asarray([8.526e-02,  1.065e-02, -1.600e-04, -1.000e-05]) #[0,2],[2,12.5]
        ce2 = []
        celim=[]
    if TH in ['swh_rms_ku','range_rms_ku']:
        if Nd == 1:
            rms_th,drms,var_bin,par_bin = plyfit(ce1,ce2,swh,rms,celim,EXP)
            if 'swh' in TH:
                ikp = np.where((drms>=0)&(rms>=0.05))[0]
            else:
                ikp = np.where((drms>=0))[0]
        else:
            rms_th0,drms0,var_bin,par_bin = plyfit(ce1,ce2,swh[:,0],rms[:,0],celim,EXP)
            rms_th1,drms1,var_bin,par_bin = plyfit(ce1,ce2,swh[:,1],rms[:,1],celim,EXP)
            if 'swh' in TH:
                ikp = np.where((drms0>=0)&(drms1>=0)&(rms>=0.05))[0]
            else:
                ikp = np.where((drms0>=0)&(drms1>=0))[0]
    else:
        ikp = np.arange(np.shape(swh)[0])
    return ikp
'''
def dynamic_threshold_sw(rms,swh,TH):
    # Purpose: provides indices for S6 swh rms, range rms, sig0 rms and off-nadir angle that fall within the dynamic threshold envelope.
    Nd = np.size(np.shape(swh))
    EXP=False
    CBAND=False
    if TH == 'swh_rms_ku':
        ce1 = np.asarray([ 0.398,  0.305, -0.15 ]) #[0,2],[2,12.5]
        ce2 = np.asarray([ 0.44, 0.02,  0.004]) #
        celim = 1.75 #[0,2],[2,12.5]
    elif TH == 'range_rms_ku':
        ce1 = np.asarray([0.06, 0.004, 0.0011])#np.asarray([8.526e-02,  1.065e-02, -1.600e-04, -1.000e-05]) #[0,2],[2,12.5]
        ce2 = []
        celim=[]
    elif TH == 'sig0_rms_ku':#0.24+0.4*np.exp(-swhC_bins_ex/1.20)#
        ce1 = np.asarray([ 0.4  , -0.091])#[ 0.248, -0.017,  0.002]
        ce2=np.asarray([ 0.248, -0.017,  0.002])
        celim=2.0
    elif TH == 'range_rms_c':
        ce1 = np.asarray([ 0.145, -0.023,  0.015])#
        ce2=np.asarray([0.1  , 0.029])
        celim=2.0
    elif TH == 'swh_rms_c':
        ce1 = np.asarray([1.177, 0.039, 0.011])#
        ce2=[]
        celim=[]
    elif TH == 'sig0_rms_c':#0.24+0.4*np.exp(-swhC_bins_ex/1.20)#
        ce1 = np.asarray([ 0.301, -0.128,  0.028])
        ce2 = np.asarray([ 0.171, -0.008 ,  0.0008])
        celim=2
    elif TH == 'off_nadir_angle_wf_ku':
        ce1 = np.asarray([0.2,0])
        ce2=[]
        ce1b = np.asarray([0.10,0])
        ce2b=[]
        celim=[]
    if Nd == 1:
        rms_th,drms,var_bin,par_bin = plyfit(ce1,ce2,swh,rms,celim,EXP)
        if 'swh' in TH:
            ikp = np.where((drms>=0)&(rms>=0.05))[0]
        elif 'off_nadir_angle_wf_ku' in TH:
            rms_thb,drmsb,var_bin,par_bin = plyfit(ce1b,ce2b,swh,rms,celim,EXP)
            ikp = np.where((drms>=0)&(drmsb<=0))[0]
        else:
            ikp = np.where((drms>=0))[0]
    else:
        rms_th0,drms0,var_bin,par_bin = plyfit(ce1,ce2,swh[:,0],rms[:,0],celim,EXP)
        rms_th1,drms1,var_bin,par_bin = plyfit(ce1,ce2,swh[:,1],rms[:,1],celim,EXP)
        if 'swh' in TH:
            ikp = np.where((drms0>=0)&(drms1>=0)&(rms>=0.05))[0]
        elif 'off_nadir_angle_wf_ku' in TH:
            rms_thb0,drmsb0,var_bin,par_bin = plyfit(ce1b,ce2b,swh[:,0],rms[:,0],celim,EXP)
            rms_thb1,drmsb1,var_bin,par_bin = plyfit(ce1b,ce2b,swh[:,1],rms[:,1],celim,EXP)
            ikp = np.where((drms0>=0)&(drmsb0<=0)&(drms1>=0)&(drmsb1<=0))[0]
        else:
            ikp = np.where((drms0>=0)&(drms1>=0))[0]
    return ikp
'''

def dynamic_threshold_sw(rms,swh,TH):
    # Purpose: provides indices for S6 swh rms, range rms, sig0 rms and off-nadir angle that fall within the dynamic threshold envelope.
    Nd = np.size(np.shape(swh))
    EXP=False
    CBAND=False
    if TH == 'swh_rms_ku':
        ce1 = np.asarray([ 0.63 ,  0.356, -0.132]) #[0,2],[2,12.5]
        ce2 = np.asarray([0.777, 0.003, 0.007]) #
        celim = 1.75
    elif TH == 'range_rms_ku':
        ce1 = np.asarray([0.083, 0.018, 0.   ])#np.asarray([8.526e-02,  1.065e-02, -1.600e-04, -1.000e-05]) #[0,2],[2,12.5]
        ce2 = []
        celim=[]
    elif TH == 'sig0_rms_ku':#0.24+0.4*np.exp(-swhC_bins_ex/1.20)#
        ce1 = np.asarray([0.32, 0.5, 1.7])
        ce2=[]
        celim=[]
        EXP=True
    elif TH == 'range_rms_c':
        ce1 = np.asarray([ 0.2027,  0.0426, -0.0009])#
        ce2=[]
        celim=[]
    elif TH == 'swh_rms_c':
        ce1 = np.asarray([ 1.957,  0.617, -0.266]) #[0,2],[2,12.5]
        ce2 = np.asarray([2.161, 0.043, 0.009]) #
        celim = 1.5
    elif TH == 'sig0_rms_c':#0.24+0.4*np.exp(-swhC_bins_ex/1.20)#
        ce1 = np.asarray([0.21, 0.3, 1.65])
        ce2=[]
        celim=[]
        EXP=True
    elif TH == 'off_nadir_angle_wf_ku':
        ce1 = np.asarray([ 0.24,0.12,1.0])
        ce2=[]
        ce1b = np.asarray([ 0.045,-0.08,1.0])
        ce2b=[]
        celim=[]
        EXP=True
    if Nd == 1:
        rms_th,drms,var_bin,par_bin = plyfit(ce1,ce2,swh,rms,celim,EXP)
        if 'swh' in TH:
            ikp = np.where((drms>=0)&(rms>=0.05))[0]
        elif 'off_nadir_angle_wf_ku' in TH:
            rms_thb,drmsb,var_bin,par_bin = plyfit(ce1b,ce2b,swh,rms,celim,EXP)
            ikp = np.where((drms>=0)&(drmsb<=0))[0]
        else:
            ikp = np.where((drms>=0))[0]
    else:
        rms_th0,drms0,var_bin,par_bin = plyfit(ce1,ce2,swh[:,0],rms[:,0],celim,EXP)
        rms_th1,drms1,var_bin,par_bin = plyfit(ce1,ce2,swh[:,1],rms[:,1],celim,EXP)
        if 'swh' in TH:
            ikp = np.where((drms0>=0)&(drms1>=0)&(rms>=0.05))[0]
        elif 'off_nadir_angle_wf_ku' in TH:
            rms_thb0,drmsb0,var_bin,par_bin = plyfit(ce1b,ce2b,swh[:,0],rms[:,0],celim,EXP)
            rms_thb1,drmsb1,var_bin,par_bin = plyfit(ce1b,ce2b,swh[:,1],rms[:,1],celim,EXP)
            ikp = np.where((drms0>=0)&(drmsb0<=0)&(drms1>=0)&(drmsb1<=0))[0]
        else:
            ikp = np.where((drms0>=0)&(drms1>=0))[0]
    return ikp

def dynamic_threshold_s6Ah(rms,swh,TH,MN,MODE):
    # Purpose: provides indices for S6 swh rms, range rms, sig0 rms and off-nadir angle that fall within the dynamic threshold envelope.
    Nd = np.size(np.shape(swh))
    EXP=False
    CBAND=False
    if TH == 'swh_rms_ku':
        ce1 = np.asarray([ 0.34926939,  0.65375903, -0.62462621,  0.16648158]) #[0,2],[2,12.5]
        ce2 = np.asarray([0.29143505, 0.0829824 , 0.00285974]) #
        celim = 2.0 #[0,2],[2,12.5]

    elif TH == 'swh_rms_c':
        print(TH+' does not exist')
        CBAND = True
    elif TH == 'range_rms_ku':
        ce1 = np.asarray([0.05087238, 0.00930197, 0.00087736])#np.asarray([8.526e-02,  1.065e-02, -1.600e-04, -1.000e-05]) #[0,2],[2,12.5]
        ce2 = []
        celim=[]
    elif TH == 'range_rms_c':
        print(TH+' does not exist')
        CBAND = True
    elif TH == 'sig0_rms_ku':#0.24+0.4*np.exp(-swhC_bins_ex/1.20)#
        ce1 = np.asarray([0.18, 0.75,  1.2 ])
        ce2=[]
        celim=[]
        EXP=True
    elif TH == 'sig0_rms_c':#0.24+0.4*np.exp(-swhC_bins_ex/1.20)#
        print(TH+' does not exist')
        CBAND = True
    elif TH == 'off_nadir_angle_wf_ku':
        print(TH+' does not exist')
        CBAND = True
    if CBAND == False:
        if Nd == 1:
            rms_th,drms,var_bin,par_bin = plyfit(ce1,ce2,swh,rms,celim,EXP)
            if 'swh' in TH:
                ikp = np.where((drms>=0)&(rms>=0.05))[0]
                rms_thb=[]
            else:
                rms_thb=[]
                ikp = np.where((drms>=0))[0]
        else:
            rms_th0,drms0,var_bin,par_bin = plyfit(ce1,ce2,swh[:,0],rms[:,0],celim,EXP)
            rms_th1,drms1,var_bin,par_bin = plyfit(ce1,ce2,swh[:,1],rms[:,1],celim,EXP)
            if 'swh' in TH:
                ikp = np.where((drms0>=0)&(drms1>=0)&(rms>=0.05))[0]
            else:
                ikp = np.where((drms0>=0)&(drms1>=0))[0]
    else:
        ikp = np.arange(np.shape(swh)[0])
    return ikp

def dynamic_threshold_s6A(rms,swh,TH,MN,MODE):
    # Purpose: provides indices for S6 swh rms, range rms, sig0 rms and off-nadir angle that fall within the dynamic threshold envelope.
    Nd = np.size(np.shape(swh))
    EXP=False
    if TH == 'swh_rms_ku':
        ce1 = np.asarray([ 0.9671,   0.29698, -0.61399,  0.20052]) #[0,2],[2,12.5]
        ce2 = np.asarray([6.9235e-01, 3.7000e-03, 1.7400e-03, 2.1000e-04]) #
        celim = 2.0 #[0,2],[2,12.5]

    elif TH == 'swh_rms_c':
        ce1 = np.asarray([ 4.56473832, -1.37666413,  0.29649385,  0.05553079]) #+0.03
        ce2=np.asarray([ 2.73989977e+00,  3.35650485e-01,  3.48903840e-04, -8.12514739e-04]) #(+0.05)
        celim=2.
    elif TH == 'range_rms_ku':
        ce1 = np.asarray([ 0.08213806,  0.01275537, -0.00015146])#np.asarray([8.526e-02,  1.065e-02, -1.600e-04, -1.000e-05]) #[0,2],[2,12.5]
        ce2 = []
        celim=[]
    elif TH == 'range_rms_c':
        ce1 = np.asarray([ 0.51401327, -0.23079258,  0.14166817, -0.02424609])#[ 0.31506,  0.06474, -0.00365])
        ce2=np.asarray([ 3.03583512e-01,  6.18299699e-02,  5.69544505e-04, -1.87019304e-04])
        celim=2
    elif TH == 'sig0_rms_ku':#0.24+0.4*np.exp(-swhC_bins_ex/1.20)#
        ce1 = np.asarray([0.45, 0.5,  1.1 ])
        ce2=[]
        celim=[]
        EXP=True
    elif TH == 'sig0_rms_c':#0.24+0.4*np.exp(-swhC_bins_ex/1.20)#
        ce1 = np.asarray([0.235, 0.22,  0.5])
        ce2=[]
        celim=[]
        EXP=True
    elif TH == 'off_nadir_angle_wf_ku':
        ce1 = np.asarray([ 0.21421493, -0.1200238,   0.03750654, -0.00457986])
        ce2=np.asarray([ 0.0797591,  -0.00027225])
        ce1b = np.asarray([-0.12843386,  0.06553327, -0.03882656,  0.01120318])
        ce2b=np.asarray([-0.05439435,  0.00018005])
        celim=2.
    if Nd == 1:
        rms_th,drms,var_bin,par_bin = plyfit(ce1,ce2,swh,rms,celim,EXP)
        if 'swh' in TH:
            ikp = np.where((drms>=0)&(rms>=0.05))[0]
            rms_thb=[]
        elif 'off_nadir_angle_wf_ku' in TH:
            rms_thb,drmsb,var_bin,par_bin = plyfit(ce1b,ce2b,swh,rms,celim,EXP)
            ikp = np.where((drms>=0)&(drmsb<=0))[0]
        else:
            rms_thb=[]
            ikp = np.where((drms>=0))[0]
    else:
        rms_th0,drms0,var_bin,par_bin = plyfit(ce1,ce2,swh[:,0],rms[:,0],celim,EXP)
        rms_th1,drms1,var_bin,par_bin = plyfit(ce1,ce2,swh[:,1],rms[:,1],celim,EXP)
        if 'swh' in TH:
            ikp = np.where((drms0>=0)&(drms1>=0)&(rms>=0.05))[0]
        elif 'off_nadir_angle_wf_ku' in TH:
            rms_thb0,drmsb0,var_bin,par_bin = plyfit(ce1b,ce2b,swh[:,0],rms[:,0],celim,EXP)
            rms_thb1,drmsb1,var_bin,par_bin = plyfit(ce1b,ce2b,swh[:,1],rms[:,1],celim,EXP)
            ikp = np.where((drms0>=0)&(drmsb0<=0)&(drms1>=0)&(drmsb1<=0))[0]
        else:
            ikp = np.where((drms0>=0)&(drms1>=0))[0]
    return ikp

def dynamic_threshold_s3(rms,swh,TH,MN,MODE):
    #Purpose: provides indices for S3 swh rms, range rms, sig0 rms and off-nadir angle that fall within the dynamic threshold envelope.
    Nd = np.size(np.shape(swh))
    EXP=False
    if TH == 'swh_rms_ku':
        if MODE=='sar':
            ce1 = np.asarray([ 0.32144,  1.11207, -0.77325,  0.15379  ]) #[0,2],[2,12.5]
            ce2 = np.asarray([1.00385, -0.31216,  0.08247, -0.00477]) #
        elif MODE=='plrm':
            ce1 = np.asarray([ 0.92804,  0.21314, -0.02314, -0.01067  ]) #[0,2],[2,12.5]
            ce2 = np.asarray([ 1.07305, 0.02612]) #
        celim = 2.5 #[0,2],[2,12.5]
    elif TH == 'swh_rms_c':
        ce1 = np.asarray([ 2.26109e+00,  1.26050e-01, -2.15300e-02,  9.60000e-04])
        ce2=[]
        celim=[]
    elif TH == 'range_rms_ku':
        if MODE=='sar':
            ce1 = np.asarray([0.07881, -0.00261,  0.00391, -0.0002]) #[0,2],[2,12.5]
            ce2 = []
        elif MODE=='plrm':
            ce1 = np.asarray([ 0.14424, 0.01546  ]) #[0,2],[2,12.5]
            ce2 = []
        celim=[]
    elif TH == 'range_rms_c':
        ce1 = np.asarray([ 3.6659e-01, -1.7370e-02,  1.5170e-02, -1.9200e-03,  6.0000e-05])
        ce2=[]
        celim=[]
    elif TH == 'sig0_rms_ku':#0.24+0.4*np.exp(-swhC_bins_ex/1.20)#
        if MODE=='sar':
            ce1 = np.asarray([0.2,  0.67, 1.3])
        elif MODE=='plrm':
            ce1 = np.asarray([ 0.3,  0.55, 1.6 ])
        ce2=[]
        celim=[]
        EXP=True
    elif TH == 'sig0_rms_c':#0.24+0.4*np.exp(-swhC_bins_ex/1.20)#
        ce1 = np.asarray([ 0.3,  0.35, 0.8])
        ce2=[]
        celim=[]
        EXP=True
    elif TH == 'off_nadir_angle_wf_ku':
        ce1 = np.asarray([ 0.03,0.08,1.1])
        ce2=[]
        ce1b = np.asarray([-0.044,-0.08,1.1])
        ce2b=[]
        celim=[]
        EXP=True
    if Nd == 1:
        rms_th,drms,var_bin,par_bin = plyfit(ce1,ce2,swh,rms,celim,EXP)
        if 'swh' in TH:
            ikp = np.where((drms>=0)&(rms>=0.05))[0]
        elif 'off_nadir_angle_wf_ku' in TH:
            rms_thb,drmsb,var_bin,par_bin = plyfit(ce1b,ce2b,swh,rms,celim,EXP)
            ikp = np.where((drms>=0)&(drmsb<=0))[0]
        else:
            ikp = np.where((drms>=0))[0]
    else:
        rms_th0,drms0,var_bin,par_bin = plyfit(ce1,ce2,swh[:,0],rms[:,0],celim,EXP)
        rms_th1,drms1,var_bin,par_bin = plyfit(ce1,ce2,swh[:,1],rms[:,1],celim,EXP)
        if 'swh' in TH:
            ikp = np.where((drms0>=0)&(drms1>=0)&(rms>=0.05))[0]
        elif 'off_nadir_angle_wf_ku' in TH:
            rms_thb0,drmsb0,var_bin,par_bin = plyfit(ce1b,ce2b,swh[:,0],rms[:,0],celim,EXP)
            rms_thb1,drmsb1,var_bin,par_bin = plyfit(ce1b,ce2b,swh[:,1],rms[:,1],celim,EXP)
            ikp = np.where((drms0>=0)&(drmsb0<=0)&(drms1>=0)&(drmsb1<=0))[0]
        else:
            ikp = np.where((drms0>=0)&(drms1>=0))[0]
    return ikp

def dynamic_threshold_jason(rms,swh,TH,MN):
    # Purpose: provides indices for Jason-(1-3) swh rms, range rms, sig0 rms and off-nadir angle that fall within the dynamic threshold envelope.
    Nd = np.size(np.shape(swh))
    EXP=False
    if 'mle3' in TH:
        if MN=='j1':
            raise('no MLE3 for J1')
    if TH == 'swh_rms_ku':
        ce1 = np.asarray([ 0.75064, -0.23178,  1.06577, -0.87798,  0.205  ]) #[0,2],[2,12.5]
        ce2 = np.asarray([ 0.79364, -0.03835,  0.01915, -0.00094]) #
        celim = 2 #[0,2],[2,12.5]
    elif TH == 'swh_rms_c':
        ce1 = np.asarray([ 2.26537,  0.08134, -0.25133,  0.09481])
        ce2=np.asarray([ 1.92622e+00,  9.01900e-02,  1.35300e-02, -9.40000e-04])
        celim=2.
        if MN=='j3':
            celim=2#1.8
            ce1[0]=ce1[0]+0.03
            ce2[0]=ce2[0]+0.05
    elif TH == 'swh_rms_ku_mle3':
        ce1 = np.asarray([ 0.75064, -0.23178,  1.06577, -0.87798,  0.205  ]) #[0,2],[2,12.5]
        ce2 = np.asarray([ 0.8764, -0.04835,  0.01915, -0.00094]) #
        celim = 1.6
    elif TH == 'range_rms_ku':
        ce1 = np.asarray([ 0.10381, -0.01169,  0.01916, -0.00464])
        ce2=np.asarray([ 0.07817,  0.02154, -0.00039])
        celim=2
    elif TH == 'range_rms_c':
        ce1 = np.asarray([ 0.25451, -0.03105,  0.02586, -0.00258])
        ce2=np.asarray([ 2.1012e-01,  3.0080e-02,  2.1800e-03, -1.9000e-04])
        celim=2
    elif TH == 'range_rms_ku_mle3':
        ce1 = np.asarray([0.10048,0.01003])
        ce2=[]
        celim=[]
    elif TH == 'sig0_rms_ku':#0.24+0.4*np.exp(-swhC_bins_ex/1.20)#
        ce1 = np.asarray([ 0.81588, -0.07736, -0.03154,  0.01029])#np.asarray([ 0.80588, -0.07736, -0.03154,  0.01029])
        ce2=np.asarray([ 0.59713,  7.0100e-03, -4.3000e-04])#np.asarray([ 5.8213e-01,  7.0100e-03, -4.3000e-04])
        celim=2.
    elif TH == 'sig0_rms_c':#0.24+0.4*np.exp(-swhC_bins_ex/1.20)#
        ce1 = np.asarray([ 0.38314, -0.11539,  0.02197])
        ce2=np.asarray([ 0.25751, -0.01255,  0.00091])
        celim=2.
        if MN=='j1':
            EXP=True
            ce1=np.asarray([0.245, 0.22, 0.7])
            ce2=[]
    elif TH == 'sig0_rms_ku_mle3':#0.24+0.4*np.exp(-swhC_bins_ex/1.20)#
        ce1 = np.asarray([ 0.32225,-0.08348])#np.asarray([ 0.32225,-0.08748])
        ce2=np.asarray([ 0.27623,-0.06426,0.00861,-0.00039])#np.asarray([ 0.26623,-0.06426,0.00861,-0.00039])
        celim=1.5
    elif TH == 'off_nadir_angle_wf_ku':#0.24+0.4*np.exp(-swhC_bins_ex/1.20)#
        ce1 = np.asarray([ 0.1354 , -0.00828, -0.02823,  0.007  ])
        ce2=np.asarray([ 0.06069, -0.00047,  0.0001 ])
        ce1b = np.asarray([ -0.1354 , 0.00828, 0.02823,  -0.007  ])
        ce2b=np.asarray([ -0.06069, 0.00047,  -0.0001 ])
        celim=2.
    if Nd == 1:
        rms_th,drms,var_bin,par_bin = plyfit(ce1,ce2,swh,rms,celim,EXP)
        if 'swh' in TH:
            ikp = np.where((drms>=0)&(rms>=0.05))[0]
        elif 'off_nadir_angle_wf_ku' in TH:
            rms_thb,drmsb,var_bin,par_bin = plyfit(ce1b,ce2b,swh,rms,celim,EXP)
            ikp = np.where((drms>=0)&(drmsb<=0))[0]
        else:
            ikp = np.where((drms>=0))[0]
    else:
        rms_th0,drms0,var_bin,par_bin = plyfit(ce1,ce2,swh[:,0],rms[:,0],celim,EXP)
        rms_th1,drms1,var_bin,par_bin = plyfit(ce1,ce2,swh[:,1],rms[:,1],celim,EXP)
        if 'swh' in TH:
            ikp = np.where((drms0>=0)&(drms1>=0)&(rms>=0.05))[0]
        elif 'off_nadir_angle_wf_ku' in TH:
            rms_thb0,drmsb0,var_bin,par_bin = plyfit(ce1b,ce2b,swh[:,0],rms[:,0],celim,EXP)
            rms_thb1,drmsb1,var_bin,par_bin = plyfit(ce1b,ce2b,swh[:,1],rms[:,1],celim,EXP)
            ikp = np.where((drms0>=0)&(drmsb0<=0)&(drms1>=0)&(drmsb1<=0))[0]
        else:
            ikp = np.where((drms0>=0)&(drms1>=0))[0]
    return ikp

def dynamic_threshold_tx(rms,swh,TH):
    # Purpose: provides indices for Topex/Poseidon swh rms, range rms, sig0 rms and off-nadir angle that fall within the dynamic threshold envelope.
    Nd = np.size(np.shape(swh))
    if TH == 'swh_rms_ku':
        ce = np.asarray([ 5.3088e-01, -1.1410e-02,  7.2900e-03, -5.2000e-04,  4.0000e-05])
        EXP=False
    elif TH == 'swh_rms_c':
        ce = np.asarray([ 8.8760e-01, -2.0056e-01,  6.8700e-02, -8.4500e-03,  3.9000e-04])
        EXP=False
    elif TH == 'swh_rms_ku_mle3':
        ce = np.asarray([ 5.7088e-01, -1.1410e-02,  7.2900e-03, -5.2000e-04,  4.0000e-05])
        EXP=False
    elif TH == 'range_rms_ku':
        ce = np.asarray([ 8.097e-02, -1.799e-02,  6.350e-03, -5.700e-04,  2.000e-05])
        EXP=False
    elif TH == 'range_rms_c':
        ce = np.asarray([ 1.7882e-01, -3.6930e-02,  1.3100e-02, -1.1400e-03,  3.0000e-05])
        EXP=False
    elif TH == 'range_rms_ku_mle3':
        ce = np.asarray([ 0.08453, -0.01543,  0.00382, -0.00019])#np.asarray([ 0.11097, -0.028,  0.007, -5.800e-04,  1.800e-05])
        EXP=False
    elif TH == 'sig0_rms_ku':
        ce = np.asarray([ 0.24, 0.40,  1.20])
        EXP=True
    elif TH == 'sig0_rms_c':
        ce = np.asarray([ 0.15, 0.24,  1.20])
        EXP=True
    elif TH == 'sig0_rms_ku_mle3':
        ce = np.asarray([ 0.30962, -0.0697,0.00995, -0.00046])
        EXP=False
    elif TH == 'off_nadir_angle_wf_ku':
        ce = np.asarray([ 0.032, 0.055,  1.20])
        ceb = np.asarray([ -0.027, -0.055,  1.20])
        EXP=True
    if Nd == 1:
        rms_th,drms,var_bin,par_bin = plyfit(ce,[],swh,rms,[],EXP)
        if 'swh' in TH:
            ikp = np.where((drms>=0)&(rms>=0.05))[0]
        elif 'off_nadir_angle_wf_ku' in TH:
            rms_thb,drmsb,var_bin,par_bin = plyfit(ceb,[],swh,rms,[],EXP)
            ikp = np.where((drms>=0)&(drmsb<=0))[0]
        else:
            ikp = np.where(drms>=0)[0]
    else:
        rms_th0,drms0,var_bin,par_bin = plyfit(ce,[],swh[:,0],rms[:,0],[],EXP)
        rms_th1,drms1,var_bin,par_bin = plyfit(ce,[],swh[:,1],rms[:,1],[],EXP)
        if 'swh' in TH:
            ikp = np.where((drms0>=0)&(drms1>=0)&(rms>=0.05))[0]
        elif 'off_nadir_angle_wf_ku' in TH:
            rms_thb0,drmsb0,var_bin,par_bin = plyfit(ceb,[],swh[:,0],rms[:,0],[],EXP)
            rms_thb1,drmsb1,var_bin,par_bin = plyfit(ceb,[],swh[:,1],rms[:,1],[],EXP)
            ikp = np.where((drms0>=0)&(drms1>=0)&(drmsb0<=0)&(drmsb1<=0))[0]
        else:
            ikp = np.where((drms0>=0)&(drms1>=0))[0]
    return ikp


def dynamic_threshold(rms,swh,TH,MN,MODE):
    # Purpose: final run function to for all dynamic_threshold_mission functions.
    '''
    rms = array of measured rms (swh, sig0 or range) or off-nadir angle values 
    swh = array of measured Ku band swh values
    TH = string that indicates rms type or off_nadir angle
    MN = mission name
    MODE = string that indicates LRM or SAR mode
    ikp = indices of swh rms, range rms, sig0 rms or off-nadir angle that fall within the dynamic threshold envelope of that variable
    '''
    if MN=='tx':
        ikp=dynamic_threshold_tx(rms,swh,TH)
    elif MN=='j1':
        ikp=dynamic_threshold_jason(rms,swh,TH,MN)
    elif MN=='j2':
        ikp=dynamic_threshold_jason(rms,swh,TH,MN)
    elif MN=='j3':
        ikp=dynamic_threshold_jason(rms,swh,TH,MN)
    elif MN=='s3':
        ikp=dynamic_threshold_s3(rms,swh,TH,MN,MODE)
    elif MN=='s6A':
        ikp=dynamic_threshold_s6A(rms,swh,TH,MN,MODE)
    elif MN=='s6Ah':
        ikp=dynamic_threshold_s6Ah(rms,swh,TH,MN,MODE)
    elif MN=='sw':
        ikp=dynamic_threshold_sw(rms,swh,TH)
    elif MN=='sa':
        ikp=dynamic_threshold_sa(rms,swh,TH)
    return ikp


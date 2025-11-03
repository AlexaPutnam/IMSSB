#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:03:17 2020

@author: alexaputnam

Author: Alexa Angelina Putnam Shilleh
Institution: University of Colorado Boulder
Software: sea state bias model generator (interpolation method)
Thesis: Sea State Bias Model Development and Error Analysis for Pulse-limited Altimetry
Semester/Year: Fall 2021
"""
import numpy as np
import time
import subprocess

###############################################
###############################################
# Define paths to pull altimeter product data and orbit files (optional), and the path to save files
# these change with the system
def save_path():
    # Purpose: define path (PATH2SOFTWARE) to IMSSB software
    PATH2SOFTWARE = '/home/aputnam/IMSSB/imssb_software/'#'/OSTM1/usr/aputnam/imssb_software/'
    return PATH2SOFTWARE

def orbit_paths(MISSIONNAME,CYC):
    '''
    Purpose: provides path to alternative orbit solution
    MISSIONNAME = altimeter mission name (shortened)
    CYC = integer cycle number
    pth = path to orbit solutions
    FN = filename of orbit solutions
    pth_sv = path and filename to save orbit file converted to netCDF4
    FN2 = second file name if there are two solutions for one cycle (J2 only)
    split_cyc = the cycle containing to two solutions (J2 only)
    '''
    ## Eventually remove both constraints for Topex and Jason-3 (Version 3, GDRF) seen in function "product2direct" in "main_product2data_ext.py"
    ##### i.e. ADDITIONAL VARIABLES: parameter statistics and JPL orbit solution --->   if MISSIONNAME !='tx' and VERSION!=3
    METHOD,OMIT,APP,SOURCE,CYC1,CYC2,BAND,DIM,EXT=[],[],'orbit',[],[],[],[],[],[]
    pth_sv = create_filename(MISSIONNAME,METHOD,OMIT,APP,SOURCE,CYC1,CYC2,BAND,DIM,EXT) 
    FN2=[]
    if MISSIONNAME == 'j2':
        split_cyc = [54]
        pth = '/ecomm/sideshow/ftp/anon/pub/usrs/sdd/jas2_gps/rlse18a/'
        if int(CYC) in split_cyc:
            FN = 'ja2_gpsr_rlse18a_'+CYC+'a.pos.gz'
            FN2 = 'ja2_gpsr_rlse18a_'+CYC+'b.pos.gz'
        else:
            FN = 'ja2_gpsr_rlse18a_'+CYC+'.pos.gz'
    elif MISSIONNAME == 'j3':
        pth = '/ecomm/sideshow/ftp/anon/pub/usrs/sdd/jas3_gps/rlse19a/'
        FN = 'ja3_gpsr_rlse19a_'+CYC+'.pos.gz'
    return pth,FN,pth_sv,FN2,split_cyc

def product_paths(MISSIONNAME,SOURCE,CYC,VERSION):
    '''
    Purpose: provides path to altimeter product pass files
    MISSIONNAME = altimeter mission name (shortened)
    CYC = integer cycle number
    VERSION = data format version
    SOURCE: altimeter data source
    PTH_src = path to by-pass product files
    '''
    if SOURCE=='jpl':
        if MISSIONNAME == 'tx':
            #PTH_src = '/OSTM1/projects/topex_gdrf/oldsols/misc/out_dec2020_tmrv5newtropo/c'+CYC+'/' 
            PTH_src = '/OSTM1/projects/topex_gdrf/out_august2021_testJDD_Ku_42to51_C_42to51_sig0adjustedBis/c'+CYC+'/' #'/OSTM1/projects/topex_gdrf/out_dec2020_tmrv5newtropo/c'+CYC+'/' #'/ALT1/usr/aputnam/jpl_topex_ssb/test_data/c'+CYC+'/' #
            #PTH_src = '/OSTM1/projects/topex_gdrf/out_may2021_correctionocomment/c'+CYC+'/'
        elif MISSIONNAME == 'j1':
            PTH_src = '/OSTM3/data/products/jason1_gdre/GDR_'+CYC+'/'
        elif MISSIONNAME == 'j2':
            PTH_src = '/OSTM3/data/products/jason2_gdrd/GDR_'+CYC+'/'
        elif MISSIONNAME == 'j3':
            if VERSION==1:
                PTH_src = '/ALT1/data/products/jason3_gdr/GPN_'+CYC+'/' #'/ALT1/data/products/jason3_gdrf/GPN_'+CYC+'/' 
            elif VERSION==3:
                PTH_src = '/ALT1/data/products/jason3_gdrf/GPN_'+CYC+'/'
                print('accessing GDRF: '+PTH_src)
        elif MISSIONNAME == 's3':
            PTH_src = '/ALT1/usr/aputnam/jpl_topex_ssb/ssb_modeling/s3/sv_data/enhanced_s3_gdr/c'+CYC+'/'
        elif MISSIONNAME == 's6A':
            PTH_src = '/OSTM1/data/products/sentinel6a/reprocess/lrm/c'+CYC+'/'#'/OSTM1/data/products/sentinel6a/stc/lr/c'+CYC+'/'
        elif MISSIONNAME == 's6Ah':
            PTH_src = '/OSTM1/data/products/sentinel6a/reprocess/sar/c'+CYC+'/'#stc/hr/c'+CYC+'/'
    elif SOURCE=='aviso':
        HEADER = aviso_file_starter(MISSIONNAME,int(CYC),VERSION)
        PTH_src = download_aviso_cyc(USERNAME,PASSWORD,CYCLES,MISSIONNAME,TYPE,HEADER)
    elif SOURCE=='rads':
        PTH_src = '/home/aputnam/IMSSB/imssb_software/output/'+MISSIONNAME+'/product2data/rads/'
        raise('RADs does not require running product2data. Move onto data2matrix')
    else:
        PTH_src=[]
    print(PTH_src)
    return PTH_src

def aviso_file_starter(MISSIONNAME,CYC,VERSION):
    '''
    Purpose: define header for product files
    MISSIONNAME = altimeter mission name (shortened)
    CYC = integer cycle number
    VERSION = data format version
    head = define header of product file names
    '''
    nCYC = int(CYC)
    if MISSIONNAME == 'tx':
        head = 'TP_GPN_2PfP'
    elif MISSIONNAME == 'j1':
        head = 'JA1_GPN_2PeP'
    elif MISSIONNAME == 'j2':
        head = 'JA2_GPN_2PdP'
    elif MISSIONNAME == 'j3':
        if VERSION==3:
            head = 'JA3_GPN_2PfP' #for GDRF
        elif VERSION==1:
            if nCYC<22:
                head = 'JA3_GPN_2PTP'
            else:
                head = 'JA3_GPN_2PdP' #JA3_GPN_2PdP024
    elif MISSIONNAME=='s3':
        head = 'S3A_SR_2_'
    elif MISSIONNAME=='s6A':
        head = 'S6A_P4_2__LR_STD__NT_'#'S6A_P4_2__LR_STD__ST_'
    elif MISSIONNAME=='s6Ah':
        head = 'S6A_P4_2__HR_STD__NT_'
    elif MISSIONNAME=='sw':
        head = 'swp_'
    return head


def download_aviso_cyc(USERNAME,PASSWORD,CYCLES,MISSIONNAME,HEADER):
    '''
    Purpose: Download altimeter product from Aviso+
    USERNAME = Aviso username
    PASSWORD = Aviso password
    HEADER = header of product file name
    '''
    # Time: 21 minutes per cycle
    # USERNAME,PASSWORD,MISSIONNAME,CYCLES,TYPE = 'alexa.putnam@colorado.edu','ZFiu2g','j2',np.arange(2,3),'model'
    N = np.shape(CYCLES)[0]
    PATHMISSIONS = save_path()
    PATHAVISO = PATHMISSIONS+'output/'+MISSIONNAME+'/product2data/aviso/'
    for ii in np.arange(N):
        t1 = time.time()
        CYC = '{:03d}'.format(CYCLES[ii])
        subprocess.check_call(['bash', PATHAVISO+'get_gdr.sh',CYC,USERNAME,PASSWORD,PATHAVISO,HEADER])
        if MISSIONNAME == 'j3':
            subprocess.check_call(['bash', PATHAVISO+'zipout.sh'])
        t2 = time.time()
        print('Time to download cycle '+CYC+' from Aviso: '+str(t2-t1))
    return PATHAVISO
###############################################
###############################################

###############################################
# Files and labels
###############################################
def file_labels(VAR,BAND,DIM,OMIT):
    # Purpose: strings required for file labeling within matrix2model
    # FILE ENDINGS
    if np.size(OMIT) == 0:
        OMITlab = '2d'
    else:
        OMITlab = OMIT
    DIMND = DIM+'_'+OMITlab
    MODELFILE='matrix2model'
    if 'usla' in VAR:
        if 'c' in BAND:
            raise('usla_ku_aps also computes usla_c_aps. No need to compute separately.')
    return DIMND,OMITlab,MODELFILE

def filenames_specific(MISSIONNAME,METHOD,SOURCE,CYC1,CYC2,BAND,DIMND,ERR=[]):
    '''
    Purpose: path+file names for all three files produced in data2matrix
    CYC1 = CYC2 = CYC
    FILE1 = data2vector path+filename
    FILE2 = data2matrix path+filename
    FILE2_GEO = data2matrix path+filename for spatial observation matrix (COE)
    '''
    VAR = 'usla'
    APP='data2matrix'
    # CREATES OBSERVATION MATRICES AND VECTOR FILE NAMES
    if METHOD=='coe':
        FILE1 = create_filename(MISSIONNAME,'dir',VAR,'data2vector',SOURCE,CYC1,CYC2,BAND,DIMND,'nc')
        FILE2 = create_filename(MISSIONNAME,'dir',VAR,'data2matrix',SOURCE,CYC1,CYC2,BAND,DIMND,'npz')
    else:
        FILE1 = create_filename(MISSIONNAME,METHOD,VAR,'data2vector',SOURCE,CYC1,CYC2,BAND,DIMND,'nc')
        FILE2 = create_filename(MISSIONNAME,METHOD,VAR,'data2matrix',SOURCE,CYC1,CYC2,BAND,DIMND,'npz')
    FILE2_GEO = FILE2[:-4]+'_geo'+FILE2[-4:]
    if np.size(ERR)!=0:
        if 'data2matrix' in APP:
            if ERR=='rms_rws':
                ERR2='_POE_rms_ws'
            elif ERR=='rms_rw':
                ERR2='_POE_rms_w'
            elif ERR=='rms_wx':
                ERR2='_POE_rms_w'
            elif ERR=='rms_Ewr':
                ERR2='_POE_rms_Ew'
            elif ERR == 'rms_rs':
                ERR2='_POE_rms_s'
            elif ERR =='rms_r':
                ERR2=''
            elif ERR =='rng_corr':
                ERR2=''
            else:
                ERR2='_POE_'+ERR
        ERR2a=ERR2
        FILE1 = FILE1[:-3]+ERR2a+FILE1[-3:]
        FILE2 = FILE2[:-4]+ERR2+FILE2[-4:]
        FILE2_GEO = FILE2_GEO[:-4]+ERR2a+FILE2_GEO[-4:]
    return FILE1,FILE2,FILE2_GEO

def create_filename(MISSIONNAME,METHOD,OMIT,APP,SOURCE,CYC1,CYC2,BAND,DIM,EXT,ERR=[],VERSION=[]):
    '''
    Purpose: provide path+filename for product2data, data2matrix and matrix2model steps, as well as the new orbit solutions
    EXT = file extension from file_labels functions
    FILE = resulting path+filename
    '''
    # CREATES FILENAMES
    PATHMISSIONS = save_path()
    if 'usla' in OMIT:
        MODELVAR = 'ssb'
    else:
	    MODELVAR = OMIT
    if BAND == 'na': #DIM == 'na':
        FILE = PATHMISSIONS+'output/'+MISSIONNAME+'/product2data/'+SOURCE+'/'+MISSIONNAME+'_dir_'+APP+'_'+SOURCE+'_'+CYC1+'_c_'+CYC2+'_'+BAND+'_'+DIM+'.'+EXT
    elif np.size(BAND)==0:
	    FILE = PATHMISSIONS+'output/'+MISSIONNAME+'/orbits/'
    else:#/rain_flag_20200909
        if METHOD=='coe':
            if 'data2matrix' in APP:
                METHOD2='dir'
            else:
                METHOD2=METHOD
        else:
            METHOD2=METHOD
        FILE = PATHMISSIONS+'output/'+MISSIONNAME+'/'+METHOD2+'/'+APP+'/'+MISSIONNAME+'_'+METHOD2+'_'+MODELVAR+'_'+APP+'_'+SOURCE+'_'+CYC1+'_c_'+CYC2+'_'+BAND+'_'+DIM+'.'+EXT
    if np.size(ERR)!=0:
        if 'data2matrix' in APP:
            if ERR=='rms_rws':
                ERR2='_POE_rms_ws'
            elif ERR=='rms_rw':
                ERR2='_POE_rms_w'
            elif ERR=='rms_wx':
                ERR2='_POE_rms_w'
            elif ERR == 'rms_rs':
                ERR2='_POE_rms_s'
            elif ERR =='rms_r':
                ERR2=''
            else:
                ERR2='_POE_'+ERR
        else:
            ERR2='_POE_'+ERR
        FILE = FILE[:-3]+ERR2+FILE[-3:]
    if APP =='matrix2model':
        if ERR in ['txj1','j1j2','j2j3','j3s6A']:
            FILE = FILE[:-3]+'_noMSSinterp'+FILE[-3:]
    return FILE

def mission_specs(MISSIONNAME):
    # ORBIT DESIGN SPECIFICS
    Ve = 0.46 # [km/s] linear velocity of the Earth's rotation
    if MISSIONNAME in ['tx','j1','j2','j3','s6A','s6Ah']: # 10 day cycle, 127 rev/cyc, 254 pass/cyc
        Vs = 5.8 # [km/s] along-track ground velocity of the satellite
        I_deg = 66.16 #66.06 # [deg] satellite orbit inclination
        alt = 1337500.0 #SV altitude
    elif MISSIONNAME =='s3': # 27 day cycle, 385 rev/cyc, 770 pass/cyc
        Vs = 5.8 # [km/s] along-track ground velocity of the satellite
        I_deg = 98.65 # [deg] satellite orbit inclination
        alt = 814500.0
    elif MISSIONNAME =='sw': # 21 day cycle, science orbit = 292 rev/cyc, fast-sampling orbit = 292 rev/cyc, period = 112.42 minutes
        # https://www.eoportal.org/satellite-missions/swot#spacecraft
        I_deg = 77.6 # [deg] satellite orbit inclination
        alt = 857244.0 # fast-sampling orbit 890582.0 # science orbit = 857244.0
        Vs = 7372.0 #7.9*((6378.0/alt)**(3.0/2.0))-0.465#np.sqrt(3986000/(6378+1337.5)) # [km/s] along-track ground velocity of the satellite
        # 7.9*((6378.0/1337.5)**(3.0/2.0))-0.465
    elif MISSIONNAME =='sa': # 35-day cycle
        # https://www.aviso.altimetry.fr/en/missions/current-missions/saral/orbit-1.html
        I_deg = 98.538 # [deg] satellite orbit inclination
        alt = 781000.0
        Vs = 7470.0
    return Ve,Vs,I_deg,alt


# Mission cycles
def bad_cycles(MISSIONNAME):
    # define cycle that are not provided by the product
    if MISSIONNAME == 'tx':
        badcyc = np.asarray([1,2,20,31,41,55,65,79,91,97,103,114,118,126,138,150,162,174,180,186,197,209,216,224,234,243,256,266,278,289,299,307,361,398,416,417,418,419,420,431,432])
	    #np.asarray([1,2,20,31,41,55,65,79,91,97,103,114,118,126,129,138,150,162,174,180,186,197,209,216,224,234,243,256,266,278,289,298,299,307,361,398,416,417,418,419,420,431,432]) #data2mat
    elif MISSIONNAME == 'j1':
        badcyc = np.asarray([178,243,260,261])
    elif MISSIONNAME == 'j2':
        badcyc = np.asarray([304,507,508])
	    #        np.asarray([53,54,62,63,73,78,80,90,91,219,304,507,508]) #53,54,62,63,73,78,80,90,91 are only bad because there is no altitude alternative--> remove after error analysis
	    #	 np.asarray([4,219,304,507,508]) # for tandem (j1j2)
    elif MISSIONNAME == 'j3':
        badcyc = np.asarray([112,213])#[]
	    # 	 np.asarray([106,112])
    elif MISSIONNAME=='s3':
        badcyc = [1,2,3,4,5]
	    #	 [1,2,3,4,5,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
    elif MISSIONNAME in ['s6A','s6Ah']:
        badcyc = np.asarray([1,2,3,4,38])
    elif MISSIONNAME in ['sw']:#5-day bad: 169,221,222 || 10-day bad: 164,169,216,217,221,222
        badcyc = np.asarray([128,132,174,226,227])#[174,226,227])
    elif MISSIONNAME in ['sa']:
        badcyc = np.arange(36)
    return badcyc

def tandem(ERR,METHOD):
    print('Tandem mission '+METHOD+' models for '+ERR)
    if ERR=='txj1':
        CYCs1i = np.asarray([x for x in np.arange(344,365) if x!=361])
        CYCs2i = CYCs1i-343#np.asarray([x for x in np.arange(1,22) if x!=18])
        MN1 = 'tx'
        MN2 = 'j1'
    elif ERR=='j1j2':
        CYCs1i = np.asarray([x for x in np.arange(240,260) if x!=243])
        CYCs2i = CYCs1i-239#np.asarray([x for x in np.arange(1,21) if x!=4])
        MN1 = 'j1'
        MN2 = 'j2'
    elif ERR=='j2j3':
        CYCs1i = np.arange(281,304)
        CYCs2i = CYCs1i-280#np.arange(1,24)
        MN1 = 'j2'
        MN2 = 'j3'
    elif ERR=='j3s6A':
        CYCs1i = np.arange(180,228)#(180,218)
        CYCs2i = CYCs1i-175#np.arange(20,38)#(5,43)
        MN1 = 'j3'
        MN2 = 's6A'
    if 'col' in METHOD:
        CYCs1 = CYCs1i[:-1]
        CYCs2 = CYCs2i[:-1]
    else:
        CYCs1 = np.copy(CYCs1i)
        CYCs2 = np.copy(CYCs2i)
    return MN1,CYCs1,MN2,CYCs2

def mission_cycles(MISSIONNAME,MNtag):
    '''
    Purpose: provide an array of cycle numbers corresponding to a time period (MNtag) and cycles not provided by the product.
    CYC = array of cycles corresponding to MNtag
    badcyc = cycle numbers that are not provided by the product
    ~~~~Tandem phases~~~~~~
    Tandem phases:
    TOPEX/J1: 344-364 for TOPEX, 1-21 for J1
    J1/J2: 240-259 for J1, 1-20 for J2
    J2/J3: 281-303 for J2, 1-23 for J3
    '''
    badcyc = bad_cycles(MISSIONNAME)
    if MISSIONNAME == 'tx': # high SLA: cycles: 66,100,107,108,110,111
        if MNtag=='txj1':
            MN1,CYC,MN2,CYCs2 = tandem(MNtag,'dir')
        elif MNtag=='sa': #sa2 = np.arange(48,122)
            CYC = np.arange(48,101)
        elif MNtag=='sb':
            CYC = np.arange(280,365)
        elif MNtag=='sb2':
            CYC = np.arange(280,317)
        elif MNtag=='y1': #1992
            CYC = np.arange(1,12)
        elif MNtag=='y2':#1993
            CYC = np.arange(12,48)
        elif MNtag=='y3':#1994
            CYC = np.arange (48,85) 
        elif MNtag=='y4':#1995
            CYC = np.arange(85,122) 
        elif MNtag=='y5':#1996
            CYC = np.arange(122,159) 
        elif MNtag=='y6':#1997
            CYC = np.arange(159,196) 
        elif MNtag=='y7': #1998
            CYC = np.arange(196,233) 
        elif MNtag=='y8': #1999
            CYC = np.arange(235,269) 
        elif MNtag=='y9': #2000
            CYC = np.arange(269,306)          
        elif MNtag=='y10': #2001
            CYC = np.arange(306,342)         
        elif MNtag=='all': #1994-2002
            CYC = np.arange(48,365) 
        elif MNtag=='sa_all': #1994-2002
            CYC = np.arange(48,236)
        elif MNtag=='sb_all': #1994-2002
            CYC = np.arange(236,365)
    elif MISSIONNAME == 'j1':
        if MNtag=='txj1':
            MN1,CYCs1,MN2,CYC = tandem(MNtag,'dir')
        elif MNtag=='j1j2':
            MN1,CYC,MN2,CYCs2 = tandem(MNtag,'dir')
        elif MNtag=='y1': #2002
            CYC = np.arange(1,37)
        elif MNtag=='y2': #2003
            CYC = np.arange(37,74)
        elif MNtag=='y3': #2004
            CYC = np.arange(74,111)
        elif MNtag=='y4': #2005
            CYC = np.arange(111,147)
        elif MNtag=='y5': #2006
            CYC = np.arange(147,184)
        elif MNtag=='y6': #2007
            CYC = np.arange(184,221)
        elif MNtag=='y7': #2008
            CYC = np.arange(221,258)
        elif MNtag=='all': #2002-2008
            CYC = np.arange(1,260)
    elif MISSIONNAME == 'j2':
        if MNtag=='cnes':
            CYC = np.arange(1,37) # (1,328), (1,21), (281,304)
        elif MNtag=='j1j2':
	    #badcyc = np.asarray([4,219,304,507,508])
            MN1,CYCs1,MN2,CYC = tandem(MNtag,'dir')
        elif MNtag=='j2j3':
            MN1,CYC,MN2,CYCs2 = tandem(MNtag,'dir')
        elif MNtag=='y1': #2008
            CYC = np.arange(1,19) 
        elif MNtag=='y2': #2009
            CYC = np.arange(19,56) 
        elif MNtag=='y3': #2010
            CYC = np.arange(56,92) 
        elif MNtag=='y4':#2011
            CYC = np.arange(92,129) 
        elif MNtag=='y5': #2012 # this will be the test year
            CYC = np.arange(129,166) 
        elif MNtag=='y6': #2013
            CYC = np.arange(166,203) 
        elif MNtag=='y7': #2014
            CYC = np.arange(203,240)          
        elif MNtag=='y8': #2015
            CYC = np.arange(240,277)
        elif MNtag=='y9': #2016
            CYC = np.arange(277,304)#314)    
        elif MNtag=='all': #2008-2016
            CYC = np.arange(1,304)
    elif MISSIONNAME == 'j3':
        if MNtag=='cls':
            CYC = np.arange(17,55)
        elif MNtag=='j2j3':
            MN1,CYCs1,MN2,CYC = tandem(MNtag,'dir')
        elif MNtag=='j3s6A':
            MN1,CYC,MN2,CYCs2 = tandem(MNtag,'dir')
        elif MNtag=='rtrk':
            CYC = np.arange(22,55) #(1,167), (1,24)
        elif MNtag=='y1':
            CYC = np.arange(1,33)#2016
        elif MNtag=='y2':
            CYC = np.arange(33,70)#2017
        elif MNtag=='y3':
            CYC = np.arange(70,106)#2018
        elif MNtag=='y4':
            CYC = np.arange(107,144)#2019
        elif MNtag=='y5':
            CYC = np.arange(144,180)#2020
        elif MNtag=='y3456w':
            CYC = np.hstack((np.arange(135,150)-35,np.hstack((np.arange(135,150),np.arange(135,150)+36))))#np.hstack((np.arange(135,150),np.arange(135,150)+36))#np.arange(135,150)#2020
        elif MNtag=='y456s':
            CYC = np.hstack((np.arange(156,171)-35,np.hstack((np.arange(156,171),np.arange(156,171)+36))))#np.hstack((np.arange(156,171),np.arange(156,171)+36))#np.arange(156,171)#2020
        elif MNtag=='y5s':
            CYC = np.arange(156,171)#2020
        elif MNtag=='y45ws':
            CYC = np.arange(135,171)#2020
        elif MNtag=='y6':
            CYC = np.arange(180,216)#2021
        elif MNtag=='y456':
            CYC = np.arange(107,216)#2021
        elif MNtag=='all': #2016-2021
            CYC = np.arange(1,228)
    elif MISSIONNAME == 's3':
        if MNtag=='y1':
            CYC = np.arange(1,12)
        elif MNtag=='y2':
            CYC = np.arange(12,27)
        elif MNtag=='y3':
            CYC = np.arange(27,41)
        elif MNtag=='y4':
            CYC = np.arange(41,54)
    elif MISSIONNAME in ['s6A','s6Ah']:
        if MNtag=='j3s6A':
            MN1,CYCs1,MN2,CYC = tandem(MNtag,'dir')
        elif MNtag=='sa': 
            CYC = np.arange(5,31)
        elif MNtag=='sb':
            CYC = np.arange(32,58)
        elif MNtag=='y1': #2021
            CYC = np.arange(5,42)
        elif MNtag=='all': #2021-2022
            CYC = np.arange(5,63)
    elif MISSIONNAME == 'sw':
        if MNtag=='y1':
            CYC = np.arange(102,278)#102,278)#174 = 2023-03-29, 249=2023-06-11, 277=2023-07-10
        elif MNtag=='y1b':
            CYC = np.arange(301,306)
        elif MNtag=='all':
            CYC = np.asarray(np.hstack((np.arange(174,278),np.arange(301,306))))
    elif MISSIONNAME == 'sa':
        if MNtag=='y1':
            CYC = np.arange(36,48)
    return CYC,badcyc

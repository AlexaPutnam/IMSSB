#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22-November-2022

@author: alexaputnam
"""
import ana_rads2imssb as arads

# /srv/data/rads/tracks/
MISSIONNAME = 'sw'
SEG = 'b'
OpenOcean = True
arads.new_rads_files(MISSIONNAME,OpenOcean,SEG=SEG)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 23:17:53 2020

@author: daniyalusmani1

This script contains functionality that is used to extract samples with changed labels for experimentation
"""

import numpy as np

def extractSamplesWithLabelsChanged(origSamples, origLabelColumn, changedSamples, changedLabelColumn):
    # TODO: have assertions about the size and shape of arrays provided
    origLabels = np.array(origSamples[origLabelColumn])
    changedLabels = np.array(changedSamples[changedLabelColumn])
    flippedSamplesIndex = np.where(origLabels != changedLabels)[0] 
    # give the % of flipped samples as well
    print("The number of flipped samples are %s"%(len(flippedSamplesIndex)))
    return flippedSamplesIndex
    
    
    
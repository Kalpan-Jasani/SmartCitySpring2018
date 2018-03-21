#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:18:49 2018

@author: etan
"""

def severity (deepest, avgDiameter):
    if deepest <= 25: 
        if avgDiameter <= 200:
            print("L")
        elif avgDiameter > 200 and avgDiameter <= 450:
            print("L")
        elif avgDiameter > 450:
            print("M")
    elif deepest > 25 and deepest <= 50:
        if avgDiameter <= 200:
            print("L")
        elif avgDiameter > 200 and avgDiameter <= 450:
            print("M")
        elif avgDiameter > 450:
            print("H")
    elif deepest > 50:
        if avgDiameter <= 200:
            print("M")
        elif avgDiameter > 200 and avgDiameter <= 450:
            print("M")
        elif avgDiameter > 450:
            print("H")
    elif deepest < 13 and avgDiameter < 100:
        print("Not a pothole")
    return;    
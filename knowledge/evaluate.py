#coding: utf-8

import os
import sys
import time
import argparse
import pdb
import numpy as np
import pandas as pd

def calc_fvalue(ans_set, est_set):
    intersect = ans_set & est_set
    precsion = len(intersect) / len(ans_set)
    recall = len(intersect) / len(est_set)
    fvalue = 2*precsion*recall / max(precsion + recall, 10e-8)
    return fvalue

def overlapping_fvalue_along_com1(member_of_com1, member_of_com2):
    all_num = sum([len(mem) for mem in member_of_com1.values()])
    f_sum = 0
    for com1, com1_mem in member_of_com1.items():
        f_max = 0
        weight = len(com1_mem) / all_num
        for com2, com2_mem in member_of_com2.items():
            fval = calc_fvalue(com1_mem, com2_mem)
            if f_max < fval:
                f_max = fval
        f_sum += f_max * weight
    return f_sum

def averaged_fvalue(member_of_com1, member_of_com2):
    f1 = overlapping_fvalue_along_com1(member_of_com1, member_of_com2)
    f2 = overlapping_fvalue_along_com1(member_of_com2, member_of_com1)
    f = (f1 + f2) / 2
    return f

if __name__=='__main__':
    member_of_ctgy = {"a":set([1,2,3,4,5]),"b":set([2,3,7,8]), "c":set([0,4,5])}
    member_of_clusters = {0:set([1,3,5]), 1:set([2,7,8]), 2:set([0,4])}
    val = averaged_fvalue(member_of_ctgy, member_of_clusters)
    print(val)


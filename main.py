#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 17:10:54 2021

@author: quentin
"""


from utils.utils import compare_pose_opt, compare_deform_opt
import sys
import ast 
from pathlib import Path

if __name__ == "__main__":
    n = len(sys.argv)
    params_file = sys.argv[1]
    file = open(Path().cwd()/"experiments"/params_file, "r")
    params_dic = ast.literal_eval(file.read())
    file.close()
    task = params_dic["task"]
    if task == "pose_opt":
        compare_pose_opt(params_file)
    else:
        compare_deform_opt(params_file)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:05:57 2021

@author: quentin
"""


from setuptools import setup


setup(
    name='pertrenderer',
    version='0.0.1',
    author="quentinll",
    author_email="quentin.le-lidec@inria.fr",
    description="Generic implementation of differentiable renderers in pytorch3d",
    url='https://quentinll.github.io/',
    packages = ['randomras'],
    #py_modules=['random_rasterizer, smoothrast, smoothagg']
)
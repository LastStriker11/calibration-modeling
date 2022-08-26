# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 16:51:43 2022

@author: lasts
"""

SPSA_para = dict(   G         = G,\
                    Min_error = 0.05,\
                    a         = 0.5,\
                    c         = 0.15,\
                    A         = 25,\
                    alpha     = 0.3,\
                    gamma     = 0.15,\
                    h         = 0.7,\
                    N         = N,\
                    seg       = 5)

PC_SPSA_para = dict(   G         = G,\
                Min_error = 0.05,\
                a         = 1,\
                c         = 0.15,\
                A         = 25,\
                alpha     = 0.3,\
                gamma     = 0.15,\
                N         = N)
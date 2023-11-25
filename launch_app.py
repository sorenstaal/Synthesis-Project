#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:18:53 2023

@author: sorenstaal
"""
import warnings
# to ignore scipy "numpy version" warning
warnings.simplefilter("ignore", UserWarning)

import os
import sys
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
import numpy as np
import lum_functions as lf

from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QSizePolicy,
    QMenuBar,
    QCheckBox,
    QHBoxLayout
)

os.chdir('/Users/sorenstaal/Documents/Uni/Kandidat/3. semester/Synthesis Project/GUI')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import utilities as utils

from app_layout import FrontWindow

        

def main():
    w = 800
    h = 1000
    app = QApplication(sys.argv)
    window = FrontWindow()
    window.resize(w, h)
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
   main()
    
    

        
        
                      

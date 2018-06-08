from fsave import fshow
import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

Tk().withdraw()
if len(sys.argv) <= 1:
    filenames = askopenfilenames(title='Select Files for Plotting', filetypes=[('Plot Files', '*.plt')])
else:
    filenames = sys.argv[1]
if type(filenames) == list or type(filenames) == tuple:
    for filename in filenames:
        fshow(filename)
else:
    fshow(filenames)
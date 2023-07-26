import nrrd
import os
from pathlib import Path

#@TODO: Remove unnecessary files - HDF5 and all other whose names start with '.'

suffixes = {}

for filename in os.listdir("/OLD-DATA-STOR/segmentation ovud"):
    sfx = Path("/OLD-DATA-STOR/segmentation ovud/"+filename).suffix
    if filename.startswith('.') and 'dotted_'+sfx not in suffixes:
        suffixes['dotted_'+sfx] = 0
    elif not filename.startswith('.') and sfx not in suffixes:
        suffixes[sfx] = 0

for filename in os.listdir("/OLD-DATA-STOR/segmentation ovud"):
    sfx = Path("/OLD-DATA-STOR/segmentation ovud/"+filename).suffix
    if filename.startswith('.'):
        suffixes['dotted_'+sfx] += 1
    else:
        suffixes[sfx] += 1


for k, v in sorted(suffixes.items()):
    print(f"{k} : {v}")



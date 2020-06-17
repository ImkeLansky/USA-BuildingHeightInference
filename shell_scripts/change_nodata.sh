#!/bin/bash

shopt -s nullglob
for filename in *.tif;
do
    echo "Processing $filename"
    temp=${filename##*/}
    name="${temp%.tif}"
    gdal_translate -of GTiff -a_nodata -9999 $filename "$name"_NoData.tif
done
shopt -u nullglob # Revert nullglob back to it's normal default state
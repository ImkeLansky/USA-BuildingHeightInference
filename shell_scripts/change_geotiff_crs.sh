#!/bin/bash

shopt -s nullglob
for filename in *.tif;
do
    echo "Processing $filename"
    temp=${filename##*/}
    name="${temp%.tif}"
    gdalwarp -s_srs EPSG:4269 -t_srs EPSG:4326 $filename "$name"_4326.tif
done
shopt -u nullglob # Revert nullglob back to it's normal default state

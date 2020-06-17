#!/bin/bash

shopt -s nullglob
for filename in *.geojson;
do
    echo "Processing $filename"
    ogr2ogr -f "GeoJSON" ./Reprojected/ogr_$filename $filename -s_srs EPSG:4326 -t_srs EPSG:102004 -progress
done
shopt -u nullglob # Revert nullglob back to it's normal default state
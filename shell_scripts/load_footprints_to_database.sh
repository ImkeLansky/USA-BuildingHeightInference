#!/bin/bash

shopt -s nullglob
for filename in ../Reprojected/*.geojson;
do
    echo "Loading $filename"
    temp=${filename##*/}
    table="${temp%.geojson}"
    ogr2ogr -f "PostgreSQL" -a_srs EPSG:102004 PG:"host=localhost port=5432 user=postgres password=<password> dbname=<name>" $filename -nln $table -geomfield geom -progress
done
shopt -u nullglob # Revert nullglob back to it's normal default state
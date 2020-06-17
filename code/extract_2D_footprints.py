"""
Extract 2D building footprints from CityJSON from the Open City Model.
"""

import json
from shapely.geometry import Polygon
from geojson import Feature, FeatureCollection, dump
import numpy as np


def main():
    """
    Read files, and extract the ground surface.
    Store this as a new GeoJSON file.
    """

    # All files to convert with their destination file
    source_files = ["../Data/Astoria/OCM/Oregon-41007-000.json",
                    "../Data/Seattle/OCM/Washington-53033-004.json",
                    "../Data/Seattle/OCM/Washington-53033-016.json",
                    "../Data/Portland/OCM/Oregon-41051-000.json",
                    "../Data/Portland/OCM/Oregon-41051-001.json",
                    "../Data/Portland/OCM/Oregon-41051-002.json",
                    "../Data/Portland/OCM/Oregon-41051-003.json",
                    "../Data/Portland/OCM/Oregon-41051-004.json",
                    "../Data/Portland/OCM/Oregon-41051-005.json",
                    "../Data/Portland/OCM/Oregon-41051-006.json",
                    "../Data/Portland/OCM/Oregon-41051-007.json",
                    "../Data/SanDiego/OCM/California-06073-002.json",
                    "../Data/SanDiego/OCM/California-06073-003.json",
                    "../Data/SanDiego/OCM/California-06073-004.json",
                    "../Data/SanDiego/OCM/California-06073-012.json"]

    dest_files = ["../Data/Astoria/OCM/2D/Oregon-41007-000_2D.geojson",
                  "../Data/Seattle/OCM/2D/Washington-53033-004_2D.geojson",
                  "../Data/Seattle/OCM/2D/Washington-53033-016_2D.geojson",
                  "../Data/Portland/OCM/2D/Oregon-41051-000_2D.geojson",
                  "../Data/Portland/OCM/2D/Oregon-41051-001_2D.geojson",
                  "../Data/Portland/OCM/2D/Oregon-41051-002_2D.geojson",
                  "../Data/Portland/OCM/2D/Oregon-41051-003_2D.geojson",
                  "../Data/Portland/OCM/2D/Oregon-41051-004_2D.geojson",
                  "../Data/Portland/OCM/2D/Oregon-41051-005_2D.geojson",
                  "../Data/Portland/OCM/2D/Oregon-41051-006_2D.geojson",
                  "../Data/Portland/OCM/2D/Oregon-41051-007_2D.geojson",
                  "../Data/SanDiego/OCM/2D/California-06073-002_2D.geojson",
                  "../Data/SanDiego/OCM/2D/California-06073-003_2D.geojson",
                  "../Data/SanDiego/OCM/2D/California-06073-004_2D.geojson",
                  "../Data/SanDiego/OCM/2D/California-06073-012_2D.geojson"]

    for i, fname in enumerate(source_files):
        print(fname)

        with open(fname) as filepointer:
            data = json.load(filepointer)

        # Extract cityobjects and vertices list
        cityobjects = data['CityObjects']
        vertices = np.array(data['vertices'])

        features = []

        for obj_id in cityobjects:

            # Extract the list with indices of the vertices
            coord_idxs = cityobjects[obj_id]['geometry'][0]['boundaries']

            attributes = cityobjects[obj_id]['attributes']
            attributes['id'] = obj_id

            # Go over all these index sets and find the one where the
            # z-value is all zero -> ground surface
            for idx_set in coord_idxs[0]:
                coordinates = vertices[idx_set[0]]
                zeros = np.count_nonzero(coordinates[:, 2])

                if zeros == 0:
                    coords_2D = np.delete(coordinates, np.s_[2], axis=1)
                    footprint = Polygon(coords_2D)
                    break

            # Check for invalid polygons, fix them if invalid
            if not footprint.is_valid:
                print("Fixing invalid polygon. ID:", obj_id)
                footprint = footprint.buffer(0)

            # Create the geojson feature based on the geometry and attributes
            geojson_feature = Feature(geometry=footprint, properties=attributes)

            # Check if the features that we store are actually valid
            if not geojson_feature.is_valid:
                print("Invalid Feature. ID:", obj_id)

            features.append(geojson_feature)

        # Make putt all features in the geojson feature collection
        feature_collection = FeatureCollection(features)

        # Write the 2D footprints with their attributes to a new file
        with open(dest_files[i], 'w') as filepointer:
            dump(feature_collection, filepointer)


if __name__ == '__main__':
    main()

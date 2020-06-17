
# Source:
#-- a simple "extruder" to obtain CityJSON LoD1 Buildings from footprints
#-- Hugo Ledoux <h.ledoux@tudelft.nl>
#-- 2019-02-28
# Slight changes are made to get it to work for this use-case

import fiona
import shapely.geometry as sg
import json
import copy


def main():
    #-- read the input footprints
    c = fiona.open('../Data/Seattle/Visualisation/Seattle_Visualisation.shp')
    print("# of features: ", len(c))
    lsgeom = [] #-- list of the geometries
    lsattributes = [] #-- list of the attributes

    print("Reading all features.")

    num_parts = 2

    for each in c:
        lsgeom.append(sg.shape(each['geometry'])) #-- geom are casted to Fiona's
        lsattributes.append(each['properties'])

    step = len(c) // num_parts

    for i in range(0, num_parts):
        print("Create CityJSON part {0}".format(i))

        if i == 0:
            geoms = lsgeom[:step]
            attribs = lsattributes[:step]
        elif i == (num_parts - 1):
            geoms = lsgeom[step * i:]
            attribs = lsattributes[step * i:]
        else:
            geoms = lsgeom[step * (i - 1):step * i]
            attribs = lsattributes[step * (i - 1):step * i]

        #-- extrude to CityJSON
        cm = output_citysjon(geoms, attribs)

        #-- save the file to disk 'mycitymodel.json'

        print("Writing to file Seattle_{0}".format(i))
        json_str = json.dumps(cm)
        fname = "Seattle_" + str(i) + ".json"
        fout = open(fname, "w")
        fout.write(json_str)
        print("done.")


def output_citysjon(lsgeom, lsattributes):
    #-- create the JSON data structure for the City Model
    cm = {}
    cm["type"] = "CityJSON"
    cm["version"] = "1.0"
    cm["CityObjects"] = {}
    cm["vertices"] = []

    for (i,geom) in enumerate(lsgeom):
        if (i % 5000) == 0:
            print(i)

        footprint = geom
        #-- one building
        oneb = {}
        oneb['type'] = 'Building'
        oneb['attributes'] = {}
        oneb['attributes']['area'] = lsattributes[i]['area']
        oneb['attributes']['perimeter'] = lsattributes[i]['perimeter']
        oneb['attributes']['compactness'] = lsattributes[i]['compactnes']
        oneb['attributes']['num_vertices'] = lsattributes[i]['num_vertic']
        oneb['attributes']['length'] = lsattributes[i]['length']
        oneb['attributes']['width'] = lsattributes[i]['width']
        oneb['attributes']['slimness'] = lsattributes[i]['slimness']
        oneb['attributes']['complexity'] = lsattributes[i]['complexity']
        oneb['attributes']['num_adjacent_blds'] = lsattributes[i]['num_adjace']
        oneb['attributes']['num_neighbours'] = lsattributes[i]['num_neighb']
        oneb['attributes']['cbd'] = lsattributes[i]['cbd']
        oneb['attributes']['height_rfr'] = lsattributes[i]['height_rfr']
        oneb['attributes']['height_svr'] = lsattributes[i]['height_svr']
        oneb['attributes']['height_mlr'] = lsattributes[i]['height_mlr']
        oneb['geometry'] = [] #-- a cityobject can have >1
        #-- the geometry
        g = {}
        g['type'] = 'Solid'
        g['lod'] = 1
        allsurfaces = [] #-- list of surfaces forming the oshell of the solid
        #-- exterior ring of each footprint
        oring = list(footprint.exterior.coords)
        oring.pop() #-- remove last point since first==last
        if footprint.exterior.is_ccw == False:
            #-- to get proper orientation of the normals
            oring.reverse()
        extrude_walls(oring, lsattributes[i]['height_rfr'], allsurfaces, cm)
        #-- interior rings of each footprint
        irings = []
        interiors = list(footprint.interiors)
        for each in interiors:
            iring = list(each.coords)
            iring.pop() #-- remove last point since first==last
            if each.is_ccw == True:
                #-- to get proper orientation of the normals
                iring.reverse()
            irings.append(iring)
            extrude_walls(iring, lsattributes[i]['height_rfr'], allsurfaces, cm)
        #-- top-bottom surfaces
        extrude_roof_ground(oring, irings, lsattributes[i]['height_rfr'], False, allsurfaces, cm)
        extrude_roof_ground(oring, irings, 0, True, allsurfaces, cm)
        #-- add the extruded geometry to the geometry
        g['boundaries'] = []
        g['boundaries'].append(allsurfaces)
        #-- add the geom to the building
        oneb['geometry'].append(g)
        #-- insert the building as one new city object
        cm['CityObjects'][lsattributes[i]['id']] = oneb
    return cm


def extrude_roof_ground(orng, irngs, height, reverse, allsurfaces, cm):
    oring = copy.deepcopy(orng)
    irings = copy.deepcopy(irngs)
    if reverse == True:
        oring.reverse()
        for each in irings:
            each.reverse()
    for (i, pt) in enumerate(oring):
        cm['vertices'].append([pt[0], pt[1], height])
        oring[i] = (len(cm['vertices']) - 1)
    for (i, iring) in enumerate(irings):
        for (j, pt) in enumerate(iring):
            cm['vertices'].append([pt[0], pt[1], height])
            irings[i][j] = (len(cm['vertices']) - 1)
    # print(oring)
    output = []
    output.append(oring)
    for each in irings:
        output.append(each)
    allsurfaces.append(output)


def extrude_walls(ring, height, allsurfaces, cm):
    #-- each edge become a wall, ie a rectangle
    for (j, v) in enumerate(ring[:-1]):
        l = []
        cm['vertices'].append([ring[j][0],   ring[j][1],   0])
        cm['vertices'].append([ring[j+1][0], ring[j+1][1], 0])
        cm['vertices'].append([ring[j+1][0], ring[j+1][1], height])
        cm['vertices'].append([ring[j][0],   ring[j][1],   height])
        t = len(cm['vertices'])
        allsurfaces.append([[t-4, t-3, t-2, t-1]])
    #-- last-first edge
    l = []
    cm['vertices'].append([ring[-1][0], ring[-1][1], 0])
    cm['vertices'].append([ring[0][0],  ring[0][1],  0])
    cm['vertices'].append([ring[0][0],  ring[0][1],  height])
    cm['vertices'].append([ring[-1][0], ring[-1][1], height])
    t = len(cm['vertices'])
    allsurfaces.append([[t-4, t-3, t-2, t-1]])


if __name__ == '__main__':
    main()
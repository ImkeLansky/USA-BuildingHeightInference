"""
Extract geometric features from 2D building footprints.
"""

from time import time
import db_funcs


def compute_areas(cursor, table):
    """
    Compute the areas of the building footprints. Store results.
    """

    print("{0} >> Computing footprint area".format(table))
    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS area DOUBLE PRECISION;")

    cursor.execute("UPDATE " + table + "_tmp SET area = subquery.area " +
                   "FROM (SELECT id, ST_AREA(geom_repr) AS area FROM " + table +
                   "_tmp) AS subquery WHERE " + table + "_tmp.id = subquery.id;")


def compute_perimeter(cursor, table):
    """
    Compute the perimeter of the building footprints. Store results.
    """

    print("{0} >> Computing footprint perimeter".format(table))
    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS perimeter " + \
                   "DOUBLE PRECISION;")

    cursor.execute("UPDATE " + table + "_tmp SET perimeter = subquery.perimeter " +
                   "FROM (SELECT id, ST_PERIMETER(geom_repr) AS perimeter FROM " +
                   table + "_tmp) AS subquery WHERE " + table + "_tmp.id = subquery.id;")


def compute_compactness(cursor, table):
    """
    Compute the shape compactness of the building footprints. This uses the
    Normalized Perimeter Index (NPI), which measures how close a shape is to
    as circle. Store results.
    """

    print("{0} >> Computing footprint compactness".format(table))
    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS compactness " +
                   "DOUBLE PRECISION;")

    cursor.execute("UPDATE " + table + "_tmp SET compactness = subquery.compactness " +
                   "FROM (SELECT id, ((2 * sqrt(PI() * area)) / perimeter) AS compactness " +
                   " FROM " + table + "_tmp) AS subquery WHERE " + table + "_tmp.id = subquery.id;")


def compute_num_vertices(cursor, table):
    """
    Compute the number of vertices in a polygon. Apply Douglas-Peucker with a low
    tolerance value to remove collinear points. Store results.
    """

    print("{0} >> Computing footprint #vertices".format(table))
    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS num_vertices INTEGER;")

    cursor.execute("UPDATE " + table + "_tmp SET num_vertices = subquery.num_vertices " +
                   "FROM (SELECT id, ST_NPoints(ST_SimplifyPreserveTopology(geom_repr, 0.1))" +
                   " as num_vertices FROM " + table + "_tmp) AS subquery WHERE "
                   + table + "_tmp.id = subquery.id;")


def create_mbr(cursor, table):
    """
    Create a minimum bounding rectangle (MBR) around all footprints.
    This MBR can be rotated so that it is the smalles rectangle around the footprint.
    Store resutls.
    """

    print("{0} >> Computing footprint MBR".format(table))
    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS bbox GEOMETRY;")

    cursor.execute("UPDATE " + table + "_tmp SET bbox = subquery.bbox " +
                   "FROM (SELECT id, ST_OrientedEnvelope(geom_repr) as bbox FROM " + table +
                   "_tmp) AS subquery WHERE " + table + "_tmp.id = subquery.id;")

    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS side_1 " +
                   "DOUBLE PRECISION;")
    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS side_2 " +
                   "DOUBLE PRECISION;")

    # Store the length of the two sides of the MBR as well.
    cursor.execute("UPDATE " + table + "_tmp SET side_1 = subquery.side_1, " +
                   "side_2 = subquery.side_2 FROM (SELECT id, " +
                   "ST_Distance(ST_Point(ST_Xmin(bbox), ST_Ymin(bbox)), " +
                   "ST_Point(ST_Xmin(bbox), ST_Ymax(bbox))) as side_1, " +
                   "ST_Distance(ST_Point(ST_Xmin(bbox), ST_Ymin(bbox)), " +
                   "ST_Point(ST_Xmax(bbox), ST_Ymin(bbox))) as side_2 " +
                   "FROM " + table + "_tmp) AS subquery WHERE " + table + "_tmp.id = subquery.id;")

    cursor.execute("ALTER TABLE " + table + "_tmp DROP COLUMN bbox;")


def compute_width_length(cursor, table):
    """
    Compute the lenght of the footprint. This is defined as the longest
    edge of the minimum bounding rectangle (MBR). Also compute the width,
    the shorest side of the MBR. Store results.
    """

    create_mbr(cursor, table)

    print("{0} >> Computing footprint width and length".format(table))
    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS length DOUBLE PRECISION;")
    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS width DOUBLE PRECISION;")

    cursor.execute("UPDATE " + table + "_tmp SET length = subquery.length, width = subquery.width" +
                   " FROM (SELECT id, CASE WHEN side_1 > side_2 THEN side_1 " +
                   "ELSE side_2 END AS length, CASE WHEN side_1 < side_2 THEN side_1 " +
                   "ELSE side_2 END AS width FROM " + table + "_tmp) AS subquery WHERE "
                   + table + "_tmp.id = subquery.id;")

    # Drop the helper columns.
    cursor.execute("ALTER TABLE " + table + "_tmp DROP COLUMN side_1;")
    cursor.execute("ALTER TABLE " + table + "_tmp DROP COLUMN side_2;")


def compute_slimness(cursor, table):
    """
    Compute the slimness of the footprint. This is defined by the length
    of the footprint divided by the width of the footprint. Store results.
    """

    print("{0} >> Computing footprint slimness".format(table))
    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS slimness DOUBLE PRECISION;")

    cursor.execute("UPDATE " + table + "_tmp SET slimness = subquery.slimness " +
                   "FROM (SELECT id, (length / width) as slimness FROM " + table +
                   "_tmp) AS subquery WHERE " + table + "_tmp.id = subquery.id;")


def compute_complexity(cursor, table):
    """
    Compute the shape complexity of the footprints. More irregularities
    in the shape means that the footprint is more complex. Store results.
    """

    print("{0} >> Computing footprint complexity".format(table))
    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS complexity " +
                   "DOUBLE PRECISION;")

    cursor.execute("UPDATE " + table + "_tmp SET complexity = subquery.complexity " +
                   "FROM (SELECT id, (perimeter / POWER(area, 0.25)) AS complexity " +
                   "FROM " + table + "_tmp) AS subquery WHERE " + table + "_tmp.id = subquery.id;")


def in_cbd(cursor, table):
    """
    Check for each footprint in the table if they lie in a neighbourhood
    classified as a central business district (CBD) or not.
    """

    print("{0} >> Checking if footprint is in CBD".format(table))
    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS cbd INTEGER;")

    # Create spatial index.
    cursor.execute("CREATE INDEX IF NOT EXISTS cbd_idx ON cbds USING GIST (geom);")

    cursor.execute("UPDATE " + table + "_tmp SET cbd = 1 " +
                   "FROM (SELECT t.id FROM " + table + " AS t JOIN cbds as cbd" +
                   " ON ST_Contains(cbd.geom, t.geom_repr)) AS subquery " +
                   "WHERE " + table + "_tmp.id = subquery.id;")

    cursor.execute("UPDATE " + table + "_tmp SET cbd = 0 WHERE cbd IS NULL;")


def compute_buffers(cursor, table, size):
    """
    Copmute buffers around all the building geometries. Buffer can be of
    different sizes. Store result.
    """

    print("{0} >> Computing buffer of {1}m around footprint".format(table, size))

    # # Create buffer field.
    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS buffer GEOMETRY;")

    cursor.execute("UPDATE " + table + "_tmp SET buffer = subquery.buffer " +
                   "FROM (SELECT id, ST_Buffer(geom_repr, " + str(size) + ") AS buffer FROM "
                   + table + "_tmp) AS subquery WHERE " + table + "_tmp.id = subquery.id;")

    # Create spatial index.
    cursor.execute("CREATE INDEX IF NOT EXISTS buf_idx_" + table + "_tmp ON " + table +
                   "_tmp USING GIST (buffer);")


def compute_num_adjacent_blds(cursor, table):
    """
    Compute the number of neighbours for buffer intersections.
    Store results. If NULL value, set number to 0.
    """

    compute_buffers(cursor, table, 1)

    print("{0} >> Computing #adjacent buildings".format(table))
    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS num_adjacent_blds INTEGER;")

    cursor.execute("UPDATE " + table + "_tmp SET num_adjacent_blds = subquery.num_adjacent_blds " +
                   "FROM (SELECT a.id, COUNT(*) as num_adjacent_blds " +
                   "FROM " + table + "_tmp AS a JOIN " + table + "_tmp AS b ON " +
                   "ST_INTERSECTS(a.buffer, b.geom_repr) " +
                   "WHERE (a.id != b.id) GROUP BY a.id) AS subquery " +
                   "WHERE " + table + "_tmp.id = subquery.id;")

    cursor.execute("UPDATE " + table + "_tmp SET num_adjacent_blds = 0 WHERE " +
                   "num_adjacent_blds IS NULL;")

    # Drop the buffer column and its index.
    cursor.execute("DROP INDEX buf_idx_" + table + "_tmp;")
    cursor.execute("ALTER TABLE " + table + "_tmp DROP COLUMN buffer;")


def compute_centroids(cursor, table):
    """
    Compute the centroid for each building. Store result.
    """

    print("{0} >> Computing footprint centroid".format(table))

    # Create centroid field.
    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS centroid GEOMETRY;")

    cursor.execute("UPDATE " + table + "_tmp SET centroid = subquery.centroid " +
                   "FROM (SELECT id, ST_Centroid(geom_repr) as centroid " +
                   "FROM " + table + "_tmp) AS subquery " +
                   "WHERE " + table + "_tmp.id = subquery.id;")

    # Create spatial index.
    cursor.execute("CREATE INDEX IF NOT EXISTS centroid_idx_" + table + "_tmp ON " + table +
                   "_tmp USING GIST (centroid);")


# https://www.compose.com/articles/geofile-everything-in-the-radius-with-postgis/
def compute_num_neighbours(cursor, table, dist):
    """
    Compute the number of neighbours for using centroid distances.
    Store results. If NULL value, set number to 0.
    """

    compute_centroids(cursor, table)

    print("{0} >> Computing #neighbours".format(table))
    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS num_neighbours INTEGER;")

    cursor.execute("UPDATE " + table + "_tmp SET num_neighbours = subquery.num_intersects " +
                   "FROM (SELECT a.id, COUNT(*) as num_intersects " +
                   "FROM " + table + "_tmp AS a JOIN " + table + "_tmp AS b ON " +
                   "ST_DWithin(a.centroid, b.centroid, " + str(dist) + ") " +
                   "WHERE (a.id != b.id) GROUP BY a.id) AS subquery " +
                   "WHERE " + table + "_tmp.id = subquery.id;")

    cursor.execute("UPDATE " + table + "_tmp SET num_neighbours = 0 WHERE num_neighbours IS NULL;")

    # Drop the centroid column its index.
    cursor.execute("DROP INDEX centroid_idx_" + table + "_tmp;")
    cursor.execute("ALTER TABLE " + table + "_tmp DROP COLUMN centroid;")


def reproject_coords(cursor, table, dest_crs):
    """
    Reproject the coordinates from EPSG:4326 to a destination CRS. Store results.
    Then update the SRID in the database to make sure it is not set to 0.
    """

    print("{0} >> Reprojecting all footprint geomteries".format(table))
    cursor.execute("ALTER TABLE " + table + "_tmp ADD COLUMN IF NOT EXISTS geom_repr GEOMETRY;")

    cursor.execute("UPDATE " + table + "_tmp SET geom_repr = subquery.reprojection " +
                   "FROM (SELECT id, ST_Transform(geom, " + str(dest_crs) + ") as reprojection " +
                   "FROM " + table + "_tmp) AS subquery " +
                   "WHERE " + table + "_tmp.id = subquery.id;")

    cursor.execute("SELECT UpdateGeometrySRID('" + table + "_tmp', 'geom_repr', "
                   + str(dest_crs) + ");")

    cursor.execute("CREATE INDEX IF NOT EXISTS bld_idx_" + table + " ON " + table +
                   "_tmp USING GIST (geom_repr);")


def main():
    """
    Call all the functions for extracting the features for all specified tables.
    """
    connection = db_funcs.setup_connection("training")
    connection.autocommit = True
    cursor = connection.cursor()

    # Loop over all tables in the database and perform the feature extractino
    # for all of them.
    for table in db_funcs.unique_tables(cursor):

        # The data from the CBDS should be excluded from the
        # feature exraction. This is only used to check what type of area
        # the buildings lie in.
        if table == 'cbds':
            continue

        starttime = time()

        # Select the correct CRS, depends on US or Canadian data.
        if table == 'toronto':
            crs = 102002
        else:
            crs = 102004

        cursor.execute("DROP TABLE IF EXISTS " + table + "_tmp CASCADE;")
        cursor.execute("CREATE UNLOGGED TABLE " + table + "_tmp AS TABLE " + table + ";")
        cursor.execute("ALTER TABLE " + table + "_tmp ADD PRIMARY KEY (id);")
        # cursor.execute("CREATE INDEX IF NOT EXISTS bld_idx_" + table + "_tmp ON " + table +
        #                "_tmp USING GIST (geom_repr);")

        reproject_coords(cursor, table, crs)
        compute_areas(cursor, table)
        compute_perimeter(cursor, table)
        compute_compactness(cursor, table)
        compute_num_vertices(cursor, table)
        compute_width_length(cursor, table)
        compute_slimness(cursor, table)
        compute_complexity(cursor, table)
        in_cbd(cursor, table)
        compute_num_adjacent_blds(cursor, table)
        compute_num_neighbours(cursor, table, 100)

        print("{0} >> Copying unlogged table to logged table".format(table))

        cursor.execute("CREATE TABLE " + table + "_new AS TABLE " + table + "_tmp;")
        cursor.execute("DROP TABLE " + table + ";")
        cursor.execute("ALTER TABLE " + table + "_new RENAME TO " + table + ";")
        cursor.execute("DROP TABLE " + table + "_tmp;")

        endtime = time()
        duration = endtime - starttime
        print(">> Computation time: ", round(duration, 2), "s")
        print(80*"-")

    db_funcs.close_connection(connection, cursor)


if __name__ == '__main__':
    main()

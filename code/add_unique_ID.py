import os
import glob
import json
from time import time

states = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'DistrictofColumbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'NewHampshire': 'NH',
    'NewJersey': 'NJ',
    'NewMexico': 'NM',
    'NewYork': 'NY',
    'NorthCarolina': 'NC',
    'NorthDakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'RhodeIsland': 'RI',
    'SouthCarolina': 'SC',
    'SouthDakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'WestVirginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}


def main():
    files = glob.glob("../Data/USBuildingFootprints/*.geojson")
    count_datasets = 1

    for dataset in files:
        starttime = time()
        progress = str(count_datasets) + "/" + str(len(files))

        print("> Reading (" + progress + "):", dataset)
        with open(dataset, 'r') as fp:
            data = json.load(fp)

        count = 0
        statename = os.path.splitext(os.path.basename(dataset))[0]
        for feature in data['features']:
            feature['id'] = states[statename] + "_" + str(count)
            count += 1

        dest_file = "../Data/USBuildingFootprints/IDs/" + statename + "_ID.geojson"

        print("> Writing (" + progress + "):", dest_file)
        with open(dest_file, 'w') as fp:
            fp.write(json.dumps(data))

        endtime = time()
        duration = endtime - starttime
        print("> Time: ", round(duration, 2), "s")
        print(80*"-")

        count_datasets += 1


if __name__ == '__main__':
    main()

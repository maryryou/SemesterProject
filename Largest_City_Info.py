#https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
#Distance Calculation and Nearest Neighbor Calulation sourced from website above
# Example of getting neighbors for an instance
from geopy.distance import geodesic
from math import sqrt
import pandas as pd
from itertools import product


def preprocess(x):
    terrordata_cities = pd.merge(x,cities,on='key').drop('key',axis=1)
    terrordata_cities.to_csv("terrordata_cities_2.csv", mode="a", header=False, index=False)


def distancer(row):
        try:
                
            coords_1 = eval(row['Event_lat_long'])
            coords_2 = eval(row['city_lat_long'])
            miles = float(geodesic(coords_1, coords_2).miles)
            return miles
        except:
            return 20000


Path = 'C:\\Users\\Grant\\Desktop\\Random\\CS_Group_Project\\'
cities = pd.read_csv("C:\\Users\\Grant\\Desktop\Random\\CS_Group_Project\\worldcities.csv")

cities = cities[cities["capital"] == "primary"]
cities['city_lat_long'] = list(zip(cities['lat'], cities['lng']))
cities = cities[['city','city_lat_long','population']]
cities['key'] = 1

terrordata = pd.read_csv(
     Path + 'GDT_2000_2019.csv')


#terrordata_condensed = terrordata.groupby(['latitude', 'longitude']).size().reset_index(name='counts')

#terrordata_condensed.to_csv("terrordata_condensed.csv", mode="w", header=True, index=False)

terrordata = pd.read_csv(
     Path + 'terrordata_condensed.csv', chunksize=10000)

for r in terrordata:
        r['lat_long'] = list(zip(r.latitude, r.longitude))
        r = r[["lat_long", 'counts']]
        r["key"] = 1
        print(r)
        preprocess(r) 




header_list = ["Event_lat_long","Counts", "City", "city_lat_long", "population"]

Geo_Data = pd.read_csv("C:\\Users\\Grant\\Desktop\Random\\CS_Group_Project\\terrordata_cities_2.csv", names=header_list, chunksize=100000)

chunk_count = 0

count = 0

capital_events = pd.DataFrame(columns = ["Event_lat_long", 'counts', "City", "city_lat_long","population" ,'Close_To_Capital_City'])


for r in Geo_Data:
        r['Miles'] = r.apply(lambda row : distancer(row), axis = 1)

        r.loc[r['Miles'] <= 20, 'Close_To_Capital_City'] = 'True' 
        Capital_City_Event = r[r['Close_To_Capital_City'] == 'True']
        capital_events = capital_events.append(Capital_City_Event)
        Capital_City_Event.to_csv("Events_Close_To_Capital.csv", mode="a", header=False, index=False)
        print(Capital_City_Event)
        chunk_count = chunk_count + 1
        print(chunk_count)


grouped_events = capital_events.describe()


print(grouped_events)





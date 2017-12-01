import pandas as pd
from urllib.request import Request, urlopen
import csv
import datetime
import json
import pdb

def get_csv_by_year(startyear,endyear):

    for year in range(startyear, endyear) :
        url = 'http://developer.nrel.gov/api/solar/nsrdb_0512_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)
        print(url)
        try:
            page = urlopen(url)
        except urllib.error.HTTPError:
            print("Exception occured ! Starting now from "+year)
            get_csv_by_year(year, endyear)
        req = Request(url,headers={'User-Agent': 'Mozilla/5.0'})
        content = page.read().decode()
        weatherCSV = open(str(year)+".csv","w")
        weatherCSV.write(content)

zipcode = 85281
location="Chicago"
lat_long_api="https://maps.googleapis.com/maps/api/geocode/json?address="+location
#lat_long_api = "http://maps.googleapis.com/maps/api/geocode/json?address="+str(zipcode)
page = urlopen(lat_long_api)
req = Request(lat_long_api, headers={'User-Agent': 'Mozilla/5.0'})
content = json.loads(page.read().decode())
# pdb.set_trace()
lat_long = content["results"][0]["geometry"]["location"]
lat = lat_long["lat"]
lon = lat_long["lng"]


api_key = 'bc9ezc45D4YpJhKyZ5ywjprQCTDFVtkx6JjYUNE1'

attributes = 'ghi,wind_speed_10m_nwp,surface_air_temperature_nwp'
leap_year = 'false'
# Set time interval in minutes to '30'.
interval = '30'
# Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
utc = 'false'
your_name = 'Rasika'
# Your reason for using the NSRDB.
reason_for_use = 'beta+testing'
# Your affiliation
your_affiliation = 'ASU'
# Your email address
your_email = 'js@sas.edu'
# Please join our mailing list so we can keep you up-to-date on new developments.
mailing_list = 'true'
y = datetime.date.today().year
year=y-1


# url = 'http://developer.nrel.gov/api/solar/nsrdb_0512_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)
# print (url)
# page = urlopen(url)
# req = Request(url,headers={'User-Agent': 'Mozilla/5.0'})
# content = page.read().decode()
# #print(content)
# weatherCSV = open(year+".csv","w")
# weatherCSV.write(content)
get_csv_by_year(year-15, year)






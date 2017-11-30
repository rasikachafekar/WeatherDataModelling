import pandas as pd
from urllib.request import Request, urlopen
import csv
import datetime

lat, lon = 33.4255104, -111.9400054

api_key = '3KEle87nPOrTBNXBVWeXqChGztQm0pvTLbepwVwW'

attributes = 'ghi,wind_speed_10m_nwp,surface_air_temperature_nwp'
leap_year = 'false'
# Set time interval in minutes to '30'.
interval = '30'
# Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
utc = 'false'
your_name = 'John+Smith'
# Your reason for using the NSRDB.
reason_for_use = 'beta+testing'
# Your affiliation
your_affiliation = 'my+institution'
# Your email address
your_email = 'john.smith@server.com'
# Please join our mailing list so we can keep you up-to-date on new developments.
mailing_list = 'true'
year = datetime.date.today().year


# url = 'http://developer.nrel.gov/api/solar/nsrdb_0512_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)
# page = urlopen(url)
# req = Request(url,headers={'User-Agent': 'Mozilla/5.0'})
# content = page.read().decode()
# #print(content)
# weatherCSV = open(year+".csv","w")
# weatherCSV.write(content)

for year in range(year, year-15 ,-1) :
    url = 'http://developer.nrel.gov/api/solar/nsrdb_0512_download.csv?wkt=POINT({lon}%20{lat})&names={year}&interval={interval}&utc=false&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, interval=interval, utc=utc, api=api_key,attr=attributes)
    page = urlopen(url)
    req = Request(url,headers={'User-Agent': 'Mozilla/5.0'})
    content = page.read().decode()
    weatherCSV = open(year+".csv","w")
    weatherCSV.write(content)


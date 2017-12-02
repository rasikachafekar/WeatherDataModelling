from __future__ import print_function

import pymysql

def get_CityandState(locationid):

    conn = pymysql.connect(host='ift540.cyhc1qzz7e7u.us-west-2.rds.amazonaws.com', port=3306, user='IFT540PSP', passwd='IFT540PSP', db='pvsystem')

    cur = conn.cursor()

    cur.execute("SELECT * from Location")

    print(cur.description)

    print()

    city=""
    state=""
    dbCallReturn = []
    for row in cur:
        dbCallReturn.append(row)

    for i in range(0, len(dbCallReturn)):
        print(dbCallReturn[i])
        if locationid == dbCallReturn[i][0]:
            city = dbCallReturn[i][1]
            state = dbCallReturn[i][2]
    cur.close()
    conn.close()

    if city != "" and state != "":
        print("Found "+city+" "+state)
        return city, state
    else:
        print("Not Found")
        return "This location does not exist in the database !"

get_CityandState(5)

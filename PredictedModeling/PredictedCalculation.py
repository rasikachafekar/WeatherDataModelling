import math
import csv
import pandas
from lxml import etree
import matplotlib.pyplot as plt
from xml.etree.ElementTree import Element, SubElement, tostring
df = pandas.read_csv('AguafriaWeather.csv', encoding = "ISO-8859-1")
Tempvals = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
WSvals = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
Gvals = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
daysCount = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
root = etree.Element('Results')
l = 0
h = 31
e = 30
o = 31
for i in range(0, 7):
    if(i == 1):
        Tempvals[i] = df[ : ][31 : 59].mean()[2]
        WSvals[i] = df[ : ][31 : 59].mean()[3]
        Gvals[i] = df[ : ][31 : 59].mean()[5]
        h = 90
        l = 59
    elif((i + 1) % 2 == 1):
        Tempvals[i] = df[ : ][l : h].mean()[2]
        WSvals[i] = df[ : ][l : h].mean()[3]
        Gvals[i] = df[ : ][l : h].mean()[5]
        l = h
        h = h + o - 1
    elif((i + 1) % 2 == 0):
        Tempvals[i] = df[ : ][l : h].mean()[2]
        WSvals[i] = df[ : ][l : h].mean()[3]
        Gvals[i] = df[ : ][l : h].mean()[5]
        l = h
        h = h + e
        if(i == 5):
            h += 1
h+=1
for i in range(7, 12):
    if ((i + 1) % 2 == 0):
        Tempvals[i] = df[ : ][l : h].mean()[2]
        WSvals[i] = df[ : ][l : h].mean()[3]
        Gvals[i] = df[ : ][l : h].mean()[5]
        l = h
        h = h + e
    elif ((i + 1) % 2 == 1):
        Tempvals[i] = df[ : ][l : h].mean()[2]
        WSvals[i] = df[ : ][l : h].mean()[3]
        Gvals[i] = df[ : ][l : h].mean()[5]
        l = h
        h = h + o
data = csv.reader(open('AguafriaSystemInfo.csv', 'r'), delimiter=",", quotechar='|')
modules = []
Ppredtargvals = []
for row in data:
    modules.append(row[0])
    Ppredtargvals.append(row[6])
Ttrc = 25
Gtrc = 1
a = -3.47
b = -0.0594
dTcond = 3
Ppredvals = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
for j in range(1, 3):
    system = etree.Element('System')
    system.text = modules[j]
    location = etree.Element('Location')
    location.text = 'Agua Fria'
    system.append(location)
    year = etree.Element('Year')
    value = etree.Element('Value')
    value.text = '2003'
    year.append(value)
    print('Module: ' + str(modules[j]))
    print('Predicted Energy Values: ')
    print('Month'+' '+'Predicted Energy(MWh)')
    for i in range(0, 12):
        Tm = Gvals[i] * (math.exp(a + b * WSvals[i])) + Tempvals[i]
        Tc = Tm + (Gvals[i] / 1000) * dTcond
        CFtcell = 1 + (1 / (Tc - Ttrc))
        Ppredvals[i] = float(Ppredtargvals[j]) * (Gvals[i] / Gtrc) * CFtcell
        Ppredvals[i] = (Ppredvals[i]*daysCount[i]*24)/ 1000000;
        month = etree.Element('Month')
        name = etree.Element('Name')
        name.text = monthNames[i]
        energy = etree.Element('Energy')
        energy.text = str.encode(str(Ppredvals[i]))
        month.append(name)
        month.append(energy)
        year.append(month)
        print(str(monthNames[i])+' '+str(Ppredvals[i]))
    system.append(year)
    root.append(system)
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    xTicks = monthNames
    plt.xticks(x, xTicks)
    plt.xticks(range(12), xTicks, rotation=45)
    plt.plot(x, Ppredvals)
    plt.title('Predicted Energy Values Module No:'+str(modules[j]))
    plt.xlabel('Month')
    plt.ylabel('Predicted Energy(MWh)')
    plt.show()
filename = "results.xml"
file_ = open(filename, 'w')
file_.write(tostring(root).decode())
file_.close()





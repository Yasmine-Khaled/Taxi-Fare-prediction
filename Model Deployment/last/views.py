from django.shortcuts import render
from django.http import HttpResponse
import manage
import pickle
from geopy.geocoders import Nominatim
import math
import pandas as pd
import numpy as np
from datetime import datetime

X_scaler = pickle.load(open('x_scaler.sav', 'rb'))
y_scaler = pickle.load(open('y_scaler.sav', 'rb'))
model = pickle.load(open('LinearRegression.sav', 'rb'))
columns = pickle.load(open('columns.sav', 'rb'))

def airport_distance(lon, lat):
    jfk_coord = (math.radians(40.639722), math.radians(-73.778889))
    ewr_coord = (math.radians(40.6925), math.radians(-74.168611))
    lga_coord = (math.radians(40.77725), math.radians(-73.872611))
    sol_coord = (math.radians(40.6892), math.radians(-74.0445))
    nyc_coord = (math.radians(40.7141667), math.radians(-74.0063889))

    jfk_distance = math.acos(
        math.sin(lat) * math.sin(jfk_coord[0]) + math.cos(lat) * math.cos(jfk_coord[0]) * math.cos(lon - jfk_coord[1]))
    ewr_distance = math.acos(
        math.sin(lat) * math.sin(ewr_coord[0]) + math.cos(lat) * math.cos(ewr_coord[0]) * math.cos(lon - ewr_coord[1]))
    lga_distance = math.acos(
        math.sin(lat) * math.sin(lga_coord[0]) + math.cos(lat) * math.cos(lga_coord[0]) * math.cos(lon - lga_coord[1]))
    sol_distance = math.acos(
        math.sin(lat) * math.sin(sol_coord[0]) + math.cos(lat) * math.cos(sol_coord[0]) * math.cos(lon - sol_coord[1]))
    nyc_distance = math.acos(
        math.sin(lat) * math.sin(nyc_coord[0]) + math.cos(lat) * math.cos(nyc_coord[0]) * math.cos(lon - nyc_coord[1]))

    return jfk_distance, ewr_distance, lga_distance, sol_distance, nyc_distance


def haversine_distance(lat1, lon1, lat2, lon2):
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    radius = 6371

    distance = radius * c
    return distance


def calculate_bearing(lat1, lon1, lat2, lon2):
    delta_lon = lon2 - lon1

    theta = math.atan2(math.sin(delta_lon) * math.cos(lat2),
                       math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))

    bearing = math.degrees(theta)
    bearing = (bearing + 360) % 360

    return math.radians(bearing)

def geocode_to_radians(pickup_address, dropoff_address):
    # Create a geocoder instance
    geolocator = Nominatim(user_agent="my_geocoder", timeout=None)

    # Geocode the pickup address
    pickup_location = geolocator.geocode(pickup_address)
    pickup_latitude_rad = math.radians(pickup_location.latitude)
    pickup_longitude_rad = math.radians(pickup_location.longitude)

    # Geocode the drop-off address
    dropoff_location = geolocator.geocode(dropoff_address)
    dropoff_latitude_rad = math.radians(dropoff_location.latitude)
    dropoff_longitude_rad = math.radians(dropoff_location.longitude)

    return pickup_latitude_rad, pickup_longitude_rad, dropoff_latitude_rad, dropoff_longitude_rad

# Create your views here.
def welcome(request):
    return render(request, 'index.html')


def fun2(request):
    manage.name = request.GET['name']
    return render(request, 'order.html', {"name": manage.name})

def fun3(request):
    pickup_address = request.GET['pickup']
    dropoff_address = request.GET['dropoff']
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude = geocode_to_radians(pickup_address, dropoff_address)
    data = {}
    data['pickup_longitude'] = float(pickup_longitude)
    data['pickup_latitude'] = float(pickup_latitude)
    data['dropoff_longitude'] = float(dropoff_longitude)
    data['dropoff_latitude'] = float(dropoff_latitude)
    data['passenger_count'] = int(request.GET['passenger_count'])
    data['hour'] = datetime.now().hour
    data['month'] = datetime.now().month
    data['weekday'] = float(datetime.now().weekday())
    data['year'] = datetime.now().year
    data['bearing'] = calculate_bearing(data['pickup_latitude'], data['pickup_longitude'], data['dropoff_latitude'],
                                        data['dropoff_longitude'])

    data['pickup_jfk_distance'], data['pickup_ewr_distance'], data['pickup_lga_distance'], \
    data['pickup_sol_distance'], data['pickup_nyc_distance'] = airport_distance(
        data['pickup_longitude'],
        data['pickup_latitude'])
    data['dropoff_jfk_distance'], data['dropoff_ewr_distance'], data['dropoff_lga_distance'], \
    data['dropoff_sol_distance'], data['dropoff_nyc_distance'] = airport_distance(
        data['dropoff_longitude'],
        data['dropoff_latitude'])
    data['distance'] = haversine_distance(data['pickup_latitude'], data['pickup_longitude'], data['dropoff_latitude'],
                                          data['dropoff_longitude'])

    df = pd.DataFrame(data, index=[0])
    to_drop = list(set(df.columns) - set(columns))
    df.drop(columns=to_drop, inplace=True)
    print(df.loc[0, :].values.tolist())
    print(columns)
    print("\n\n")
    print(df.columns)
    X_test = pd.DataFrame(X_scaler.transform(df), columns=columns)
    print(X_test.head())
    y = model.predict(X_test)
    y = y_scaler.inverse_transform(y.reshape(-1, 1))
    return render(request, 'ticket.html', {"name": manage.name, "passenger_count": request.GET['passenger_count'], "fare_amount": round(float(y[0][0]), 3)})

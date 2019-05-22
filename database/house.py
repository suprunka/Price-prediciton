from database.dbConnection import connect_to_houses
import random
import datetime


class House:
    def __init__(self, id, date=None, price=0, bedrooms=0, bathrooms=0, sqft_living=0, sqft_lot=0,
                 floors=0, waterfront=0, view=0, condition=0,
                 grade=0, sqft_above=0, sqft_basement=0, yr_built=0, yr_renovated=0, zipcode=0, lat=0, long=0,
                 sqft_living15=0, sqft_lot15=0):
        self.id = id
        self.date = date
        self.price = price
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.sqft_living = sqft_living
        self.sqft_lot = sqft_lot
        self.floors = floors
        self.waterfront = waterfront
        self.view = view
        self.condition = condition
        self.grade = grade
        self.sqft_above = sqft_above
        self.sqft_basement = sqft_basement
        self.yr_built = yr_built
        self.yr_renovated = yr_renovated
        self.zipcode = zipcode
        self.lat = lat
        self.long = long
        self.sqft_living15 = sqft_living15
        self.sqft_lot15 = sqft_lot15

    def __call__(self, kwargs):
        self.__dict__.update(kwargs)
        return self


def create_house():
    id = "\"" + str(round(random.randint(1, 600000) * datetime.datetime.now().second + 0.1 / 55 / 0.02)) + "\""
    result = connect_to_houses().find_one({"id": id})
    if result is not None:
        create_house()
    else:
        return House(id)


def set_properties(house, dictionary):
    house.__call__(dictionary)
    return house


def add_house(house):
    house.sqft_above = str(round(float(house.sqft_above) * 10.76391, 0))
    house.sqft_basement = str(round(float(house.sqft_basement) * 10.76391, 0))
    house.sqft_lot = str(round(float(house.sqft_lot) * 10.76391, 0))
    house.sqft_living = str(round(float(house.sqft_living) * 10.76391, 0))
    house.date = "\"" + house.date.replace('-', '') + 'T000000' + "\""
    house.floors = "\"" + house.floors + "\""
    house.zipcode = "\"" + house.zipcode + "\""
    house.waterfront = str(house.waterfront)
    house.view = str(house.view)
    house.sqft_living15 = str(house.sqft_living15)
    house.sqft_lot15 = str(house.sqft_lot15)

    connect_to_houses().insert_one(house.__dict__)

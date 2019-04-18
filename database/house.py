import random
import datetime
class House:
    def __init__(self, id, price=0, bedrooms=0, bathrooms=0, sqm_living=0, sqm_lot=0, floors=0, view=0, condition=0,
                 grade=0, sqm_above=0,sqm_basement=0, yr_built=0, yr_renovated=0, zipcode=0, latitude=0, longitude=0):
        self.id = id
        self.price = price
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.sqm_living = sqm_living
        self.sqm_lot = sqm_lot
        self.floors = floors
        self.view = view
        self.condition = condition
        self.grade = grade
        self.sqm_above = sqm_above
        self.sqm_basement = sqm_basement
        self.yr_built = yr_built
        self.yr_renovated = yr_renovated
        self.zipcode = zipcode
        self.latitude = latitude
        self.longitude = longitude

    def __call__(self, kwargs):
        self.__dict__.update(kwargs)
        return self


def create_house():
    return House(round(random.randint(1, 600000) / datetime.datetime.now().second+0.1 / 0.02), 0)


def set_properties(house, dictionary):
    house.__call__(dictionary)
    return house
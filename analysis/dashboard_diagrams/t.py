from database import house
from database import dbConnection
housing_n =dbConnection.get_data()
housing_n = housing_n.convert_objects(convert_numeric=True)
print(housing_n.shape)
print(housing_n.nunique())
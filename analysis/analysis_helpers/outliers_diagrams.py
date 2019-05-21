import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from database import dbConnection





housing = dbConnection.get_data()
housing = housing.convert_objects(convert_numeric=True)



#Getting rid of outliers
def get_rid_of_outliers(num_data):
    Q1 = num_data.quantile(0.1)
    Q3 = num_data.quantile(0.9)
    IQR = Q3 - Q1
    return num_data[~((num_data < (Q1 - 1.5 * IQR)) |(num_data > (Q3 + 1.5 * IQR))).any(axis=1)]

# sns.boxplot(x=housing['bedrooms'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(housing['bedrooms'], housing['price'])
# ax.set_xlabel('Number of bedrooms')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Bedrooms/Price")


# sns.boxplot(x=housing['bathrooms'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(housing['bathrooms'], housing['price'])
# ax.set_xlabel('Number of bathrooms')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Bathrooms/Price")
#
#
# sns.boxplot(x=housing['floors'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(housing['floors'], housing['price'])
# ax.set_xlabel('Number of floors')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Floors/Price")
#
#
# sns.boxplot(x=housing['view'])#to exclude
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(housing['view'], housing['price'])
# ax.set_xlabel('Number of people seen it')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("View/Price")
#
#
# sns.boxplot(x=housing['condition'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(housing['condition'], housing['price'])
# ax.set_xlabel('Condition of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Condition/Price")
#
# sns.boxplot(x=housing['grade'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(housing['grade'], housing['price'])
# ax.set_xlabel('Grade of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Grade/Price")
#
#
# sns.boxplot(x=housing['yr_built'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(housing['yr_built'], housing['price'])
# ax.set_xlabel('Year built of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Year built/Price")
#
#
# sns.boxplot(x=housing['yr_renovated'])#to exclude
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(housing['yr_renovated'], housing['price'])
# ax.set_xlabel('Year of renovation of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Year renovated/Price")
#
# sns.boxplot(x=housing['zipcode'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(housing['zipcode'], housing['price'])
# ax.set_xlabel('Zipcode of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Zipcode/Price")
#
# sns.boxplot(x=housing['lat'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(housing['lat'], housing['price'])
# ax.set_xlabel('Latitude the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Latitude/Price")
#
# sns.boxplot(x=housing['long'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(housing['long'], housing['price'])
# ax.set_xlabel('Long of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Long/Price")
#
# sns.boxplot(x=housing['sqm_living'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(housing['sqm_living'], housing['price'])
# ax.set_xlabel('Square meters of living of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Square meters living/Price")
#
# sns.boxplot(x=housing['sqft_lot'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(housing['sqft_lot'], housing['price'])
# ax.set_xlabel('Square feets of lot of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Square feets lot/Price")
#
# sns.boxplot(x=housing['sqm_above'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(housing['sqm_above'], housing['price'])
# ax.set_xlabel('Square meters of above of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Square meters above/Price")
#
# sns.boxplot(x=housing['sqm_basement'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(housing['sqm_basement'], housing['price'])
# ax.set_xlabel('Square meters of basement of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Square meters basement/Price")
#
data_filtered = get_rid_of_outliers(housing)

# sns.boxplot(x=data_filtered['bedrooms'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(data_filtered['bedrooms'], data_filtered['price'])
# ax.set_xlabel('Number of bedrooms')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Bedrooms/Price")


# sns.boxplot(x=data_filtered['bathrooms'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(data_filtered['bathrooms'], data_filtered['price'])
# ax.set_xlabel('Number of bathrooms')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Bathrooms/Price")
#
#
# sns.boxplot(x=data_filtered['floors'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(data_filtered['floors'], data_filtered['price'])
# ax.set_xlabel('Number of floors')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Floors/Price")
#
#
# sns.boxplot(x=data_filtered['view'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(data_filtered['view'], data_filtered['price'])
# ax.set_xlabel('Number of people seen it')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("View/Price")
#
#
# sns.boxplot(x=data_filtered['condition'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(data_filtered['condition'], data_filtered['price'])
# ax.set_xlabel('Condition of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Condition/Price")
#
# sns.boxplot(x=data_filtered['grade'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(data_filtered['grade'], data_filtered['price'])
# ax.set_xlabel('Grade of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Grade/Price")
#
#
# sns.boxplot(x=data_filtered['yr_built'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(data_filtered['yr_built'], data_filtered['price'])
# ax.set_xlabel('Year built of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Year built/Price")
#
#
# sns.boxplot(x=data_filtered['yr_renovated'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(data_filtered['yr_renovated'], data_filtered['price'])
# ax.set_xlabel('Year of renovation of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Year renovated/Price")
#
# sns.boxplot(x=data_filtered['zipcode'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(data_filtered['zipcode'], data_filtered['price'])
# ax.set_xlabel('Zipcode of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Zipcode/Price")
#
# sns.boxplot(x=data_filtered['lat'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(data_filtered['lat'], data_filtered['price'])
# ax.set_xlabel('Latitude the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Latitude/Price")
#
# sns.boxplot(x=data_filtered['long'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(data_filtered['long'], data_filtered['price'])
# ax.set_xlabel('Long of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Long/Price")
#
# sns.boxplot(x=data_filtered['sqm_living'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(data_filtered['sqm_living'], data_filtered['price'])
# ax.set_xlabel('Square meters of living of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Square meters living/Price")
#
sns.boxplot(x=data_filtered['sqft_lot'])
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(data_filtered['sqft_lot'], data_filtered['price'])
ax.set_xlabel('Square feets of lot of the house')
ax.set_ylabel('House price')
plt.show()
plt.title("Square feets lot/Price")
#
# sns.boxplot(x=data_filtered['sqm_above'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(data_filtered['sqm_above'], data_filtered['price'])
# ax.set_xlabel('Square meters of above of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Square meters above/Price")
#
# sns.boxplot(x=data_filtered['sqm_basement'])
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(data_filtered['sqm_basement'], data_filtered['price'])
# ax.set_xlabel('Square meters of basement of the house')
# ax.set_ylabel('House price')
# plt.show()
# plt.title("Square meters basement/Price")
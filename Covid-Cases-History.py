#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# View Countries available in API


# In[5]:


import requests

# Define the API URL
url = "https://covid-193.p.rapidapi.com/countries"

# Set the headers with your RapidAPI key and host
headers = {
    "X-RapidAPI-Key": "09cb84aee1msh037978908aa29e7p1966d6jsn69d6f1f59c6a",
    "X-RapidAPI-Host": "covid-193.p.rapidapi.com"
}

# Send an HTTP GET request to the API
response = requests.get(url, headers=headers)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    
    # Print the data
    print(data)
else:
    print("Failed to retrieve data from the COVID-19 API.")


# In[ ]:


# Display covid cases comparing Malaysia and China 


# In[9]:


import requests
import matplotlib.pyplot as plt

# Define the API URL
url = "https://covid-193.p.rapidapi.com/history"

# Define the countries you want to fetch data for
countries = ["Malaysia", "China"]

# Initialize a dictionary to store data for each country
country_data = {}

# Set the headers with your RapidAPI key and host
headers = {
    "X-RapidAPI-Key": "09cb84aee1msh037978908aa29e7p1966d6jsn69d6f1f59c6a",
    "X-RapidAPI-Host": "covid-193.p.rapidapi.com"
}

# Fetch and process COVID-19 data for each country
for country in countries:
    querystring = {"country": country}
    response = requests.get(url, headers=headers, params=querystring)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Extract the history data for the country
        history_data = data.get("response", [])
        
        # Extract the latest data (most recent date)
        if history_data:
            latest_data = history_data[-1]
            total_cases = latest_data.get("cases", {}).get("total")
            if total_cases is not None:
                country_data[country] = int(total_cases)
    
# Sort the country_data dictionary by total cases in descending order
sorted_country_data = dict(sorted(country_data.items(), key=lambda item: item[1], reverse=True))

# Create a bar chart to show most cases to least cases
plt.figure(figsize=(12, 6))
plt.bar(sorted_country_data.keys(), sorted_country_data.values(), color='skyblue')
plt.xlabel('Country')
plt.ylabel('Total COVID-19 Cases')
plt.title('COVID-19 Cases by Country (Most Cases to Least Cases)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:


# Display new and total covid cases in Asian Countries 


# In[11]:


import requests
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta

# Define the API URL
url = "https://covid-193.p.rapidapi.com/history"

# Define a list of Asian countries
asian_countries = ["Malaysia", "China", "Japan", "South Korea", "India", "Thailand", "Vietnam", "Singapore", "Indonesia", "Philippines"]

# Initialize dictionaries to store data for each country
total_cases_data = {}
new_cases_data = {}

# Set the headers with your RapidAPI key and host
headers = {
    "X-RapidAPI-Key": "09cb84aee1msh037978908aa29e7p1966d6jsn69d6f1f59c6a",
    "X-RapidAPI-Host": "covid-193.p.rapidapi.com"
}

# Fetch and process COVID-19 data for each Asian country
for country in asian_countries:
    querystring = {"country": country}
    response = requests.get(url, headers=headers, params=querystring)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Extract the history data for the country
        history_data = data.get("response", [])
        
        # Extract the latest data (most recent date)
        if history_data:
            latest_data = history_data[-1]
            total_cases = latest_data.get("cases", {}).get("total")
            new_cases = latest_data.get("cases", {}).get("new")
            
            if total_cases is not None:
                total_cases_data[country] = int(total_cases)
            if new_cases is not None:
                new_cases_data[country] = int(new_cases)

# Sort the dictionaries by values in descending order
sorted_total_cases_data = dict(sorted(total_cases_data.items(), key=lambda item: item[1], reverse=True))
sorted_new_cases_data = dict(sorted(new_cases_data.items(), key=lambda item: item[1], reverse=True))

# Create bar charts to show total cases and new cases for Asian countries
plt.figure(figsize=(12, 6))

# Total Cases Bar Chart
plt.subplot(1, 2, 1)
plt.bar(sorted_total_cases_data.keys(), sorted_total_cases_data.values(), color='skyblue')
plt.xlabel('Country')
plt.ylabel('Total COVID-19 Cases')
plt.title('COVID-19 Total Cases in Asian Countries')
plt.xticks(rotation=45)

# New Cases Bar Chart
plt.subplot(1, 2, 2)
plt.bar(sorted_new_cases_data.keys(), sorted_new_cases_data.values(), color='salmon')
plt.xlabel('Country')
plt.ylabel('New COVID-19 Cases')
plt.title('COVID-19 New Cases in Asian Countries')
plt.xticks(rotation=45)

plt.tight_layout()

# Predictive Modeling for Forecasting Future Cases
plt.figure(figsize=(12, 6))
plt.title('Predictive Modeling for Forecasting Future Cases')

# Prepare data for forecasting
dates = [datetime.strptime(data_entry['day'], '%Y-%m-%d') for data_entry in history_data]
cases = [int(data_entry['cases']['total']) for data_entry in history_data]

# Create features (days since the first data point)
days_since_start = [(date - dates[0]).days for date in dates]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(np.array(days_since_start).reshape(-1, 1), cases, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future cases
days_in_future = 30  # Number of days to forecast
future_dates = [dates[-1] + timedelta(days=i) for i in range(1, days_in_future + 1)]
future_days_since_start = [(date - dates[0]).days for date in future_dates]
future_cases = model.predict(np.array(future_days_since_start).reshape(-1, 1))

# Plot the historical and forecasted data
plt.plot(dates, cases, label='Historical Cases', marker='o')
plt.plot(future_dates, future_cases, label='Forecasted Cases', linestyle='--', marker='x')
plt.xlabel('Date')
plt.ylabel('Total COVID-19 Cases')
plt.legend()
plt.grid()

plt.show()


# In[ ]:


# View Malaysia's Covid cases on a specific date 


# In[20]:


import requests

# Define the API URL
url = "https://covid-193.p.rapidapi.com/history"

# Specify the country and the date you want to check
country = "Malaysia"
date = "2023-09-30"

# Set the headers with your RapidAPI key and host
headers = {
    "X-RapidAPI-Key": "09cb84aee1msh037978908aa29e7p1966d6jsn69d6f1f59c6a",
    "X-RapidAPI-Host": "covid-193.p.rapidapi.com"
}

# Create the query string with the country and date
querystring = {"country": country, "day": date}

# Send an HTTP GET request to the API
response = requests.get(url, headers=headers, params=querystring)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    
    # Extract the history data for the specified date
    history_data = data.get("response", [])
    
    # Check if there is data for the specified date
    if history_data:
        data_for_date = history_data[0]
        total_cases = data_for_date.get("cases", {}).get("total")
        
        if total_cases is not None:
            print(f"COVID-19 Cases in {country} on {date}: {total_cases}")
        else:
            print(f"No data available for {country} on {date}")
    else:
        print(f"No data available for {country} on {date}")
else:
    print("Failed to retrieve data from the COVID-19 API.")


# In[19]:


import requests

# Define the API URL
url = "https://covid-193.p.rapidapi.com/history"

# Specify the country for which you want to find date range
country = "Malaysia"

# Set the headers with your RapidAPI key and host
headers = {
    "X-RapidAPI-Key": "09cb84aee1msh037978908aa29e7p1966d6jsn69d6f1f59c6a",
    "X-RapidAPI-Host": "covid-193.p.rapidapi.com"
}

# Create the query string with the country
querystring = {"country": country}

# Send an HTTP GET request to the API
response = requests.get(url, headers=headers, params=querystring)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    
    # Extract the history data for the specified country
    history_data = data.get("response", [])
    
    # Initialize variables to keep track of max and min dates
    max_date = None
    min_date = None
    
    # Iterate through the data to find max and min dates
    for entry in history_data:
        date = entry.get("day")
        if date:
            if max_date is None or date > max_date:
                max_date = date
            if min_date is None or date < min_date:
                min_date = date
    
    # Print the results
    if max_date and min_date:
        print(f"Max Date: {max_date}")
        print(f"Min Date: {min_date}")
    else:
        print("No date range information available.")
else:
    print("Failed to retrieve data from the COVID-19 API.")


# In[21]:


pip install dash


# In[ ]:


# Check if the data can be extracted, and view format 


# In[6]:


import requests

url = "https://covid-193.p.rapidapi.com/history"

# Specify the country and day for which you want historical data
querystring = {"country": "usa", "day": "2023-06-02"}

headers = {
	"X-RapidAPI-Key": "09cb84aee1msh037978908aa29e7p1966d6jsn69d6f1f59c6a",
    "X-RapidAPI-Host": "covid-193.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

if response.status_code == 200:
    data = response.json()
    history_data = data.get("response", [])

    if history_data:
        # Print some information from the response
        print(f"Date: {history_data[0]['day']}")
        print(f"Total Cases: {history_data[0]['cases']['total']}")
        print(f"Total Deaths: {history_data[0]['deaths']['total']}")
        print(f"Total Tests Conducted: {history_data[0]['tests']['total']}")
    else:
        print("No historical data found for the specified date.")
else:
    print("Error fetching data from the API.")



# In[2]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import requests

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("COVID-19 Historical Data for the USA"),
    dcc.Graph(id='covid-graph'),
    dcc.DatePickerSingle(
        id='date-picker',
        min_date_allowed='2020-01-22',
        max_date_allowed='2023-09-30',
        initial_visible_month='2023-06-02',
        date='2023-06-02'
    )
])

# Callback to update the line chart based on selected date
@app.callback(
    Output('covid-graph', 'figure'),
    [Input('date-picker', 'date')]
)
def update_graph(selected_date):
    # API URL and parameters
    url = "https://covid-193.p.rapidapi.com/history"
    querystring = {"country": "usa", "day": selected_date}

    headers = {
        "X-RapidAPI-Key": "09cb84aee1msh037978908aa29e7p1966d6jsn69d6f1f59c6a",
        "X-RapidAPI-Host": "covid-193.p.rapidapi.com"
    }

    # Fetch historical data from the API
    response = requests.get(url, headers=headers, params=querystring)

    if response.status_code == 200:
        data = response.json()
        history_data = data.get("response", [])

        if history_data:
            # Extract data for the selected date
            date = history_data[0]['day']
            total_cases = int(history_data[0]['cases']['total'])
            total_deaths = int(history_data[0]['deaths']['total'])
            total_tests = int(history_data[0]['tests']['total'])

            # Create a line chart
            figure = {
                'data': [
                    {'x': [date], 'y': [total_cases], 'type': 'line', 'name': 'Total Cases'},
                    {'x': [date], 'y': [total_deaths], 'type': 'line', 'name': 'Total Deaths'},
                    {'x': [date], 'y': [total_tests], 'type': 'line', 'name': 'Total Tests Conducted'},
                ],
                'layout': {
                    'title': f'COVID-19 Data for the USA on {date}',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Count'},
                }
            }
            return figure

    return {'data': [], 'layout': {}}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


# In[4]:


import requests
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import Input, Output
import dash_table
from datetime import datetime
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("COVID-19 Historical Data Search"),

    # Country search input
    dcc.Input(id='country-search', type='text', value='', placeholder='Enter a country name'),

    # Search button
    html.Button('Search', id='search-button', n_clicks=0),

    # Result section
    html.Div(id='search-result'),

    # Data table
    html.Div([
        html.H2("COVID-19 Historical Data"),
        dash_table.DataTable(
            id='covid-data-table',
            columns=[
                {"name": "Date", "id": "date"},
                {"name": "Total Cases", "id": "total_cases"},
                {"name": "Total Deaths", "id": "total_deaths"},
                {"name": "Total Tests Conducted", "id": "total_tests"},
            ],
            data=[]  # Initial empty data
        )
    ])
])

# Callback to search for historical data
@app.callback(
    [Output('search-result', 'children'),
     Output('covid-data-table', 'data')],
    [Input('search-button', 'n_clicks')],
    [dash.dependencies.State('country-search', 'value')]
)
def search_historical_data(n_clicks, country_search):
    if n_clicks > 0:
        # API URL and parameters for the search
        url = "https://covid-193.p.rapidapi.com/history"
        querystring = {"country": country_search}

        headers = {
            "X-RapidAPI-Key": "09cb84aee1msh037978908aa29e7p1966d6jsn69d6f1f59c6a",
            "X-RapidAPI-Host": "covid-193.p.rapidapi.com"
        }

        # Make the API request
        response = requests.get(url, headers=headers, params=querystring)

        if response.status_code == 200:
            data = response.json()
            history_data = data.get("response", [])

            if history_data:
                # Extract data for the latest date
                latest_data = history_data[0]
                result_text = f"Latest Data for {country_search}:\n"
                result_text += f"Date: {latest_data['day']}\n"
                result_text += f"Total Cases: {latest_data['cases']['total']}\n"
                result_text += f"Total Deaths: {latest_data['deaths']['total']}\n"
                result_text += f"Total Tests Conducted: {latest_data['tests']['total']}\n"

                # Create data for the data table
                data_table = [
                    {
                        'date': data_entry['day'],
                        'total_cases': data_entry['cases']['total'],
                        'total_deaths': data_entry['deaths']['total'],
                        'total_tests': data_entry['tests']['total']
                    }
                    for data_entry in history_data
                ]

                return result_text, data_table
            else:
                return f"No historical data found for {country_search}.", []
        else:
            return f"Error fetching data from the API.", []

    return "", []

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


# In[6]:





# In[ ]:





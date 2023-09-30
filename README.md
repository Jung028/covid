# COVID-19 Data Analysis and Predictive Modeling

<img src="https://github.com/Jung028/covid/blob/main/Search-Data.png?raw=true" alt="Graph0" width="900"/>

## About

The COVID-19 Data Analysis and Predictive Modeling project aims to provide insights into the historical progression of COVID-19 cases, deaths, and testing, as well as forecasting future trends using various modeling techniques. You can choose from time series forecasting, machine learning, or deep learning methods.

## Features

- Historical COVID-19 data analysis.
- Predictive modeling for forecasting future cases.
- Interactive dashboard for visualization.
- Customizable for specific countries or regions.

## Getting Started

To get started with this project, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the application using `python app.py`.

## Usage

- Enter the name of a country or region in the search input.
- Click the "Search" button to view historical data and predictions.
- Explore the interactive dashboard.


## Explaination 
The




## Code Snippets

### Data Retrieval

```python
import requests
import matplotlib.pyplot as plt

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
plt.show()

```
<img src="https://github.com/Jung028/covid/blob/main/covid-asian.png?raw=true" alt="Graph1" width="900"/>





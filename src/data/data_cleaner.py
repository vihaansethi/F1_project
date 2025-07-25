import pandas as pd
import numpy as np

from pathlib import Path


class DataCleaner:

    def __init__(self, data_path):
        """initialize with path to csv files"""
        self.data_path = Path(data_path)
        self.data = {}

    def load_all_data(self):
        """load all csv files"""
        csv_files = [
            'circuits.csv', 'constructors.csv', 'drivers.csv',
            'lap_times.csv', 'pit_stops.csv', 'races.csv',
            'results.csv', 'status.csv', 'driver_standings.csv',
            'constructor_standings.csv', 'qualifying.csv'
        ]

        for csv_file in csv_files:
            file_path = self.data_path / csv_file
            if file_path.exists():
                table_name = csv_file.replace('.csv', '')
                print(f"Loading {csv_file}...")
                self.data[table_name] = pd.read_csv(file_path)
                print(f" Loaded {len(self.data[table_name]):,} rows")
            else:
                print(f" Warning: {csv_file} not found")

    def clean_circuits_data(self):
        """clean circuits data and fix country names"""
        if 'circuits' not in self.data:
            return
        df = self.data['circuits']

        city_to_country = {
            'Catalunya': 'Spain', 'Valencia': 'Spain', 'Barcelona': 'Spain',
            'Indianapolis': 'USA', 'Phoenix': 'USA', 'Detroit': 'USA',
            'Las Vegas': 'USA', 'Miami': 'USA', 'Austin': 'USA',
            'Silverstone': 'UK', 'Brands Hatch': 'UK',
            'Monza': 'Italy', 'Imola': 'Italy',
            'Monte-Carlo': 'Monaco', 'Spa': 'Belgium',
            'Suzuka': 'Japan', 'Shanghai': 'China',
            'Marina Bay': 'Singapore', 'Yas Marina': 'UAE',
            'Sakhir': 'Bahrain', 'Istanbul': 'Turkey',
            'Budapest': 'Hungary', 'Interlagos': 'Brazil',
            'Montreal': 'Canada', 'Melbourne': 'Australia',
            'Baku': 'Azerbaijan', 'Sochi': 'Russia'
        }

        #fix countries
        for city, country in city_to_country.items():
            df.loc[df['location'].str.contains(city, na=False), 'country'] = country
            df.loc[df['country'] == city, 'country'] = country

        #standardize names
        df['country'] = df['country'].replace({
            'UK': 'United Kingdom', 'USA': 'United States',
            'UAE': 'United Arab Emirates', 'Korea': 'South Korea'
        })

        self.data['circuits'] = df
        print(f"Cleaned circuits: {df['country'].nunique()} unique countries")

    def clean_lap_times_data(self):
        """remove outlier lap times"""
        if 'lap_times' not in self.data:
            return
        df = self.data['lap_times']
        original = len(df)
        #keep only reasonble lap times between 1 and 3 minutes
        mask = (df['milliseconds'] >= 60000) & (df['milliseconds'] <= 180000)
        self.data['lap_times'] = df[mask].copy()
        print(f"Cleaned lap times: removed {original - len(self.data['lap_times']):,} outliers")

    def clean_pit_stops_data(self):
        """remove unrealistci pit stops"""
        if 'pit_stops' not in self.data:
            return
        df = self.data['pit_stops']
        original = len(df)

        #keep reasonable pit stop times
        mask = (df['milliseconds'] >= 1500) & (df['milliseconds'] <= 60000)
        self.data['pit_stops'] = df[mask].copy()
        print(f"Cleaned pit stops: removed {original - len(self.data['pit_stops']):,} outliers")

    def clean_results_data(self):
        """quite self explanatory"""
        if 'results' not in self.data:
            return
        df = self.data['results'].copy()
        numeric_cols = ['position', 'milliseconds', 'points', 'grid', 'laps', 'fastestLap', 'rank']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        self.data['results'] = df
        print("Cleaned results data")

    def clean_all_data(self):
        """run all the cleaning operations"""
        print("\nCleaning data...")
        self.clean_circuits_data()
        self.clean_lap_times_data()
        self.clean_pit_stops_data()
        self.clean_results_data
        print("Data cleaning complete!\n")
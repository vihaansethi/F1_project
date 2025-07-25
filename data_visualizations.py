import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

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

    def plot_circuits_by_country(self):
        """bar plot of circuits per country"""
        if 'circuits' not in self.data:
            return
        country_counts = (self.data['circuits']
                          .groupby('country')
                          .size()
                          .reset_index(name='circuit_count')
                          .sort_values('circuit_count', ascending=False)
                          .head(15))
        plt.figure(figsize=(10, 8))
        colors = sns.color_palette("viridis", n_colors=len(country_counts))
        ax = sns.barplot(data=country_counts, x='circuit_count', y='country', palette=colors)

        # add value labels
        for i, (_, row) in enumerate(country_counts.iterrows()):
            ax.text(row['circuit_count'] + 0.1, i, str(row['circuit_count']), va='center')
        plt.title('Top 15 Countries by Number of F1 Circuits', fontsize=16, pad=20)
        plt.xlabel('Number of Circuits')
        plt.ylabel('Country')
        plt.tight_layout()
        plt.show()

    def plot_constructor_wins(self):
        """Clean heatmap showing the wins by consturctyor and year"""
        if not all(k in self.data for k in ['results', 'races', 'constructors']):
            return
        results = self.data['results'].copy()
        wins = results[(results['position'] == 1) | (results['position'] == '1')].copy()
        wins = wins.merge(self.data['races'][['raceId', 'year']], on='raceId')
        wins = wins.merge(self.data['constructors'][['constructorId', 'name']], on='constructorId')
        wins = wins.dropna(subset=['year', 'name'])
        wins['year'] = pd.to_numeric(wins['year'], errors='coerce')
        wins = wins[wins['year'] >= 2010]

        wins_by_year = wins.groupby(['name', 'year']).size().reset_index(name='wins')

        top_constructors = (wins_by_year.groupby('name')['wins']
                            .sum()
                            .nlargest(8)
                            .index.tolist())
        plot_data = wins_by_year[wins_by_year['name'].isin(top_constructors)]
        pivot_data = plot_data.pivot(index='name', columns='year', values='wins').fillna(0)
        pivot_data['total'] = pivot_data.sum(axis=1)
        pivot_data = pivot_data.sort_values('total', ascending=False).drop('total', axis=1)

        plt.figure(figsize=(12, 6))

        sns.heatmap(pivot_data,
                    annot=True,
                    fmt='.0f',
                    cmap='RdYlBu_r',
                    cbar_kws={'label': 'Number of Wins'},
                    linewidths=0.5,
                    linecolor='gray',
                    annot_kws={'size': 10})
        plt.title('Constructor Race Wins Heatmap (2010-Present)',
                  fontsize=18, pad=20, weight='bold')
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Constructor', fontsize=14)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()



    def plot_driver_podiums(self):
        """Horizontal bar plot of driver podium finishes"""
        if not all(k in self.data for k in ['results', 'drivers']):
            return

        # Check position values first
        print("Checking position data...")
        results = self.data['results'].copy()

        # Convert position to numeric first
        results['position'] = pd.to_numeric(results['position'], errors='coerce')

        # Get podium finishes (now checking numeric values)
        podiums = results[results['position'].isin([1, 2, 3])].copy()

        print(f"Found {len(podiums)} podium finishes")

        if len(podiums) == 0:
            print("No podium data found!")
            return

        # Merge with drivers
        podiums = podiums.merge(self.data['drivers'][['driverId', 'forename', 'surname']], on='driverId')
        podiums['driver'] = podiums['forename'] + ' ' + podiums['surname']

        # Calculate podium statistics
        driver_stats = []
        for driver in podiums['driver'].unique():
            driver_data = podiums[podiums['driver'] == driver]

            # Now these comparisons will work because position is numeric
            wins = len(driver_data[driver_data['position'] == 1])
            seconds = len(driver_data[driver_data['position'] == 2])
            thirds = len(driver_data[driver_data['position'] == 3])
            total = wins + seconds + thirds

            if total > 0:  # Only add if driver has podiums
                driver_stats.append({
                    'driver': driver,
                    'wins': wins,
                    'seconds': seconds,
                    'thirds': thirds,
                    'total_podiums': total
                })

        if len(driver_stats) == 0:
            print("No driver statistics calculated!")
            return

        df_stats = pd.DataFrame(driver_stats).sort_values('total_podiums', ascending=False).head(20)

        # Prepare data for stacked bar chart
        df_melted = df_stats[['driver', 'wins', 'seconds', 'thirds']].melt(
            id_vars='driver', var_name='position', value_name='count'
        )

        plt.figure(figsize=(10, 12))

        # Create pivot for stacked bar
        pivot_df = df_melted.pivot(index='driver', columns='position', values='count')
        pivot_df = pivot_df.loc[df_stats['driver']]  # Maintain order

        pivot_df.plot(kind='barh', stacked=True,
                    color=['gold', 'silver', '#CD7F32'],
                    figsize=(10, 12))

        plt.title('Top 20 F1 Drivers by Podium Finishes', fontsize=16, pad=20)
        plt.xlabel('Number of Podiums')
        plt.ylabel('')
        plt.legend(['1st Place', '2nd Place', '3rd Place'],
                loc='lower right', frameon=True)
        plt.tight_layout()
        plt.show()

    def plot_lap_time_distribution(self):
        """Violin plot of lap times by circuit type"""
        if not all(k in self.data for k in ['lap_times', 'races', 'circuits']):
            return

        # Merge lap times with races and circuits
        lap_data = self.data['lap_times'].merge(self.data['races'][['raceId', 'circuitId', 'year']], on='raceId')
        lap_data = lap_data.merge(self.data['circuits'][['circuitId', 'name', 'country']], on='circuitId')

        # Filter recent years
        lap_data = lap_data[lap_data['year'] >= 2018]

        # Calculate average lap time per circuit
        circuit_avg = (lap_data.groupby(['circuitId', 'name'])['milliseconds']
                      .agg(['mean', 'count'])
                      .reset_index())
        circuit_avg = circuit_avg[circuit_avg['count'] >= 100]  # Sufficient data
        circuit_avg['avg_seconds'] = circuit_avg['mean'] / 1000

        # Categorize circuits
        circuit_avg['circuit_type'] = pd.cut(
            circuit_avg['avg_seconds'],
            bins=[0, 75, 85, 95, 200],
            labels=['Short (<75s)', 'Medium (75-85s)', 'Long (85-95s)', 'Very Long (>95s)']
        )

        # Merge back with lap data
        lap_data = lap_data.merge(circuit_avg[['circuitId', 'circuit_type']], on='circuitId')
        lap_data['lap_seconds'] = lap_data['milliseconds'] / 1000

        plt.figure(figsize=(12, 8))

        sns.violinplot(data=lap_data, x='circuit_type', y='lap_seconds',
                       palette='Set2', inner='box')

        plt.title('Lap Time Distribution by Circuit Type (2018-Present)', fontsize=16, pad=20)
        plt.xlabel('Circuit Type')
        plt.ylabel('Lap Time (seconds)')
        plt.ylim(60, 120)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_driver_podiums(self):
        """Horizontal bar plot of driver podium finishes"""
        if not all(k in self.data for k in ['results', 'drivers']):
            return

        # Check position values first
        print("Checking position data...")
        results = self.data['results'].copy()

        # Convert position to numeric first
        results['position'] = pd.to_numeric(results['position'], errors='coerce')

        # Get podium finishes (now checking numeric values)
        podiums = results[results['position'].isin([1, 2, 3])].copy()

        print(f"Found {len(podiums)} podium finishes")

        if len(podiums) == 0:
            print("No podium data found!")
            return

        # Merge with drivers
        podiums = podiums.merge(self.data['drivers'][['driverId', 'forename', 'surname']], on='driverId')
        podiums['driver'] = podiums['forename'] + ' ' + podiums['surname']

        # Calculate podium statistics
        driver_stats = []
        for driver in podiums['driver'].unique():
            driver_data = podiums[podiums['driver'] == driver]

            # Now these comparisons will work because position is numeric
            wins = len(driver_data[driver_data['position'] == 1])
            seconds = len(driver_data[driver_data['position'] == 2])
            thirds = len(driver_data[driver_data['position'] == 3])
            total = wins + seconds + thirds

            if total > 0:  # Only add if driver has podiums
                driver_stats.append({
                    'driver': driver,
                    'wins': wins,
                    'seconds': seconds,
                    'thirds': thirds,
                    'total_podiums': total
                })

        if len(driver_stats) == 0:
            print("No driver statistics calculated!")
            return

        df_stats = pd.DataFrame(driver_stats).sort_values('total_podiums', ascending=False).head(20)

        # Prepare data for stacked bar chart
        df_melted = df_stats[['driver', 'wins', 'seconds', 'thirds']].melt(
            id_vars='driver', var_name='position', value_name='count'
        )

        plt.figure(figsize=(10, 12))

        # Create pivot for stacked bar
        pivot_df = df_melted.pivot(index='driver', columns='position', values='count')
        pivot_df = pivot_df.loc[df_stats['driver']]  # Maintain order

        pivot_df.plot(kind='barh', stacked=True,
                    color=['gold', 'silver', '#CD7F32'],
                    figsize=(10, 12))

        plt.title('Top 20 F1 Drivers by Podium Finishes', fontsize=16, pad=20)
        plt.xlabel('Number of Podiums')
        plt.ylabel('')
        plt.legend(['1st Place', '2nd Place', '3rd Place'],
                loc='lower right', frameon=True)
        plt.tight_layout()
        plt.show()

    def plot_lap_time_distribution(self):
        """Violin plot of lap times by circuit type"""
        if not all(k in self.data for k in ['lap_times', 'races', 'circuits']):
            return

        # Merge lap times with races and circuits
        lap_data = self.data['lap_times'].merge(self.data['races'][['raceId', 'circuitId', 'year']], on='raceId')
        lap_data = lap_data.merge(self.data['circuits'][['circuitId', 'name', 'country']], on='circuitId')

        # Filter recent years
        lap_data = lap_data[lap_data['year'] >= 2018]

        # Calculate average lap time per circuit
        circuit_avg = (lap_data.groupby(['circuitId', 'name'])['milliseconds']
                      .agg(['mean', 'count'])
                      .reset_index())
        circuit_avg = circuit_avg[circuit_avg['count'] >= 100]  # Sufficient data
        circuit_avg['avg_seconds'] = circuit_avg['mean'] / 1000

        # Categorize circuits
        circuit_avg['circuit_type'] = pd.cut(
            circuit_avg['avg_seconds'],
            bins=[0, 75, 85, 95, 200],
            labels=['Short (<75s)', 'Medium (75-85s)', 'Long (85-95s)', 'Very Long (>95s)']
        )

        # Merge back with lap data
        lap_data = lap_data.merge(circuit_avg[['circuitId', 'circuit_type']], on='circuitId')
        lap_data['lap_seconds'] = lap_data['milliseconds'] / 1000

        plt.figure(figsize=(12, 8))

        sns.violinplot(data=lap_data, x='circuit_type', y='lap_seconds',
                       palette='Set2', inner='box')

        plt.title('Lap Time Distribution by Circuit Type (2018-Present)', fontsize=16, pad=20)
        plt.xlabel('Circuit Type')
        plt.ylabel('Lap Time (seconds)')
        plt.ylim(60, 120)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_pit_stop_trends(self):
        """Line plot showing pit stop duration trends - FIXED RANGES"""
        if not all(k in self.data for k in ['pit_stops', 'races']):
            print("Missing required data")
            return

        # Check data
            print(f"Total pit stops: {len(self.data['pit_stops'])}")

        # Merge pit stops with races
        pit_data = self.data['pit_stops'].merge(self.data['races'][['raceId', 'year']], on='raceId')

        # Ensure year is numeric
        pit_data['year'] = pd.to_numeric(pit_data['year'], errors='coerce')

        # Filter to 2010-2024
        pit_data = pit_data[(pit_data['year'] >= 2010) & (pit_data['year'] <= 2024)]
        pit_data['seconds'] = pit_data['milliseconds'] / 1000

        print(f"Pit stops from 2010-2024: {len(pit_data)}")

        # Calculate statistics by year
        yearly_stats = pit_data.groupby('year')['seconds'].agg([
            'mean', 'median', 'min', 'count',
            ('q1', lambda x: x.quantile(0.25)),
            ('q3', lambda x: x.quantile(0.75))
        ]).reset_index()

        # Filter years with enough data
        yearly_stats = yearly_stats[yearly_stats['count'] >= 30]

        if len(yearly_stats) == 0:
            print("Not enough data for visualization")
            return

        print(f"\nYearly statistics summary:")
        print(f"Average range: {yearly_stats['mean'].min():.1f}s to {yearly_stats['mean'].max():.1f}s")
        print(f"Fastest stops range: {yearly_stats['min'].min():.2f}s to {yearly_stats['min'].max():.2f}s")

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Plot 1: Average and Median with IQR - FIXED Y-AXIS
        ax1.plot(yearly_stats['year'], yearly_stats['mean'],
                marker='o', linewidth=3, markersize=8,
                label='Average', color='#2E86AB')
        ax1.plot(yearly_stats['year'], yearly_stats['median'],
                marker='s', linewidth=3, markersize=8,
                label='Median', color='#A23B72')

        # Add shaded IQR
        ax1.fill_between(yearly_stats['year'],
                        yearly_stats['q1'],
                        yearly_stats['q3'],
                        alpha=0.3, color='gray',
                        label='25th-75th Percentile')

        # Add value labels for average
        for _, row in yearly_stats.iterrows():
            ax1.annotate(f"{row['mean']:.1f}",
                        (row['year'], row['mean']),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha='center',
                        fontsize=8,
                        color='#2E86AB')

        ax1.set_ylabel('Duration (seconds)', fontsize=12)
        ax1.set_title('F1 Pit Stop Duration Trends: Average and Median (2010-2024)',
                    fontsize=16, pad=20, weight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # SET FIXED Y-AXIS RANGE FOR TOP PLOT
        ax1.set_ylim(10, 30)
        ax1.set_yticks(range(10, 31, 2))  # Ticks every 2 seconds from 10 to 30

        # Plot 2: Fastest pit stops - ADJUSTED RANGE
        ax2.plot(yearly_stats['year'], yearly_stats['min'],
                marker='^', linewidth=3, markersize=10,
                color='#F18F01', label='Fastest Stop')

        # Add value labels for ALL fastest stops
        for _, row in yearly_stats.iterrows():
            ax2.annotate(f"{row['min']:.2f}s",
                        (row['year'], row['min']),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha='center',
                        fontsize=9,
                        color='#F18F01',
                        weight='bold')

        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Duration (seconds)', fontsize=12)
        ax2.set_title('Fastest Pit Stop Each Year', fontsize=14, weight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, linestyle='--')

        # SET APPROPRIATE Y-AXIS RANGE FOR BOTTOM PLOT
        # Find min and max of fastest stops and add padding
        ax2.set_ylim(10, 20)
        ax2.set_yticks(range(10, 21, 1))

        # Format x-axis to show years from 2010 to 2024
        years = range(2010, 2025, 1)  # Every year
        ax2.set_xticks(years)
        ax2.set_xticklabels(years, rotation=45, ha='right')

        # Add a horizontal line at key thresholds
        ax1.axhline(y=20, color='red', linestyle=':', alpha=0.5, label='20s threshold')
        ax2.axhline(y=2.0, color='green', linestyle=':', alpha=0.5, label='Sub-2s barrier')

        # Add trend information
        if len(yearly_stats) > 1:
            # Calculate trend
            years_numeric = yearly_stats['year'].values
            avg_times = yearly_stats['mean'].values
            trend = np.polyfit(years_numeric, avg_times, 1)

            # Add trend text
            trend_text = f"Average trend: {trend[0]:.2f}s per year"
            ax1.text(0.02, 0.98, trend_text, transform=ax1.transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

    def plot_constructor_dominance_heatmap(self):
        """Heatmap showing constructor dominance by year"""
        if not all(k in self.data for k in ['results', 'races', 'constructors']):
            return

        # Merge results with races and constructors
        results = self.data['results'].merge(self.data['races'][['raceId', 'year']], on='raceId')
        results = results.merge(self.data['constructors'][['constructorId', 'name']], on='constructorId')

        # Filter hybrid era
        results = results[results['year'] >= 2014]

        # Calculate points by constructor and year
        constructor_points = (results.groupby(['name', 'year'])['points']
                             .sum()
                             .reset_index())

        # Get top constructors
        top_constructors = (constructor_points.groupby('name')['points']
                           .sum()
                           .nlargest(8)
                           .index)

        # Filter and pivot
        plot_data = constructor_points[constructor_points['name'].isin(top_constructors)]
        pivot_data = plot_data.pivot(index='name', columns='year', values='points').fillna(0)
        pivot_data = pivot_data.loc[pivot_data.sum(axis=1).sort_values(ascending=False).index]

        plt.figure(figsize=(14, 8))

        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd',
                    cbar_kws={'label': 'Total Points'},
                    linewidths=0.5, linecolor='gray')

        plt.title('Constructor Championship Points by Year (Hybrid Era)', fontsize=16, pad=20)
        plt.xlabel('Year')
        plt.ylabel('Constructor')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

# Example usage
def main():
    # Initialize analyzer with your data path
    analyzer = DataCleaner('/Users/vihaansethi/Downloads/f1_championship')
    
    # Load and clean data
    analyzer.load_all_data()
    analyzer.clean_all_data()
    
    # Create visualizations
    print("Creating visualizations...")
    analyzer.plot_circuits_by_country()
    analyzer.plot_constructor_wins() # fixed
    analyzer.plot_driver_podiums() # this one is a shit too
    analyzer.plot_lap_time_distribution()
    analyzer.plot_pit_stop_trends() # fix it!!
    analyzer.plot_constructor_dominance_heatmap()

if __name__ == "__main__":
    main()

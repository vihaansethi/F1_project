import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Load your actual F1 circuits data
def load_f1_data():
    """Load your actual F1 circuits data"""
    path = r"/Users/vihaansethi/Downloads/f1_championship/circuits.csv"
    df = pd.read_csv(path)
    
    # Fix negative altitude values for visualization (some circuits are below sea level)
    print(f"Original altitude range: {df['alt'].min()} to {df['alt'].max()}")
    
    # Create positive values for size (plotly requires positive sizes)
    df['alt_for_size'] = df['alt'].apply(lambda x: max(abs(x), 1))  # Use absolute value, minimum 1
    df['alt_normalized'] = df['alt'] - df['alt'].min() + 1  # Shift all values to be positive
    
    print(f"✅ Processed {len(df)} circuits from {df['country'].nunique()} countries")
    print(f"Altitude for size range: {df['alt_for_size'].min()} to {df['alt_for_size'].max()}")
    
    return df

class F1CircuitsDashboard:
    """Complete F1 Circuits Interactive Dashboard - Fixed Version"""
    
    def __init__(self, dataframe):
        self.df = dataframe
        self.filtered_df = dataframe.copy()
        
    def create_interactive_world_map(self):
        """Interactive world map - FIXED VERSION"""
        try:
            # Use scatter_geo instead of scatter_mapbox to avoid mapbox issues
            fig = px.scatter_geo(
                self.filtered_df,
                lat="lat", 
                lon="lng",
                hover_name="name",
                hover_data={
                    "location": True,
                    "country": True,
                    "alt": True,
                    "lat": False,
                    "lng": False
                },
                color="alt",  # Color by actual altitude (can be negative)
                size="alt_for_size",  # Size by positive values only
                color_continuous_scale="RdYlBu_r",  # Red-Yellow-Blue reversed
                size_max=15,
                title="🌍 F1 Circuits Around the World (Interactive Map)",
                labels={"alt": "Altitude (m)"}
            )
            
            fig.update_layout(
                height=600,
                font=dict(size=14),
                geo=dict(
                    showland=True,
                    landcolor="rgb(243, 243, 243)",
                    coastlinecolor="rgb(204, 204, 204)",
                    projection_type="equirectangular"
                )
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating world map: {e}")
            # Fallback to simple scatter plot
            return self.create_simple_geographic_plot()
    
    def create_simple_geographic_plot(self):
        """Simple geographic scatter plot as fallback"""
        fig = px.scatter(
            self.filtered_df,
            x="lng",
            y="lat", 
            hover_name="name",
            color="alt",
            size="alt_for_size",
            title="🗺️ F1 Circuits Geographic Distribution",
            labels={
                "lng": "Longitude",
                "lat": "Latitude", 
                "alt": "Altitude (m)"
            }
        )
        fig.update_layout(height=500)
        return fig
    
    def create_interactive_scatter_plot(self):
        """Interactive scatter plot with filters"""
        fig = px.scatter(
            self.filtered_df,
            x="lat",
            y="alt",
            color="country",
            size="alt_for_size",
            hover_name="name",
            hover_data=["location", "country"],
            title="🏔️ Circuit Altitude vs Latitude",
            labels={
                "lat": "Latitude (°)",
                "alt": "Altitude (meters)",
                "country": "Country"
            },
            size_max=20
        )
        
        # Add interactive elements
        fig.update_layout(
            height=500,
            showlegend=True,
            hovermode='closest'
        )
        
        return fig
    
    def create_interactive_bar_chart(self):
        """Interactive bar chart with sorting"""
        country_stats = self.filtered_df.groupby('country').agg({
            'circuitId': 'count',
            'alt': ['mean', 'max', 'min']
        }).round(2)
        
        country_stats.columns = ['circuit_count', 'avg_altitude', 'max_altitude', 'min_altitude']
        country_stats = country_stats.reset_index().sort_values('circuit_count', ascending=False)
        
        fig = px.bar(
            country_stats.head(15),  # Show top 15 countries
            x='country',
            y='circuit_count',
            color='avg_altitude',
            hover_data=['max_altitude', 'min_altitude'],
            title="📊 Top 15 Countries by Number of F1 Circuits",
            labels={
                'country': 'Country',
                'circuit_count': 'Number of Circuits',
                'avg_altitude': 'Average Altitude (m)'
            },
            color_continuous_scale="Viridis"
        )
        
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_altitude_analysis(self):
        """Detailed altitude analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Altitude Distribution', 
                'Altitude by Country (Top 10)',
                'Positive vs Negative Altitude', 
                'Geographic Spread'
            ],
            specs=[
                [{"type": "histogram"}, {"type": "box"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Altitude histogram
        fig.add_trace(
            go.Histogram(
                x=self.filtered_df['alt'],
                nbinsx=20,
                name="Altitude Distribution",
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # 2. Box plot by country (top 10)
        top_countries = self.filtered_df['country'].value_counts().head(10).index
        filtered_top = self.filtered_df[self.filtered_df['country'].isin(top_countries)]
        
        for country in top_countries:
            country_data = filtered_top[filtered_top['country'] == country]
            fig.add_trace(
                go.Box(
                    y=country_data['alt'],
                    name=country,
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Positive vs Negative altitude counts
        positive_count = (self.filtered_df['alt'] >= 0).sum()
        negative_count = (self.filtered_df['alt'] < 0).sum()
        
        fig.add_trace(
            go.Bar(
                x=['Above Sea Level', 'Below Sea Level'],
                y=[positive_count, negative_count],
                marker_color=['green', 'red'],
                name="Altitude Categories",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Geographic spread
        fig.add_trace(
            go.Scatter(
                x=self.filtered_df['lng'],
                y=self.filtered_df['lat'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=self.filtered_df['alt'],
                    colorscale='RdYlBu_r',
                    showscale=False
                ),
                text=self.filtered_df['name'],
                name="Circuits",
                showlegend=False,
                hovertemplate="<b>%{text}</b><br>Lat: %{y}<br>Lng: %{x}<extra></extra>"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="🏎️ F1 Circuits Altitude Analysis Dashboard",
            showlegend=True
        )
        
        return fig
    
    def create_country_summary(self):
        """Summary statistics by country"""
        summary = self.filtered_df.groupby('country').agg({
            'circuitId': 'count',
            'alt': ['min', 'max', 'mean'],
            'lat': ['min', 'max'],
            'lng': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        summary.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in summary.columns]
        summary = summary.reset_index()
        
        # Rename for clarity
        summary.columns = ['country', 'circuit_count', 'min_alt', 'max_alt', 'avg_alt', 
                          'min_lat', 'max_lat', 'min_lng', 'max_lng']
        
        # Create interactive table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(summary.columns),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[summary[col] for col in summary.columns],
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title="📋 F1 Circuits Summary by Country",
            height=600
        )
        
        return fig
    
    def create_extreme_circuits(self):
        """Highlight extreme circuits (highest, lowest, northernmost, etc.)"""
        extremes = {
            'Highest': self.filtered_df.loc[self.filtered_df['alt'].idxmax()],
            'Lowest': self.filtered_df.loc[self.filtered_df['alt'].idxmin()],
            'Northernmost': self.filtered_df.loc[self.filtered_df['lat'].idxmax()],
            'Southernmost': self.filtered_df.loc[self.filtered_df['lat'].idxmin()],
            'Easternmost': self.filtered_df.loc[self.filtered_df['lng'].idxmax()],
            'Westernmost': self.filtered_df.loc[self.filtered_df['lng'].idxmin()]
        }
        
        extreme_data = []
        for category, circuit in extremes.items():
            extreme_data.append({
                'Category': category,
                'Circuit': circuit['name'],
                'Location': f"{circuit['location']}, {circuit['country']}",
                'Latitude': circuit['lat'],
                'Longitude': circuit['lng'],
                'Altitude': f"{circuit['alt']}m"
            })
        
        extreme_df = pd.DataFrame(extreme_data)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(extreme_df.columns),
                fill_color='gold',
                align='center',
                font=dict(size=14, color='black')
            ),
            cells=dict(
                values=[extreme_df[col] for col in extreme_df.columns],
                fill_color='lightyellow',
                align='center',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(
            title="🏆 Extreme F1 Circuits Records",
            height=400
        )
        
        return fig
    
    def display_all_charts(self):
        """Display all interactive charts"""
        print("🏎️ F1 Circuits Interactive Dashboard")
        print("=" * 50)
        
        charts = [
            ("Interactive World Map", self.create_interactive_world_map()),
            ("Interactive Scatter Plot", self.create_interactive_scatter_plot()),
            ("Interactive Bar Chart", self.create_interactive_bar_chart()),
            ("Altitude Analysis Dashboard", self.create_altitude_analysis()),
            ("Country Summary Table", self.create_country_summary()),
            ("Extreme Circuits Records", self.create_extreme_circuits())
        ]
        
        for title, fig in charts:
            print(f"\n📊 {title}")
            try:
                fig.show()
            except Exception as e:
                print(f"Error displaying {title}: {e}")
                
        return charts

# Main execution
if __name__ == "__main__":
    print("🚀 Loading F1 Circuits Data...")
    
    # Load and process data
    df = load_f1_data()
    
    # Display basic info about your dataset
    print(f"\n📊 Dataset Info:")
    print(f"   • Total circuits: {len(df)}")
    print(f"   • Countries: {df['country'].nunique()}")
    print(f"   • Altitude range: {df['alt'].min()}m to {df['alt'].max()}m")
    print(f"   • Circuits below sea level: {(df['alt'] < 0).sum()}")
    print(f"   • Sample circuit: {df.iloc[0]['name']}")
    
    # Initialize and run dashboard
    dashboard = F1CircuitsDashboard(df)
    dashboard.display_all_charts()
    
    print("\n✅ Dashboard creation complete!")
    print("🎯 All plots are interactive - you can:")
    print("   • Hover for details")
    print("   • Zoom and pan") 
    print("   • Click legend items to hide/show data")
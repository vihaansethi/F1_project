"""
Dynamic Plotly Dashboard Builder
Create interactive visualizations by selecting columns and chart types dynamically.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Available chart types for the dashboard."""
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    PIE = "pie"
    SUNBURST = "sunburst"
    TREEMAP = "treemap"
    SCATTER_3D = "scatter_3d"
    SCATTER_GEO = "scatter_geo"
    CHOROPLETH = "choropleth"
    DENSITY_CONTOUR = "density_contour"
    PARALLEL_COORDINATES = "parallel_coordinates"
    RADAR = "radar"
    FUNNEL = "funnel"
    WATERFALL = "waterfall"


@dataclass
class ChartConfig:
    """Configuration for creating a chart."""
    chart_type: ChartType
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    z_column: Optional[str] = None
    color_column: Optional[str] = None
    size_column: Optional[str] = None
    facet_col: Optional[str] = None
    facet_row: Optional[str] = None
    animation_frame: Optional[str] = None
    hover_data: Optional[List[str]] = None
    title: Optional[str] = None
    custom_params: Optional[Dict[str, Any]] = None


class DataAnalyzer:
    """Analyzes data to suggest appropriate chart types and configurations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.column_types = self._analyze_columns()
    
    def _analyze_columns(self) -> Dict[str, str]:
        """Analyze column types for better chart suggestions."""
        column_types = {}
        
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if self.df[col].nunique() < 10 and self.df[col].min() >= 0:
                    column_types[col] = 'categorical_numeric'
                else:
                    column_types[col] = 'continuous'
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                column_types[col] = 'datetime'
            elif self.df[col].nunique() < len(self.df) * 0.1:
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'text'
        
        return column_types
    
    def get_numeric_columns(self) -> List[str]:
        """Get numeric columns suitable for quantitative analysis."""
        return [col for col, dtype in self.column_types.items() 
                if dtype in ['continuous', 'categorical_numeric']]
    
    def get_categorical_columns(self) -> List[str]:
        """Get categorical columns suitable for grouping."""
        return [col for col, dtype in self.column_types.items() 
                if dtype in ['categorical', 'categorical_numeric']]
    
    def get_datetime_columns(self) -> List[str]:
        """Get datetime columns for time-series analysis."""
        return [col for col, dtype in self.column_types.items() if dtype == 'datetime']
    
    def suggest_charts(self, max_suggestions: int = 5) -> List[ChartConfig]:
        """Suggest appropriate chart configurations based on data."""
        suggestions = []
        numeric_cols = self.get_numeric_columns()
        categorical_cols = self.get_categorical_columns()
        datetime_cols = self.get_datetime_columns()
        
        # Scatter plots for numeric relationships
        if len(numeric_cols) >= 2:
            suggestions.append(ChartConfig(
                chart_type=ChartType.SCATTER,
                x_column=numeric_cols[0],
                y_column=numeric_cols[1],
                color_column=categorical_cols[0] if categorical_cols else None,
                title=f"{numeric_cols[1]} vs {numeric_cols[0]}"
            ))
        
        # Histograms for distributions
        if numeric_cols:
            suggestions.append(ChartConfig(
                chart_type=ChartType.HISTOGRAM,
                x_column=numeric_cols[0],
                color_column=categorical_cols[0] if categorical_cols else None,
                title=f"Distribution of {numeric_cols[0]}"
            ))
        
        # Bar charts for categorical analysis
        if categorical_cols and numeric_cols:
            suggestions.append(ChartConfig(
                chart_type=ChartType.BAR,
                x_column=categorical_cols[0],
                y_column=numeric_cols[0],
                title=f"{numeric_cols[0]} by {categorical_cols[0]}"
            ))
        
        # Box plots for distributions by category
        if categorical_cols and numeric_cols:
            suggestions.append(ChartConfig(
                chart_type=ChartType.BOX,
                x_column=categorical_cols[0],
                y_column=numeric_cols[0],
                title=f"{numeric_cols[0]} Distribution by {categorical_cols[0]}"
            ))
        
        # Time series if datetime columns exist
        if datetime_cols and numeric_cols:
            suggestions.append(ChartConfig(
                chart_type=ChartType.LINE,
                x_column=datetime_cols[0],
                y_column=numeric_cols[0],
                color_column=categorical_cols[0] if categorical_cols else None,
                title=f"{numeric_cols[0]} over {datetime_cols[0]}"
            ))
        
        return suggestions[:max_suggestions]


class DynamicDashboard:
    """Dynamic dashboard builder with configurable charts."""
    
    # Chart styling configuration
    CHART_CONFIG = {
        'template': 'plotly_white',
        'font': dict(family='Inter, sans-serif', size=12, color='#2D3748'),
        'title_font_size': 16,
        'colorscale': 'viridis',
        'height': 500,
    }
    
    def __init__(self, df: pd.DataFrame, title: str = "Dynamic Dashboard"):
        self.df = df
        self.title = title
        self.analyzer = DataAnalyzer(df)
        self.charts = []
        logger.info(f"Dashboard initialized with {len(df)} rows and {len(df.columns)} columns")
    
    def add_chart(self, config: ChartConfig) -> None:
        """Add a chart configuration to the dashboard."""
        self.charts.append(config)
        logger.info(f"Added {config.chart_type.value} chart")
    
    def create_chart(self, config: ChartConfig) -> go.Figure:
        """Create a single chart based on configuration."""
        try:
            chart_method = getattr(self, f'_create_{config.chart_type.value}')
            fig = chart_method(config)
            
            # Apply common styling
            fig.update_layout(
                template=self.CHART_CONFIG['template'],
                font=self.CHART_CONFIG['font'],
                title_font_size=self.CHART_CONFIG['title_font_size'],
                height=config.custom_params.get('height', self.CHART_CONFIG['height']) if config.custom_params else self.CHART_CONFIG['height']
            )
            
            return fig
        except AttributeError:
            logger.error(f"Chart type {config.chart_type.value} not implemented")
            return self._create_placeholder_chart(f"Chart type '{config.chart_type.value}' not implemented")
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            return self._create_placeholder_chart(f"Error: {str(e)}")
    
    def _create_scatter(self, config: ChartConfig) -> go.Figure:
        """Create scatter plot."""
        return px.scatter(
            self.df,
            x=config.x_column,
            y=config.y_column,
            color=config.color_column,
            size=config.size_column,
            facet_col=config.facet_col,
            facet_row=config.facet_row,
            animation_frame=config.animation_frame,
            hover_data=config.hover_data,
            title=config.title or f"{config.y_column} vs {config.x_column}"
        )
    
    def _create_line(self, config: ChartConfig) -> go.Figure:
        """Create line plot."""
        return px.line(
            self.df,
            x=config.x_column,
            y=config.y_column,
            color=config.color_column,
            facet_col=config.facet_col,
            facet_row=config.facet_row,
            animation_frame=config.animation_frame,
            hover_data=config.hover_data,
            title=config.title or f"{config.y_column} over {config.x_column}"
        )
    
    def _create_bar(self, config: ChartConfig) -> go.Figure:
        """Create bar chart."""
        if config.y_column:
            # Grouped bar chart
            return px.bar(
                self.df,
                x=config.x_column,
                y=config.y_column,
                color=config.color_column,
                facet_col=config.facet_col,
                facet_row=config.facet_row,
                animation_frame=config.animation_frame,
                hover_data=config.hover_data,
                title=config.title or f"{config.y_column} by {config.x_column}"
            )
        else:
            # Count bar chart
            counts = self.df[config.x_column].value_counts()
            return px.bar(
                x=counts.index,
                y=counts.values,
                title=config.title or f"Count of {config.x_column}"
            )
    
    def _create_histogram(self, config: ChartConfig) -> go.Figure:
        """Create histogram."""
        return px.histogram(
            self.df,
            x=config.x_column,
            color=config.color_column,
            facet_col=config.facet_col,
            facet_row=config.facet_row,
            animation_frame=config.animation_frame,
            hover_data=config.hover_data,
            title=config.title or f"Distribution of {config.x_column}"
        )
    
    def _create_box(self, config: ChartConfig) -> go.Figure:
        """Create box plot."""
        return px.box(
            self.df,
            x=config.x_column,
            y=config.y_column,
            color=config.color_column,
            facet_col=config.facet_col,
            facet_row=config.facet_row,
            animation_frame=config.animation_frame,
            hover_data=config.hover_data,
            title=config.title or f"{config.y_column} Distribution by {config.x_column}"
        )
    
    def _create_violin(self, config: ChartConfig) -> go.Figure:
        """Create violin plot."""
        return px.violin(
            self.df,
            x=config.x_column,
            y=config.y_column,
            color=config.color_column,
            facet_col=config.facet_col,
            facet_row=config.facet_row,
            hover_data=config.hover_data,
            title=config.title or f"{config.y_column} Distribution by {config.x_column}"
        )
    
    def _create_heatmap(self, config: ChartConfig) -> go.Figure:
        """Create heatmap."""
        # For correlation heatmap
        if not config.x_column and not config.y_column:
            numeric_cols = self.analyzer.get_numeric_columns()
            if len(numeric_cols) > 1:
                corr_matrix = self.df[numeric_cols].corr()
                return px.imshow(
                    corr_matrix,
                    title=config.title or "Correlation Heatmap",
                    color_continuous_scale='RdBu'
                )
        
        # Pivot table heatmap
        if config.x_column and config.y_column and config.color_column:
            pivot_table = self.df.pivot_table(
                values=config.color_column,
                index=config.y_column,
                columns=config.x_column,
                aggfunc='mean'
            )
            return px.imshow(
                pivot_table,
                title=config.title or f"{config.color_column} by {config.x_column} and {config.y_column}"
            )
        
        return self._create_placeholder_chart("Heatmap requires appropriate columns")
    
    def _create_pie(self, config: ChartConfig) -> go.Figure:
        """Create pie chart."""
        if config.y_column:
            # Values provided
            return px.pie(
                self.df,
                names=config.x_column,
                values=config.y_column,
                color=config.color_column,
                title=config.title or f"{config.y_column} by {config.x_column}"
            )
        else:
            # Count pie chart
            counts = self.df[config.x_column].value_counts()
            return px.pie(
                values=counts.values,
                names=counts.index,
                title=config.title or f"Distribution of {config.x_column}"
            )
    
    def _create_sunburst(self, config: ChartConfig) -> go.Figure:
        """Create sunburst chart."""
        path_cols = [col for col in [config.x_column, config.y_column, config.z_column] if col]
        if len(path_cols) < 2:
            return self._create_placeholder_chart("Sunburst requires at least 2 categorical columns")
        
        return px.sunburst(
            self.df,
            path=path_cols,
            values=config.size_column,
            color=config.color_column,
            title=config.title or f"Hierarchy: {' > '.join(path_cols)}"
        )
    
    def _create_treemap(self, config: ChartConfig) -> go.Figure:
        """Create treemap."""
        path_cols = [col for col in [config.x_column, config.y_column, config.z_column] if col]
        if not path_cols:
            return self._create_placeholder_chart("Treemap requires categorical columns")
        
        return px.treemap(
            self.df,
            path=path_cols,
            values=config.size_column,
            color=config.color_column,
            title=config.title or f"Treemap: {' > '.join(path_cols)}"
        )
    
    def _create_scatter_3d(self, config: ChartConfig) -> go.Figure:
        """Create 3D scatter plot."""
        if not all([config.x_column, config.y_column, config.z_column]):
            return self._create_placeholder_chart("3D scatter requires X, Y, and Z columns")
        
        return px.scatter_3d(
            self.df,
            x=config.x_column,
            y=config.y_column,
            z=config.z_column,
            color=config.color_column,
            size=config.size_column,
            hover_data=config.hover_data,
            title=config.title or f"3D: {config.x_column}, {config.y_column}, {config.z_column}"
        )
    
    def _create_scatter_geo(self, config: ChartConfig) -> go.Figure:
        """Create geographic scatter plot."""
        # Assume x_column is latitude, y_column is longitude
        lat_col = config.x_column or 'lat'
        lon_col = config.y_column or 'lon'
        
        if lat_col not in self.df.columns or lon_col not in self.df.columns:
            return self._create_placeholder_chart(f"Geographic plot requires '{lat_col}' and '{lon_col}' columns")
        
        return px.scatter_geo(
            self.df,
            lat=lat_col,
            lon=lon_col,
            color=config.color_column,
            size=config.size_column,
            hover_data=config.hover_data,
            title=config.title or "Geographic Distribution"
        )
    
    def _create_density_contour(self, config: ChartConfig) -> go.Figure:
        """Create density contour plot."""
        return px.density_contour(
            self.df,
            x=config.x_column,
            y=config.y_column,
            color=config.color_column,
            facet_col=config.facet_col,
            title=config.title or f"Density: {config.y_column} vs {config.x_column}"
        )
    
    def _create_parallel_coordinates(self, config: ChartConfig) -> go.Figure:
        """Create parallel coordinates plot."""
        numeric_cols = self.analyzer.get_numeric_columns()
        if len(numeric_cols) < 3:
            return self._create_placeholder_chart("Parallel coordinates requires at least 3 numeric columns")
        
        return px.parallel_coordinates(
            self.df,
            dimensions=numeric_cols[:6],  # Limit to 6 dimensions
            color=config.color_column,
            title=config.title or "Parallel Coordinates"
        )
    
    def _create_placeholder_chart(self, message: str) -> go.Figure:
        """Create a placeholder chart with error message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="Chart Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    def create_dashboard(self, cols: int = 2) -> go.Figure:
        """Create a multi-chart dashboard."""
        if not self.charts:
            logger.warning("No charts configured")
            return self._create_placeholder_chart("No charts configured")
        
        rows = (len(self.charts) + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[config.title or f"Chart {i+1}" for i, config in enumerate(self.charts)],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        for i, config in enumerate(self.charts):
            row = i // cols + 1
            col = i % cols + 1
            
            try:
                chart = self.create_chart(config)
                for trace in chart.data:
                    fig.add_trace(trace, row=row, col=col)
            except Exception as e:
                logger.error(f"Error adding chart {i}: {e}")
        
        fig.update_layout(
            template=self.CHART_CONFIG['template'],
            title_text=self.title,
            title_font_size=20,
            height=rows * 400,
            showlegend=False
        )
        
        return fig
    
    def show_dashboard(self, cols: int = 2) -> None:
        """Display the complete dashboard."""
        dashboard_fig = self.create_dashboard(cols)
        dashboard_fig.show()
    
    def show_individual_charts(self) -> None:
        """Display each chart individually."""
        for i, config in enumerate(self.charts):
            logger.info(f"Displaying chart {i+1}: {config.chart_type.value}")
            fig = self.create_chart(config)
            fig.show()
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary information about the dataset."""
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'numeric_columns': self.analyzer.get_numeric_columns(),
            'categorical_columns': self.analyzer.get_categorical_columns(),
            'datetime_columns': self.analyzer.get_datetime_columns(),
            'column_types': self.analyzer.column_types,
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from various file formats."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        if path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif path.suffix.lower() == '.json':
            return pd.read_json(file_path)
        elif path.suffix.lower() == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def demo_dashboard():
    """Demo function showing how to use the dynamic dashboard."""
    # Load sample data (replace with your data path)
    DATA_PATH = r"/Users/vihaansethi/Downloads/f1_championship/circuits.csv"
    
    try:
        # Load data
        df = load_data(DATA_PATH)
        
        # Create dashboard
        dashboard = DynamicDashboard(df, "F1 Circuits Dynamic Dashboard")
        
        # Print data summary
        summary = dashboard.get_data_summary()
        print("📊 Data Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Get suggested charts
        suggestions = dashboard.analyzer.suggest_charts()
        print(f"\n💡 Suggested Charts ({len(suggestions)}):")
        for i, config in enumerate(suggestions):
            print(f"  {i+1}. {config.chart_type.value}: {config.title}")
        
        # Add suggested charts
        for config in suggestions:
            dashboard.add_chart(config)
        
        # Add custom charts
        dashboard.add_chart(ChartConfig(
            chart_type=ChartType.HEATMAP,
            title="Correlation Matrix",
        ))
        
        dashboard.add_chart(ChartConfig(
            chart_type=ChartType.SCATTER_GEO,
            x_column='lat',
            y_column='lng',
            color_column='alt',
            size_column='alt',
            title="Geographic Distribution"
        ))
        
        # Show dashboard
        print(f"\n🚀 Displaying dashboard with {len(dashboard.charts)} charts...")
        dashboard.show_dashboard(cols=2)
        
        # Optionally show individual charts
        # dashboard.show_individual_charts()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    demo_dashboard()
# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import datetime
from datetime import datetime as dt


print("🚀 Starting LFB Data Processing Pipeline...")
print("=" * 60)

# Define the column data types for consistent data loading
dtype_spec = {
    'IncidentNumber': 'string',
    'DateOfCall': 'object',
    'CalYear': 'Int16',
    'TimeOfCall': 'object',
    'HourOfCall': 'UInt8',
    'IncidentGroup': 'category',
    'StopCodeDescription': 'category',
    'SpecialServiceType': 'category',
    'PropertyCategory': 'category',
    'PropertyType': 'category',
    'AddressQualifier': 'category',
    'Postcode_full': 'string',
    'Postcode_district': 'string',
    'UPRN': 'Float64',
    'USRN': 'Int64',
    'IncGeo_BoroughCode': 'category',
    'IncGeo_BoroughName': 'category',
    'ProperCase': 'category',
    'IncGeo_WardCode': 'category',
    'IncGeo_WardName': 'category',
    'IncGeo_WardNameNew': 'category',
    'Easting_m': 'float64',
    'Northing_m': 'float64',
    'Easting_rounded': 'Int32',
    'Northing_rounded': 'Int32',
    'Latitude': 'float64',
    'Longitude': 'float64',
    'FRS': 'category',
    'IncidentStationGround': 'category',
    'FirstPumpArriving_AttendanceTime': 'Int16',
    'FirstPumpArriving_DeployedFromStation': 'category',
    'SecondPumpArriving_AttendanceTime': 'Int16',
    'SecondPumpArriving_DeployedFromStation': 'category',
    'NumStationsWithPumpsAttending': 'UInt8',
    'NumPumpsAttending': 'UInt8',
    'PumpCount': 'UInt16',
    'PumpMinutesRounded': 'UInt32',
    'Notional Cost (£)': 'UInt32',
    'NumCalls': 'UInt16'
}

print("📁 STEP 1: Loading Incident Data Files...")
print("-" * 40)

# Read CSV file (2009-2017 data)
print("📊 Loading CSV file: LFB Incident data from 2009 - 2017.csv")
csv_file_path = "LFB Incident data from 2009 - 2017.csv"
df_csv = pd.read_csv(
    csv_file_path,
    dtype=dtype_spec
)
print(f"   ✅ CSV loaded: {len(df_csv):,} records, {len(df_csv.columns)} columns")

# Process date and time columns for CSV data
print("   ⏰ Processing date/time columns for CSV data...")
df_csv['DateOfCall'] = pd.to_datetime(df_csv['DateOfCall'], format='mixed').dt.date
df_csv['DateOfCall'] = pd.to_datetime(df_csv['DateOfCall'])
df_csv['TimeOfCall'] = pd.to_timedelta(df_csv['TimeOfCall'].astype(str))
print("   ✅ Date/Time processing completed for CSV data")

# Read first XLSX file (2018-2023 data)
print("📊 Loading XLSX file: LFB Incident data from 2018 - 2023.xlsx")
xlsx_file1_path = "LFB Incident data from 2018 - 2023.xlsx"
df_xlsx1 = pd.read_excel(
    xlsx_file1_path,
    dtype=dtype_spec
)
print(f"   ✅ XLSX 2018-2023 loaded: {len(df_xlsx1):,} records, {len(df_xlsx1.columns)} columns")

# Process date and time columns for first XLSX data
print("   ⏰ Processing date/time columns for 2018-2023 data...")
df_xlsx1['DateOfCall'] = pd.to_datetime(df_xlsx1['DateOfCall'], format='mixed').dt.date
df_xlsx1['DateOfCall'] = pd.to_datetime(df_xlsx1['DateOfCall'])
df_xlsx1['TimeOfCall'] = pd.to_timedelta(df_xlsx1['TimeOfCall'].astype(str))
print("   ✅ Date/Time processing completed for 2018-2023 data")

# Read second XLSX file (2024 onwards data)
print("📊 Loading XLSX file: LFB Incident data from 2024 onwards.xlsx")
xlsx_file2_path = "LFB Incident data from 2024 onwards.xlsx"
df_xlsx2 = pd.read_excel(
    xlsx_file2_path,
    dtype=dtype_spec
)
print(f"   ✅ XLSX 2024+ loaded: {len(df_xlsx2):,} records, {len(df_xlsx2.columns)} columns")

# Process date and time columns for second XLSX data
print("   ⏰ Processing date/time columns for 2024+ data...")
df_xlsx2['DateOfCall'] = pd.to_datetime(df_xlsx2['DateOfCall'], format='mixed').dt.date
df_xlsx2['DateOfCall'] = pd.to_datetime(df_xlsx2['DateOfCall'])
df_xlsx2['TimeOfCall'] = pd.to_timedelta(df_xlsx2['TimeOfCall'].astype(str))
print("   ✅ Date/Time processing completed for 2024+ data")

print("🔄 STEP 2: Combining Incident Data...")
print("-" * 40)

# Combine all incident dataframes into one master dataframe
print("🔗 Concatenating all incident dataframes...")
LFB_Inc = pd.concat([df_csv, df_xlsx1, df_xlsx2], ignore_index=True)
print(f"   ✅ Combined incident data: {len(LFB_Inc):,} total records")

# Convert specific columns to category dtype for memory optimization and performance
print("🏷️  Converting text columns to category dtype for optimization...")
category_columns = [
    'StopCodeDescription', 'SpecialServiceType', 'PropertyType',
    'IncGeo_WardCode', 'IncGeo_WardName', 'IncGeo_WardNameNew',
    'IncidentStationGround', 'FirstPumpArriving_DeployedFromStation',
    'SecondPumpArriving_DeployedFromStation'
]

for col in category_columns:
    if col in LFB_Inc.columns:
        if col == 'PropertyType':  # Special handling for PropertyType
            LFB_Inc[col] = LFB_Inc[col].str.strip().astype('category')
        else:
            LFB_Inc[col] = LFB_Inc[col].astype('category')
        print(f"   ✅ Converted {col} to category")

print(f"🎉 Incident data processing completed! Final size: {len(LFB_Inc):,} records")

print("\n" + "=" * 60)
print("📁 STEP 3: Loading Mobilisation Data Files...")
print("-" * 40)

# Load mobilisation data from multiple files
print("📊 Loading mobilisation data files...")
df_2009_2014 = pd.read_excel("LFB Mobilisation data from January 2009 - 2014.xlsx")
df_2015_2020 = pd.read_excel("LFB Mobilisation data from 2015 - 2020.xlsx")
df_2021_2024 = pd.read_csv("LFB Mobilisation data from 2021 - 2024.csv")
df_2025 = pd.read_csv("LFB Mobilisation data from 2025.csv")

print(f"   ✅ 2009-2014: {len(df_2009_2014):,} records")
print(f"   ✅ 2015-2020: {len(df_2015_2020):,} records")
print(f"   ✅ 2021-2024: {len(df_2021_2024):,} records")
print(f"   ✅ 2025: {len(df_2025):,} records")

def convert_dtypes(df, df_name=""):
    """
    Convert data types for LFB dataset columns to optimize memory usage and ensure consistency

    Parameters:
    df (pd.DataFrame): DataFrame to modify
    df_name (str): Optional name for logging/debugging

    Returns:
    pd.DataFrame: Modified DataFrame with converted dtypes
    """
    if df_name:
        print(f"   🔧 Converting dtypes for {df_name}...")

    # Define the conversion mapping for mobilisation data
    dtype_conversions = {
        'IncidentNumber': 'string',
        'CalYear': 'Int16',
        'HourOfCall': 'UInt8',
        'ResourceMobilisationId': 'string',
        'Resource_Code': 'category',
        'PerformanceReporting': 'category',
        'TurnoutTimeSeconds': 'Int16',
        'TravelTimeSeconds': 'Int16',
        'AttendanceTimeSeconds': 'Int16',
        'DeployedFromStation_Code': 'category',
        'DeployedFromStation_Name': 'category',
        'DeployedFromLocation': 'category',
        'PumpOrder': 'category',
        'PlusCode_Code': 'category',
        'PlusCode_Description': 'category',
        'DelayCodeId': 'category',
        'DelayCode_Description': 'category'
    }

    # Apply conversions only for columns that exist in the dataframe
    conversion_count = 0
    for column, dtype in dtype_conversions.items():
        if column in df.columns:
            df[column] = df[column].astype(dtype)
            conversion_count += 1
        elif df_name:
            print(f"   ⚠️  Warning: Column '{column}' not found in {df_name}")

    if df_name:
        print(f"   ✅ Converted {conversion_count} columns in {df_name}")

    return df

print("🔄 STEP 4: Processing Mobilisation Data...")
print("-" * 40)

# Create dictionary of dataframes for batch processing
dataframes = {
    '2009-2014': df_2009_2014,
    '2015-2020': df_2015_2020,
    '2021-2024': df_2021_2024,
    '2025': df_2025
}

# Apply dtype conversion to all mobilisation dataframes
for name, df in dataframes.items():
    dataframes[name] = convert_dtypes(df, f"df_{name}")

# Update the original dataframe variables
df_2009_2014, df_2015_2020, df_2021_2024, df_2025 = dataframes.values()

print("🗑️  Removing unnecessary columns from recent data...")
# Remove redundant columns from 2021-2024 and 2025 data
columns_to_drop = ['BoroughName', 'WardName']
for col in columns_to_drop:
    if col in df_2021_2024.columns:
        df_2021_2024 = df_2021_2024.drop(columns=col)
        print(f"   ✅ Dropped {col} from 2021-2024 data")

for col in columns_to_drop:
    if col in df_2025.columns:
        df_2025 = df_2025.drop(columns=col)
        print(f"   ✅ Dropped {col} from 2025 data")

print("🔗 Combining all mobilisation dataframes...")
# Combine all mobilisation dataframes
LFB_Mob = pd.concat(
    [df_2009_2014, df_2015_2020, df_2021_2024, df_2025],
    ignore_index=True,
    sort=False
)
print(f"   ✅ Combined mobilisation data: {len(LFB_Mob):,} total records")

# Apply final dtype conversion and process datetime columns
print("⏰ Processing datetime columns for mobilisation data...")
LFB_Mob = convert_dtypes(LFB_Mob, "LFB_Mob (final)")
LFB_Mob['DateAndTimeMobilised'] = pd.to_datetime(LFB_Mob['DateAndTimeMobilised'])
LFB_Mob['DateAndTimeMobile'] = pd.to_datetime(LFB_Mob['DateAndTimeMobile'])
LFB_Mob['DateAndTimeArrived'] = pd.to_datetime(LFB_Mob['DateAndTimeArrived'])
LFB_Mob['DateAndTimeLeft'] = pd.to_datetime(LFB_Mob['DateAndTimeLeft'])
print("   ✅ Datetime processing completed for mobilisation data")

print(f"🎉 Mobilisation data processing completed! Final size: {len(LFB_Mob):,} records")

print("\n" + "=" * 60)
print("📊 STEP 5: Creating Data Type Documentation...")
print("-" * 40)

print("📋 Gathering data type information from both datasets...")
# Create data type documentation for both datasets
dtypes_LFB_Inc = pd.DataFrame({
    'Column': LFB_Inc.columns,
    'Dtype': LFB_Inc.dtypes.values,
    'Source': 'LFB_Inc'
})

dtypes_LFB_Mob = pd.DataFrame({
    'Column': LFB_Mob.columns,
    'Dtype': LFB_Mob.dtypes.values,
    'Source': 'LFB_Mob'
})

print("🔗 Combining data type information...")
LFB_stacked_dtypes = pd.concat([dtypes_LFB_Inc, dtypes_LFB_Mob], ignore_index=True)
print(f"   ✅ Combined dtype info: {len(LFB_stacked_dtypes)} columns documented")

print("\n" + "=" * 60)
print("📊 STEP 6: Creating unique and NA Data Documentation...")
print("-" * 40)

print("📋 Gathering unique values and empty cells (isna) data information from both datasets...")
# Create unique values and empty cells (isna) data documentation for both datasets
LFB_Inc_cal = pd.concat([LFB_Inc.notna().sum(),LFB_Inc.isna().sum(),LFB_Inc.nunique()],axis=1)
LFB_Inc_cal = LFB_Inc_cal.reset_index()
LFB_Inc_cal.columns = ['Column', 'Not_NA_Count', 'NA_Count','unique_Count']

LFB_Mob_cal = pd.concat([LFB_Mob.notna().sum(),LFB_Mob.isna().sum(),LFB_Mob.nunique()],axis=1)
LFB_Mob_cal = LFB_Mob_cal.reset_index()
LFB_Mob_cal.columns = ['Column', 'Not_NA_Count', 'NA_Count','unique_Count']

print("🔗 Combining unique and empty cells (isna) data information...")
LFB_cal = pd.concat([LFB_Inc_cal, LFB_Mob_cal], ignore_index=True)
print(f"   ✅ Combined unique and isna info: {len(LFB_cal)} columns documented")

print("📖 Loading and combining metadata files...")
# Load and combine metadata from Excel files
LFB_Inc_md = pd.read_excel('Metadata.xlsx')
LFB_Mob_md = pd.read_excel('Mobilisations Metadata.xlsx')
LFB_md = pd.concat([LFB_Inc_md, LFB_Mob_md], ignore_index=True)

print("🔍 Merging metadata with data types and NA & unique value counts...")
# Merge metadata with data types,NA and unique values
LFB_md_dt = pd.merge(pd.merge(LFB_md,LFB_stacked_dtypes,on='Column'),LFB_cal,on='Column')
print("💾 Saving final metadata documentation...")
# Save the comprehensive metadata documentation
LFB_md_dt.to_excel('LFB_md_dt.xlsx', index=False)
print("   ✅ Metadata saved to 'LFB_md_dt.xlsx'")

print("\n" + "=" * 60)
print("🎉 PROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"📈 FINAL DATASET SUMMARY:")
print(f"   • LFB_Inc: {len(LFB_Inc):,} records")
print(f"   • LFB_Mob: {len(LFB_Mob):,} records")
print(f"   • Total documented columns: {len(LFB_stacked_dtypes)}")
print(f"   • Metadata file: LFB_md_dt.xlsx")
print("=" * 60)

df = LFB_Inc.copy()
df['Postcode_full'] = df['Postcode_full'].fillna('SW17 0QT')
categorical_columns = df.select_dtypes(include=['category']).columns

def fill_with_mode(series):
    mode_val = series.mode()
    if not mode_val.empty:
        return series.fillna(mode_val[0])
    return series

df[categorical_columns] = df[categorical_columns].apply(fill_with_mode)

numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = df[numerical_cols].fillna(0)


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

# Fixed data preprocessing function
def preprocess_data(df):
    # Make a copy to avoid modifying the original
    df_processed = df.copy()

    # Convert date columns if needed
    if 'DateOfCall' in df_processed.columns:
        df_processed['DateOfCall'] = pd.to_datetime(df_processed['DateOfCall'], errors='coerce')
        df_processed['Month'] = df_processed['DateOfCall'].dt.month
        df_processed['DayOfWeek'] = df_processed['DateOfCall'].dt.day_name()

    # Handle categorical columns properly
    categorical_columns = ['IncidentGroup', 'StopCodeDescription', 'IncGeo_BoroughName']

    for col in categorical_columns:
        if col in df_processed.columns:
            # Convert to string first, then handle missing values
            df_processed[col] = df_processed[col].astype(str)
            df_processed[col] = df_processed[col].replace('nan', 'Unknown')
            df_processed[col] = df_processed[col].fillna('Unknown')
            # Convert back to categorical if it was categorical
            if hasattr(df[col], 'cat'):
                df_processed[col] = pd.Categorical(df_processed[col])

    return df_processed

# Alternative simpler preprocessing function
def preprocess_data_simple(df):
    """Simpler preprocessing that avoids categorical issues"""
    df_processed = df.copy()

    # Convert date columns
    if 'DateOfCall' in df_processed.columns:
        df_processed['DateOfCall'] = pd.to_datetime(df_processed['DateOfCall'], errors='coerce')
        df_processed['Month'] = df_processed['DateOfCall'].dt.month
        df_processed['DayOfWeek'] = df_processed['DateOfCall'].dt.day_name()

    # Convert potential categorical columns to string to avoid issues
    categorical_cols = ['IncidentGroup', 'StopCodeDescription', 'IncGeo_BoroughName', 'SpecialServiceType']

    for col in categorical_cols:
        if col in df_processed.columns:
            # Convert to string and handle missing values
            df_processed[col] = df_processed[col].astype(str)
            df_processed.loc[df_processed[col] == 'nan', col] = 'Unknown'
            df_processed[col] = df_processed[col].fillna('Unknown')

    return df_processed

# Preprocess your data (use the simple version to avoid issues)
df_processed = preprocess_data_simple(df)

# Get unique values for filters
available_years = sorted(df_processed['CalYear'].unique())
boroughs = sorted(df_processed['IncGeo_BoroughName'].unique())
incident_groups = sorted(df_processed['IncidentGroup'].unique())

# Remove 'Unknown' from filter options if present
boroughs = [b for b in boroughs if b != 'Unknown']
incident_groups = [ig for ig in incident_groups if ig != 'Unknown']

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("London Fire Brigade Incident Records",
                style={'color': '#1a3f6c', 'marginBottom': '10px', 'fontWeight': 'bold'}),
        html.P("Interactive dashboard showing London Fire Brigade incident data",
               style={'color': '#666', 'fontSize': '16px'})
    ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderBottom': '2px solid #dee2e6'}),

    # Filters Section
    html.Div([
        html.Div([
            html.Label("Year Range:", style={'fontWeight': 'bold'}),
            dcc.RangeSlider(
                id='year-slider',
                min=available_years[0],
                max=available_years[-1],
                value=[available_years[-3], available_years[-1]],
                marks={str(year): str(year) for year in available_years if year % 2 == 0},
                step=1
            )
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.Div([
                html.Label("Borough:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='borough-dropdown',
                    options=[{'label': 'All Boroughs', 'value': 'all'}] +
                            [{'label': borough, 'value': borough} for borough in boroughs],
                    value='all',
                    multi=False
                )
            ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '1%'}),

            html.Div([
                html.Label("Incident Type:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='incident-type-dropdown',
                    options=[{'label': 'All Types', 'value': 'all'}] +
                            [{'label': inc_type, 'value': inc_type} for inc_type in incident_groups],
                    value='all',
                    multi=False
                )
            ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '1%'}),

            html.Div([
                html.Label("Time Period:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='time-period-dropdown',
                    options=[
                        {'label': 'All Time', 'value': 'all'},
                        {'label': 'Last 7 Days', 'value': '7d'},
                        {'label': 'Last 30 Days', 'value': '30d'},
                        {'label': 'Last 90 Days', 'value': '90d'},
                        {'label': 'Last Year', 'value': '1y'}
                    ],
                    value='all'
                )
            ], style={'width': '32%', 'display': 'inline-block'})
        ], style={'marginBottom': '20px'})
    ], style={'padding': '20px', 'backgroundColor': 'white', 'borderBottom': '1px solid #dee2e6'}),

    # Key Metrics Section
    html.Div([
        html.Div([
            html.Div([
                html.H4("TOTAL INCIDENTS", style={'color': '#666', 'marginBottom': '5px', 'fontSize': '14px'}),
                html.H2(id="total-incidents", style={'color': '#1a3f6c', 'margin': '0', 'fontSize': '32px'})
            ], className='metric-box', style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white',
                                            'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Div([
                html.H4("FIRES", style={'color': '#666', 'marginBottom': '5px', 'fontSize': '14px'}),
                html.H2(id="fires-count", style={'color': '#dc3545', 'margin': '0', 'fontSize': '32px'})
            ], className='metric-box', style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white',
                                            'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Div([
                html.H4("SPECIAL SERVICES", style={'color': '#666', 'marginBottom': '5px', 'fontSize': '14px'}),
                html.H2(id="special-services-count", style={'color': '#28a745', 'margin': '0', 'fontSize': '32px'})
            ], className='metric-box', style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white',
                                            'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Div([
                html.H4("FALSE ALARMS", style={'color': '#666', 'marginBottom': '5px', 'fontSize': '14px'}),
                html.H2(id="false-alarms-count", style={'color': '#ffc107', 'margin': '0', 'fontSize': '32px'})
            ], className='metric-box', style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white',
                                            'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px'})
    ], style={'padding': '20px', 'textAlign': 'center'}),

    # Charts Section
    html.Div([
        # First row of charts
        html.Div([
            html.Div([
                html.H4("Incidents by Type", style={'textAlign': 'center', 'color': '#1a3f6c'}),
                dcc.Graph(id='incident-type-chart')
            ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),

            html.Div([
                html.H4("Incidents by Hour of Day", style={'textAlign': 'center', 'color': '#1a3f6c'}),
                dcc.Graph(id='hourly-distribution-chart')
            ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'}),

            html.Div([
                html.H4("Monthly Trend", style={'textAlign': 'center', 'color': '#1a3f6c'}),
                dcc.Graph(id='monthly-trend-chart')
            ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px'})
        ]),

        # Second row of charts
        html.Div([
            html.Div([
                html.H4("Top Boroughs", style={'textAlign': 'center', 'color': '#1a3f6c'}),
                dcc.Graph(id='borough-chart')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

            html.Div([
                html.H4("Incident Response Times", style={'textAlign': 'center', 'color': '#1a3f6c'}),
                dcc.Graph(id='response-time-chart')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
        ])
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa'}),

    # Data Table Section
    html.Div([
        html.H4("Incident Details", style={'color': '#1a3f6c', 'marginBottom': '15px'}),
        html.Div([
            dcc.Dropdown(
                id='table-page-size',
                options=[
                    {'label': 'Show 10 rows', 'value': 10},
                    {'label': 'Show 25 rows', 'value': 25},
                    {'label': 'Show 50 rows', 'value': 50}
                ],
                value=10,
                style={'width': '200px', 'marginBottom': '10px'}
            )
        ]),
        html.Div(id='incidents-table-container')
    ], style={'padding': '20px'}),

    # Download Section
    html.Div([
        html.H4("Download Filtered Data", style={'color': '#1a3f6c', 'marginBottom': '15px'}),
        html.Button("Download CSV", id="btn-download-csv",
                   style={'backgroundColor': '#1a3f6c', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px'}),
        dcc.Download(id="download-dataframe-csv")
    ], style={'padding': '20px', 'borderTop': '1px solid #dee2e6'})
])

# Callback to update all components based on filters
@app.callback(
    [Output('total-incidents', 'children'),
     Output('fires-count', 'children'),
     Output('special-services-count', 'children'),
     Output('false-alarms-count', 'children'),
     Output('incident-type-chart', 'figure'),
     Output('hourly-distribution-chart', 'figure'),
     Output('monthly-trend-chart', 'figure'),
     Output('borough-chart', 'figure'),
     Output('response-time-chart', 'figure'),
     Output('incidents-table-container', 'children')],
    [Input('year-slider', 'value'),
     Input('borough-dropdown', 'value'),
     Input('incident-type-dropdown', 'value'),
     Input('time-period-dropdown', 'value'),
     Input('table-page-size', 'value')]
)
def update_dashboard(selected_years, selected_borough, selected_incident_type, selected_time_period, page_size):
    # Filter data based on selections
    filtered_df = df_processed.copy()

    # Year filter
    filtered_df = filtered_df[
        (filtered_df['CalYear'] >= selected_years[0]) &
        (filtered_df['CalYear'] <= selected_years[1])
    ]

    # Borough filter
    if selected_borough != 'all':
        filtered_df = filtered_df[filtered_df['IncGeo_BoroughName'] == selected_borough]

    # Incident type filter
    if selected_incident_type != 'all':
        filtered_df = filtered_df[filtered_df['IncidentGroup'] == selected_incident_type]

    # Time period filter (only if DateOfCall exists)
    if selected_time_period != 'all' and 'DateOfCall' in filtered_df.columns:
        today = pd.Timestamp.now()
        if selected_time_period == '7d':
            cutoff_date = today - pd.Timedelta(days=7)
        elif selected_time_period == '30d':
            cutoff_date = today - pd.Timedelta(days=30)
        elif selected_time_period == '90d':
            cutoff_date = today - pd.Timedelta(days=90)
        elif selected_time_period == '1y':
            cutoff_date = today - pd.Timedelta(days=365)

        filtered_df = filtered_df[filtered_df['DateOfCall'] >= cutoff_date]

    # Calculate metrics
    total_incidents = len(filtered_df)
    fires_count = len(filtered_df[filtered_df['IncidentGroup'] == 'Fire'])
    special_services_count = len(filtered_df[filtered_df['IncidentGroup'] == 'Special Service'])
    false_alarms_count = len(filtered_df[filtered_df['IncidentGroup'] == 'False Alarm'])

    # Format numbers with commas
    total_incidents_str = f"{total_incidents:,}"
    fires_count_str = f"{fires_count:,}"
    special_services_str = f"{special_services_count:,}"
    false_alarms_str = f"{false_alarms_count:,}"

    # 1. Incident Type Chart (Pie)
    incident_type_data = filtered_df['IncidentGroup'].value_counts()
    incident_type_fig = px.pie(
        values=incident_type_data.values,
        names=incident_type_data.index,
        color=incident_type_data.index,
        color_discrete_map={
            'Fire': '#dc3545',
            'Special Service': '#28a745',
            'False Alarm': '#ffc107'
        }
    )
    incident_type_fig.update_layout(showlegend=True, margin=dict(t=30, b=0, l=0, r=0))

    # 2. Hourly Distribution Chart
    hourly_data = filtered_df['HourOfCall'].value_counts().sort_index()
    hourly_fig = go.Figure()
    hourly_fig.add_trace(go.Bar(
        x=hourly_data.index,
        y=hourly_data.values,
        marker_color='#1a3f6c',
        opacity=0.8
    ))
    hourly_fig.update_layout(
        xaxis_title='Hour of Day',
        yaxis_title='Number of Incidents',
        showlegend=False,
        margin=dict(t=30, b=0, l=0, r=0)
    )

    # 3. Monthly Trend Chart
    monthly_trend = filtered_df.groupby(['CalYear', 'IncidentGroup']).size().reset_index(name='Count')
    trend_fig = px.line(
        monthly_trend,
        x='CalYear',
        y='Count',
        color='IncidentGroup',
        color_discrete_map={
            'Fire': '#dc3545',
            'Special Service': '#28a745',
            'False Alarm': '#ffc107'
        }
    )
    trend_fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Number of Incidents',
        margin=dict(t=30, b=0, l=0, r=0)
    )

    # 4. Borough Chart
    borough_data = filtered_df['IncGeo_BoroughName'].value_counts().head(10)
    borough_fig = px.bar(
        x=borough_data.values,
        y=borough_data.index,
        orientation='h',
        color=borough_data.values,
        color_continuous_scale='Blues'
    )
    borough_fig.update_layout(
        xaxis_title='Number of Incidents',
        yaxis_title='Borough',
        showlegend=False,
        margin=dict(t=30, b=0, l=0, r=0)
    )

    # 5. Response Time Chart (if data available)
    response_time_fig = go.Figure()
    if 'FirstPumpArriving_AttendanceTime' in filtered_df.columns:
        # Convert to numeric and remove non-numeric values
        filtered_df['ResponseTime'] = pd.to_numeric(filtered_df['FirstPumpArriving_AttendanceTime'], errors='coerce')
        response_times = filtered_df.groupby('IncidentGroup')['ResponseTime'].mean().dropna()

        if not response_times.empty:
            response_time_fig.add_trace(go.Bar(
                x=response_times.index,
                y=response_times.values,
                marker_color=['#dc3545', '#28a745', '#ffc107']
            ))
        else:
            response_time_fig.add_annotation(text="No response time data available",
                                           xref="paper", yref="paper",
                                           x=0.5, y=0.5, showarrow=False)
    else:
        response_time_fig.add_annotation(text="Response time data not available",
                                       xref="paper", yref="paper",
                                       x=0.5, y=0.5, showarrow=False)

    response_time_fig.update_layout(
        xaxis_title='Incident Type',
        yaxis_title='Average Response Time (minutes)',
        margin=dict(t=30, b=0, l=0, r=0)
    )

    # 6. Data Table
    table_columns = ['IncidentNumber', 'DateOfCall', 'IncidentGroup', 'StopCodeDescription', 'IncGeo_BoroughName']
    available_columns = [col for col in table_columns if col in filtered_df.columns]

    table_data = filtered_df[available_columns].head(page_size)

    table = dash_table.DataTable(
        data=table_data.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in available_columns],
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={
            'backgroundColor': '#1a3f6c',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data={
            'backgroundColor': 'white',
            'color': 'black'
        },
        page_size=page_size
    )

    return (total_incidents_str, fires_count_str, special_services_str, false_alarms_str,
            incident_type_fig, hourly_fig, trend_fig, borough_fig, response_time_fig, table)

# Download callback
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-download-csv", "n_clicks"),
    [State('year-slider', 'value'),
     State('borough-dropdown', 'value'),
     State('incident-type-dropdown', 'value'),
     State('time-period-dropdown', 'value')],
    prevent_initial_call=True,
)
def download_csv(n_clicks, selected_years, selected_borough, selected_incident_type, selected_time_period):
    # Apply the same filters as the main callback
    filtered_df = df_processed.copy()
    filtered_df = filtered_df[
        (filtered_df['CalYear'] >= selected_years[0]) &
        (filtered_df['CalYear'] <= selected_years[1])
    ]

    if selected_borough != 'all':
        filtered_df = filtered_df[filtered_df['IncGeo_BoroughName'] == selected_borough]

    if selected_incident_type != 'all':
        filtered_df = filtered_df[filtered_df['IncidentGroup'] == selected_incident_type]

    return dcc.send_data_frame(filtered_df.to_csv, "lfb_incidents_filtered.csv")


print('visit http://127.0.0.1:8050')
app.run(debug=False,port=8050)
print("Press Enter to exit...")
input()  # Wait for Enter key
print("Exiting program...")

#if __name__ == '__main__':
    #print('visit http://127.0.0.1:8050')
    #app.run(debug=True, port=8050)
    #print("Press Enter to exit...")
    #input()  # Wait for Enter key
    #print("Exiting program...")

#print("Press Enter to exit...")
#input()  # Wait for Enter key

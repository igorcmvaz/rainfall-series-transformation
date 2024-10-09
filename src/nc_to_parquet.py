import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Função para extrair os dados de precipitação
def extract_precipitation(file_path, target_lat, target_lon):
    try:
        dataset = nc.Dataset(file_path, 'r')
    except OSError as e:
        print(f"Erro ao abrir o arquivo: {e}")
        return None

    latitudes = dataset.variables['lat'][:]
    longitudes = dataset.variables['lon'][:]
    times = dataset.variables['time'][:]
    precipitation = dataset.variables['pr'][:]

    lat_index = np.abs(latitudes - target_lat).argmin()
    lon_index = np.abs(longitudes - target_lon).argmin()

    time_units = dataset.variables['time'].units
    reference_date_str = time_units.split('since')[1].strip()
    reference_date = datetime.strptime(reference_date_str, "%Y-%m-%dT%H:%M:%S")
    dates = [reference_date + timedelta(days=float(t)) for t in times]

    precipitation_series = [
        (dates[t_index], float(precipitation[t_index, lat_index, lon_index]))
        for t_index in range(len(dates))
        if not np.ma.is_masked(precipitation[t_index, lat_index, lon_index])
    ]

    dataset.close()
    return precipitation_series

# Função para calcular os índices climáticos
def calculate_indices(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df_wet_days = df[df['precipitation'] >= 1]

    prcptot = df_wet_days.groupby('year')['precipitation'].sum()
    r95_threshold = df['precipitation'].quantile(0.95)
    r95p = df[df['precipitation'] > r95_threshold].groupby('year')['precipitation'].sum()
    rx1day = df.groupby('year')['precipitation'].max()
    df['rolling_5day'] = df['precipitation'].rolling(window=5, min_periods=1).sum()
    rx5day = df.groupby('year')['rolling_5day'].max()
    sdii = df_wet_days.groupby('year')['precipitation'].mean()
    r20mm = df[df['precipitation'] > 20].groupby('year').size()
    df['dry'] = df['precipitation'] < 1
    cdd = df.groupby('year')['dry'].apply(lambda x: x.astype(int).groupby((x != x.shift()).cumsum()).sum().max())
    df['wet'] = df['precipitation'] >= 1
    cwd = df.groupby('year')['wet'].apply(lambda x: x.astype(int).groupby((x != x.shift()).cumsum()).sum().max())

    # Cálculo do índice de sazonalidade (S)
    Ai = df.groupby('year')['precipitation'].sum()  # Precipitação total anual
    Mi = df.groupby(['year', 'month'])['precipitation'].mean()  # Precipitação média mensal

    # Cálculo do índice de sazonalidade S para cada ano
    seasonality_index = Ai.copy()
    for year in Ai.index:
        total_precip = Ai[year]
        monthly_precip = Mi.loc[year]
        S = (1 / total_precip) * np.sum(np.abs(monthly_precip - (total_precip / 12)))
        seasonality_index[year] = S

    result_df = pd.DataFrame({
        'PRCPTOT': prcptot,
        'R95p': r95p,
        'RX1day': rx1day,
        'RX5day': rx5day,
        'SDII': sdii,
        'R20mm': r20mm,
        'CDD': cdd,
        'CWD': cwd,
        'Seasonality_Index': seasonality_index  # Índice de sazonalidade adicionado
    }).reset_index()

    return result_df

# Carregar as coordenadas corrigidas do arquivo CSV
coords_df = pd.read_csv('coordenadas_capitais_validas.csv')

# Lista dos modelos
models = [
    'ACCESS-CM2', 'ACCESS-ESM1-5', 'CMCC-ESM2', 'EC-EARTH3', 'GFDL-CM4', 'GFDL-ESM4',
    'HadGEM3-GC31-LL', 'INM-CM4_8', 'INM-CM5', 'IPSL-CM6A-LR', 'KACE', 'KIOST', 'MIROC6',
    'MPI-ESM1-2', 'MRI-ESM2', 'NESM3', 'NorESM2-MM', 'TaiESM1', 'UKESM1-0-LL'
]

# DataFrame para armazenar todos os dados
all_data = []

counter = 0

# Loop sobre as cidades e modelos
for i, row in coords_df.iterrows():
    city = row['Capital']
    target_lat = row['Nearest Latitude']
    target_lon = row['Nearest Longitude']
    print(counter)

    for model in models:
        counter += 1
        files = {
            'Histórico': f'{model}-pr-hist.nc',
            'SSP245': f'{model}-pr-ssp245.nc',  # Cenário adicionado
            'SSP585': f'{model}-pr-ssp585.nc'
        }

        for scenario, file_path in files.items():
            series = extract_precipitation(file_path, target_lat, target_lon)
            if series:
                dates, values = zip(*series)
                df = pd.DataFrame({'date': pd.to_datetime(dates), 'precipitation': values})
                indices = calculate_indices(df)

                # Adiciona os dados da cidade, modelo, cenário, latitude e longitude
                indices['City'] = city
                indices['Model'] = model
                indices['Scenario'] = scenario
                indices['Latitude'] = target_lat
                indices['Longitude'] = target_lon

                # Append the data to the main dataframe
                all_data.append(indices)

# Concatenar todos os dados em um único DataFrame
all_data_df = pd.concat(all_data, ignore_index=True)

# Salvar o DataFrame como um arquivo Parquet com nome atualizado
all_data_df.to_parquet('climate_indices_with_corrected_coordinates_V3.parquet', index=False)

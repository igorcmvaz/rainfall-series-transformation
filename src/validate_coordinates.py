import netCDF4 as nc
import numpy as np
import pandas as pd

# Função para verificar se há dados de precipitação na coordenada fornecida
def has_precipitation_data(precipitation, lat_index, lon_index):
    # Verifica se todos os dados ao longo do tempo estão mascarados (sem dados)
    return not np.all(np.ma.getmaskarray(precipitation[:, lat_index, lon_index]))

# Função para encontrar a coordenada válida mais próxima
def find_nearest_valid_coordinate(latitudes, longitudes, precipitation, target_lat, target_lon):
    lat_index = np.abs(latitudes - target_lat).argmin()
    lon_index = np.abs(longitudes - target_lon).argmin()

    # Inicializa a variável found como False
    found = False

    # Se a coordenada inicial não tiver dados, procuramos pela mais próxima que tenha
    if not has_precipitation_data(precipitation, lat_index, lon_index):
        distance = 1  # Começa a procurar em torno da coordenada inicial
        max_distance = 15  # Define um limite de quantas posições ao redor podem ser verificadas

        while distance <= max_distance and not found:
            for lat_shift in range(-distance, distance + 1):
                for lon_shift in range(-distance, distance + 1):
                    new_lat_index = lat_index + lat_shift
                    new_lon_index = lon_index + lon_shift
                    if 0 <= new_lat_index < len(latitudes) and 0 <= new_lon_index < len(longitudes):
                        if has_precipitation_data(precipitation, new_lat_index, new_lon_index):
                            lat_index, lon_index = new_lat_index, new_lon_index
                            found = True
                            break
                if found:
                    break
            distance += 1

    # Verifica se encontrou uma coordenada válida
    if found or has_precipitation_data(precipitation, lat_index, lon_index):
        return lat_index, lon_index, latitudes[lat_index], longitudes[lon_index]
    else:
        # Caso não encontre uma coordenada válida dentro do limite, retorna None
        return None, None, None, None

# Função para gerar o CSV com as coordenadas válidas
def generate_valid_coordinates_csv(file_path, capitals_coords, output_csv):
    # Tentar abrir o arquivo NetCDF
    try:
        dataset = nc.Dataset(file_path, 'r')
    except OSError as e:
        print(f"Erro ao abrir o arquivo: {e}")
        return

    # Obter latitudes, longitudes e precipitação do arquivo NetCDF
    latitudes = dataset.variables['lat'][:]
    longitudes = dataset.variables['lon'][:]
    precipitation = dataset.variables['pr'][:]

    # Lista para armazenar as coordenadas válidas
    valid_coords = []

    # Loop sobre todas as capitais
    for city, (target_lat, target_lon) in capitals_coords.items():
        # Encontrar a coordenada válida mais próxima
        lat_index, lon_index, valid_lat, valid_lon = find_nearest_valid_coordinate(latitudes, longitudes, precipitation, target_lat, target_lon)
        
        # Verifica se uma coordenada válida foi encontrada
        if valid_lat is not None and valid_lon is not None:
            valid_coords.append({
                "Capital": city,
                "Target Latitude": target_lat,
                "Target Longitude": target_lon,
                "Nearest Latitude": valid_lat,
                "Nearest Longitude": valid_lon
            })
        else:
            print(f"Não foi possível encontrar uma coordenada válida para {city}")

    # Converter a lista de coordenadas válidas em um DataFrame
    df_valid_coords = pd.DataFrame(valid_coords)

    # Salvar o DataFrame em um arquivo CSV
    df_valid_coords.to_csv(output_csv, index=False)
    print(f"CSV gerado: {output_csv}")

    # Fechar o dataset
    dataset.close()

# Coordenadas das 27 capitais brasileiras
capitals_coords = {
    "Brasília": (-15.7801, -47.9292),
    "Rio de Janeiro": (-22.9068, -43.1729),
    "São Paulo": (-23.5505, -46.6333),
    "Salvador": (-12.9714, -38.5014),
    "Fortaleza": (-3.7172, -38.5434),
    "Belo Horizonte": (-19.9167, -43.9345),
    "Manaus": (-3.1019, -60.0250),
    "Curitiba": (-25.4284, -49.2733),
    "Recife": (-8.0476, -34.8770),
    "Porto Alegre": (-30.0346, -51.2177),
    "Belém": (-1.4558, -48.5039),
    "Goiânia": (-16.6869, -49.2648),
    "São Luís": (-2.5387, -44.2829),
    "Maceió": (-9.6659, -35.7350),
    "Natal": (-5.7945, -35.2110),
    "Teresina": (-5.0892, -42.8019),
    "Campo Grande": (-20.4697, -54.6201),
    "João Pessoa": (-7.1195, -34.8450),
    "Aracaju": (-10.9472, -37.0731),
    "Cuiabá": (-15.6010, -56.0979),
    "Palmas": (-10.1675, -48.3277),
    "Boa Vista": (2.8194, -60.6738),
    "Macapá": (0.0349, -51.0694),
    "Porto Velho": (-8.7612, -63.9004),
    "Rio Branco": (-9.974, -67.8243),
    "Florianópolis": (-27.5954, -48.5480),
    "Vitória": (-20.3155, -40.3128)
}

# Caminho do arquivo NetCDF (substitua pelo caminho correto)
file_path = 'TaiESM1-pr-ssp585.nc'

# Nome do arquivo CSV de saída
output_csv = 'coordenadas_capitais_validas.csv'

# Gerar o CSV com as coordenadas válidas
generate_valid_coordinates_csv(file_path, capitals_coords, output_csv)

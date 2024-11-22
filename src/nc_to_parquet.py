import json
import logging
import time
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from nc_to_csv_timeseries import CLIMATE_MODELS, extract_precipitation


def calculate_indices(df: pd.DataFrame):
    # Reference: http://etccdi.pacificclimate.org/list_27_indices.shtml
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


def main(args: Namespace) -> None:
    setup_start: float = time.perf_counter()
    logging.basicConfig(
        format="%(asctime)s    %(levelname)-8.8s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )
    if args.quiet == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.quiet == 2:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.quiet >= 3:
        logging.getLogger().setLevel(logging.ERROR)

    coordinates_path: Path = Path(args.coordinates)
    if not coordinates_path.is_file():
        logging.critical(
            f"File with cities coordinates not found at '{coordinates_path.resolve()}'")
        return None
    with open(coordinates_path) as file:
        city_coordinates: dict[str, dict[str, Sequence[float]]] = json.load(file)
    logging.info(f"Setup time: {round(1000*(time.perf_counter() - setup_start))}ms")

    all_data = []
    counter = 0
    for city, details in city_coordinates.items():
        latitude = details["Nearest Coordinates"][0]
        longitude = details["Nearest Coordinates"][1]

        for model in CLIMATE_MODELS:
            counter += 1
            files = {
                'Histórico': f'{model}-pr-hist.nc',
                'SSP245': f'{model}-pr-ssp245.nc',
                'SSP585': f'{model}-pr-ssp585.nc'
            }

            for scenario, file_path in files.items():
                series = extract_precipitation(Path(file_path), latitude, longitude)
                if series:
                    dates, values = zip(*series)
                    df = pd.DataFrame({
                        'date': pd.to_datetime(dates),
                        'precipitation': values})
                    indices = calculate_indices(df)

                    # Adiciona os dados da cidade, modelo, cenário, latitude e longitude
                    indices['City'] = city
                    indices['Model'] = model
                    indices['Scenario'] = scenario
                    indices['Latitude'] = latitude
                    indices['Longitude'] = longitude

                    # Append the data to the main dataframe
                    all_data.append(indices)

    # Concatenar todos os dados em um único DataFrame
    all_data_df = pd.concat(all_data, ignore_index=True)

    # Salvar o DataFrame como um arquivo Parquet com nome atualizado
    all_data_df.to_parquet('climate_indices_with_corrected_coordinates_V3.parquet', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument(
    #     "input", type=str, metavar="path/to/input",
    #     help="path to a directory where the input NetCDF4 files are stored")
    # parser.add_argument(
    #     "-o", "--output", type=str, metavar="path/to/output", default="output",
    #     help="path to a directory where the output files will be saved. "
    #     "Defaults to 'output'")
    parser.add_argument(
        "-q", "--quiet", action="count", default=0,
        help="turn on quiet mode (cumulative), which hides log entries of levels lower "
        "than INFO/WARNING")
    parser.add_argument(
        "-c", "--coordinates", type=str, metavar="path/to/coordinates.json",
        help="path to a JSON file containing coordinates of the desired cities")
    main(parser.parse_args())

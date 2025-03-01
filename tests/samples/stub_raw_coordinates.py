import csv
from pathlib import Path

SAMPLE_RAW_CITIES_PATH = Path(
    Path(__file__).parent, "sample_raw_city_coordinates.csv")
SAMPLE_RAW_COORDINATES = {
    "São Paulo": {
        "ibge_code": 3550308,
        "latitude": -23.533,
        "longitude": -46.64
    },
    "Rio de Janeiro": {
        "ibge_code": 3304557,
        "latitude": -22.913,
        "longitude": -43.2
    },
    "Brasília": {
        "ibge_code": 5300108,
        "latitude": -15.78,
        "longitude": -47.93
    },
    "Fortaleza": {
        "ibge_code": 2304400,
        "latitude": -3.717,
        "longitude": -38.542
    },
    "Salvador": {
        "ibge_code": 2927408,
        "latitude": -12.972,
        "longitude": -38.501
    },
}


class RawCoordinatesSampleGenerator:

    @classmethod
    def create_sample_stub(cls, output_path: Path) -> None:
        with open(output_path, "w+", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerow(["ibge_code", "city", "latitude", "longitude"])
            for city_name, details in SAMPLE_RAW_COORDINATES.items():
                writer.writerow([
                    details["ibge_code"],
                    city_name,
                    details["latitude"],
                    details["longitude"]])


if __name__ == "__main__":
    RawCoordinatesSampleGenerator.create_sample_stub(SAMPLE_RAW_CITIES_PATH)

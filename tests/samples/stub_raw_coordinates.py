import csv
from pathlib import Path

SAMPLE_RAW_CITIES_PATH = Path(
    Path(__file__).parent, "sample_raw_city_coordinates.csv")
SAMPLE_RAW_COORDINATES = {
    "Elsweyr": {
        "ibge_code": 8850308,
        "latitude": -34.0058,
        "longitude": -74.0085
    },
    "Auridon": {
        "ibge_code": 8804557,
        "latitude": -33.6025,
        "longitude": -73.6052
    },
    "Summerset": {
        "ibge_code": 8500108,
        "latitude": -32.9123,
        "longitude": -72.8035
    },
    "Cyrodiil": {
        "ibge_code": 9904400,
        "latitude": -31.4040,
        "longitude": -71.3033
    },

    "Skyrim": {
        "ibge_code": 8927408,
        "latitude": -30.1111,
        "longitude": -70.1313
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

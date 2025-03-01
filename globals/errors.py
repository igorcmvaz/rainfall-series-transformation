from pathlib import Path


class InvalidFileError(Exception):
    def __init__(self, source_path: Path):
        super().__init__(f"No file at '{source_path.resolve()}'")


class InvalidTargetCoordinatesError(Exception):
    def __init__(self, latitude: float, longitude: float):
        super().__init__(f"No matching coordinates for ({latitude}, {longitude})")


class CoordinatesNotAvailableError(Exception):
    def __init__(self, details: dict[str, dict[str, float]]):
        super().__init__(
            f"Coordinates not in the expected key path ('nearest'->'lat' and "
            f"'nearest'->'lon'). Actual data: {details}")


class InvalidClimateScenarioError(Exception):
    def __init__(self, scenario: str):
        super().__init__(f"No climate scenario matches the value '{scenario}'")


class InvalidSourceDirectoryError(Exception):
    def __init__(self, source_directory: Path):
        super().__init__(f"Provided path '{source_directory.resolve()}' is not a directory")


class InvalidSourceFileError(InvalidFileError):
    pass


class InvalidCoordinatesFileError(InvalidFileError):
    pass

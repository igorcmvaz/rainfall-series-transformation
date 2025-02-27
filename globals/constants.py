PARQUET_CONF = {
    "engine": "pyarrow",
    "compression": "lz4",
    "compression_level": 11
}

CLIMATE_MODELS: list[str] = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "CMCC-ESM2",
    "EC-EARTH3",
    "GFDL-CM4",
    "GFDL-ESM4",
    "HadGEM3-GC31-LL",
    "INM-CM4_8",
    "INM-CM5",
    "IPSL-CM6A-LR",
    "KACE",
    "KIOST",
    "MIROC6",
    "MPI-ESM1-2",
    "MRI-ESM2",
    "NESM3",
    "NorESM2-MM",
    "TaiESM1",
    "UKESM1-0-LL",
    ]

SSP_SCENARIOS: dict[str, dict[str, str]] = {
    "Histórico":
        {
            "label": "Histórico",
            "start_date": "1980-01-01",
            "end_date": "2013-12-31"
        },
    "SSP245":
        {
            "label": "SSP245_2015_2100",
            "start_date": "2015-01-01",
            "end_date": "2100-12-31"
        },
    "SSP585":
        {
            "label": "SSP585_2015_2100",
            "start_date": "2015-01-01",
            "end_date": "2100-12-31"
        },
    }

INPUT_FILENAME_FORMAT: dict[str, str] = {
    "Histórico": "{model}-pr-hist.nc",
    "SSP245": "{model}-pr-ssp245.nc",
    "SSP585": "{model}-pr-ssp585.nc"
}

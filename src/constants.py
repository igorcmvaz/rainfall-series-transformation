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

SSP_SCENARIOS: dict[str, list[dict[str, str]]] = {
    "Histórico": [
        {
            "label": "Histórico",
            "start_date": "1980-01-01",
            "end_date": "2000-01-01"
        }
    ],
    "SSP245": [
        {
            "label": "SSP245_2015_2035",
            "start_date": "2015-01-01",
            "end_date": "2035-01-01"
        },
        {
            "label": "SSP245_2024_2074",
            "start_date": "2024-01-01",
            "end_date": "2074-01-01"
        },
        {
            "label": "SSP245_2040_2060",
            "start_date": "2040-01-01",
            "end_date": "2060-01-01"
        },
        {
            "label": "SSP245_2060_2080",
            "start_date": "2060-01-01",
            "end_date": "2080-01-01"
        },
    ],
    "SSP585": [
        {
            "label": "SSP585_2015_2035",
            "start_date": "2015-01-01",
            "end_date": "2035-01-01",
        },
        {
            "label": "SSP585_2024_2074",
            "start_date": "2024-01-01",
            "end_date": "2074-01-01"
        },
        {
            "label": "SSP585_2040_2060",
            "start_date": "2040-01-01",
            "end_date": "2060-01-01"
        },
        {
            "label": "SSP585_2060_2080",
            "start_date": "2060-01-01",
            "end_date": "2080-01-01"
        },
    ]
    }

INPUT_FILENAME_FORMAT: dict[str, str] = {
    "Histórico": "{model}-pr-hist.nc",
    "SSP245": "{model}-pr-ssp245.nc",
    "SSP585": "{model}-pr-ssp585.nc"
}

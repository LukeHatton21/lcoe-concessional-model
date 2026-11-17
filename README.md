# LCOE Concessional Model

The LCOE Concessional Model is a research-oriented Python model for estimating the spatial cost and concessional-finance requirements of renewable electricity. It combines gridded renewable resource data, country-level technology costs and weighted average cost of capital (WACC), electricity-price benchmarks, land and offshore constraints, and power-sector emissions factors.

The current code evaluates solar PV, onshore and offshore wind, hydropower, geothermal, and bioenergy. It produces technology-level and least-cost results, estimates the grant-equivalent support needed to reach benchmark electricity prices, and expresses that support as an abatement cost.

> [!IMPORTANT]
> This repository is a partial research snapshot, not a self-contained release. Several required datasets and one local plotting module are not tracked, and `lcoe_model.py` contains machine-specific paths. See [Required external inputs](#required-external-inputs) and [Limitations](#limitations) before attempting a run.

## Methodology and capabilities

At a high level, the model:

1. Aligns gridded solar and wind capacity factors with country, land-cover, bathymetry, and distance-to-shore data.
2. Calculates annual generation and LCOE from country-specific CAPEX, OPEX, technology lifetime, and real WACC assumptions.
3. Builds national wholesale and retail price benchmarks using the bundled electricity-price, inflation, exchange-rate, and cost-recovery data.
4. Calculates the WACC reduction and grant-equivalent share of CAPEX required for a project to meet a benchmark price. Solar and wind benchmarks apply capture-rate assumptions.
5. Estimates technical potential using capacity-factor cutoffs, grid-cell area, land-use factors, and technology-specific power densities.
6. Estimates lifetime avoided emissions using IFI operating/build margins and an additionality factor derived from planned fossil and renewable capacity.
7. Reports grant support per tonne of avoided CO2 and compares technologies cell by cell.

Core economic assumptions are stored in [`DATA/economic_parameters.csv`](DATA/economic_parameters.csv). Additional constants, including capture rates, hurdle rate, capacity-factor cutoffs, power densities, benchmark scenarios, and emissions-factor weights, are currently defined in the Python source.

## Prerequisites

- Python 3
- A Python environment capable of installing geospatial packages (Cartopy, GeoPandas, Rasterio/rioxarray, Shapely, and Regionmask)
- The external data files listed below
- Enough memory and storage for global gridded NetCDF and raster processing

There is currently no dependency manifest or pinned environment. To create a local environment and install the packages imported by the source:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas xarray scipy matplotlib seaborn \
  geopandas rioxarray rasterio geopy shapely regionmask cartopy \
  openpyxl netCDF4
```

On platforms without binary wheels, the geospatial packages may also require system libraries such as GDAL, GEOS, and PROJ.

## Data inputs

The [`DATA/`](DATA/) directory contains country mappings, economic parameters, WACC estimates, land-use factors, renewable cost assumptions, electricity-price series, IMF inflation and exchange-rate series, GDP data, IFI grid factors, UNFCCC mappings, cost-recovery data, and the `Renewable_Inputs.xlsx` workbook used for hydro, geothermal, and bioenergy assumptions.

The primary input groups expected by `LCOEEstimator` are:

| Input group | Purpose | Typical format |
|---|---|---|
| Solar and wind resource | Mean capacity factors and gridded renewable output | NetCDF |
| Country and physical geography | Land/sea country IDs, land cover, bathymetry, and distance to shore | NetCDF |
| Technology costs | Country CAPEX, OPEX, lifetime, and firm-renewable assumptions | CSV/XLSX |
| Finance | Technology-specific country WACC estimates | CSV |
| Electricity prices | Retail/wholesale prices, cost recovery, inflation, and exchange rates | CSV |
| Other renewable potential | Hydro assets, geothermal placements, and biomass residues | Shapefile/CSV/GeoTIFF |
| Emissions and additionality | Ember generation/emissions, IFI grid factors, and planned power capacity | CSV |

### Required external inputs

The checked-in workflow references the following files that are **not present in this repository**:

- `DATA/SOLAR_CF_MEAN.2022.nc`
- `DATA/WIND_CF_MEAN.2022.nc`
- `DATA/country_grids.nc`
- `DATA/GlobalLandCover.nc`
- `DATA/ETOPO_bathymetry.nc`
- `DATA/distance2shore.nc`
- `DATA/CountryElecPrices.csv`
- `DATA/CountryCoding.csv`
- `DATA/Ember Yearly Data 2026.csv`
- `DATA/Global-Integrated-Power-September-2026.csv`
- `DATA/IRENA_STATS.csv`
- `DATA/Geothermal_allPlacements.csv`
- `DATA/Biomass_crop_residue_ton.tif`
- the complete `DATA/GDW_barriers_v1_0.*` and `DATA/GDW_reservoirs_v1_0.*` shapefile sets
- `bubble_plot.py`, which must provide `BubbleRiskPlot`

Some source methods also refer to `countries_gridded_0.1deg_v0.1.nc`, although that file is not used by the default pipeline. Input schemas are implicit in the column and variable names accessed by the source; there is no separate schema specification.

## Running the model

The executable workflow is at the bottom of [`lcoe_model.py`](lcoe_model.py). Before running it:

1. Place the external inputs in `DATA/`, preserving the filenames above, and provide `bubble_plot.py`.
2. Change `data_path` and `output_folder` near the end of `lcoe_model.py` to paths on your machine. Both paths should include a trailing path separator because the source concatenates paths as strings.
3. Create the results directories:

   ```bash
   mkdir -p RESULTS OUTPUT
   ```

4. If using the repository-local output directory, set:

   ```python
   data_path = "./DATA/"
   output_folder = "./OUTPUT/"
   ```

5. Run from the repository root:

   ```bash
   python lcoe_model.py
   ```

The script first runs `LCOEEstimator.concessional_model_pipeline()`, writes NetCDF results, and then starts an interactive plotting workflow. Respond with uppercase `Y` to generate an optional plot or `N` to skip it.

For programmatic use, the main classes are:

- `LCOEEstimator` in `lcoe_model.py`: input preparation, electricity-price benchmarks, LCOE, required WACC/grant support, and technology comparison
- `AbatementEstimator` in `abatement_model.py`: technical potential, IFI emissions factors, additionality, avoided emissions, and abatement cost inputs
- `Global_Data` in `geodata_v3.py`: country, offshore, bathymetry, and distance-to-shore alignment
- `PostProcessor` in `postprocessor.py`: CSV summaries, maps, supply curves, and marginal abatement cost plots

Importing `lcoe_model.py` currently executes the full workflow because the script has no `if __name__ == "__main__"` guard.

## Outputs

The core pipeline writes these gridded NetCDF files to `RESULTS/`:

```text
CHECKFINALFullResults_Geothermal.nc
CHECKFINALFullResults_Hydro.nc
CHECKFINALFullResults_Bioenergy.nc
CHECKFINALFullResults_Wind.nc
CHECKFINALFullResults_Solar.nc
CHECKFINALFullResults_Full_Cheapest.nc
```

Depending on the technology and benchmark, result variables include capacity factor, annual electricity production, technical potential, CAPEX and OPEX, estimated WACC, calculated and hurdle-rate LCOE, electricity-price benchmarks, required benchmark WACC, grant equivalent, avoided emissions, additionality, and abatement cost. LCOE is reported in USD/MWh; WACC fields are generally percentages; grant equivalents are stored both as a fraction of CAPEX and as a currency amount; abatement costs are in USD/tCO2.

The plotting workflow also writes `WIND_RESULTS_SUMMARY.csv` and `SOLAR_RESULTS_SUMMARY.csv` in the repository root, an electricity-price assumptions CSV in the configured output folder, a GEM/IFI additionality log in the root, and selected PNG maps and curves in the configured output folder.

## Repository structure

```text
.
├── DATA/                 # Bundled tabular and workbook inputs
├── abatement_model.py    # Emissions, additionality, potential, and abatement calculations
├── geodata_v3.py         # Gridded geography alignment and masks
├── lcoe_model.py         # Main estimator and executable pipeline
└── postprocessor.py      # Summaries, maps, and curve generation
```

## Limitations

- The repository cannot run end to end as checked in because required data and `bubble_plot.py` are absent.
- Paths and the workflow configuration are hard-coded in `lcoe_model.py`; there is no CLI or configuration file.
- Dependencies and Python versions are not pinned, and there is no automated test suite.
- The full run is interactive, and plotting assumes a working Matplotlib/Cartopy environment.
- The visualisation pipeline references `future_abatement_cost`, but the current main pipeline does not create that variable.
- Warnings are globally suppressed in the main model and postprocessor, which can hide numerical or data-quality warnings.
- Several assumptions are embedded directly in source code, and missing national observations may be filled with regional or other fallback values.
- Outputs are research-model estimates and depend on the supplied source data and assumptions; they should not be treated as investment advice.
- No license file is included, so no reuse terms are stated by this repository.

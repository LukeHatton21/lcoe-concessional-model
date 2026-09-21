import numpy as np
import xarray as xr
import pandas as pd
from scipy import interpolate



class AbatementEstimator:
    def __init__(self, wind_results, solar_results, carbon_intensity, geodata_class, postprocessor, ember_data, IFI_data, GEM_data):
        """ Initialises the AbatementEstimator class, which is used for calculating the lifetime abatement
       
        Inputs:
        wind_results: Wind processed results with LCOE
        solar_results: Solar processed results with LCOE
        carbon_intensity: Average grid carbon intensity data
        IFI_data: Reported data from IFIs on grid factors
        geodata_class: Class which includes geodata
        postprocessor: Class which includes postprocessing
        ember_data: Dataframe with electricity production data from Ember
        
        """
        
        # Read in results
        self.wind_results = wind_results
        self.solar_results = solar_results
        
        # Read in input data
        self.carbon_intensity = pd.read_csv(carbon_intensity)
        self.lifetime = 20
        self.year = 2023
        self.ember_data = ember_data
        self.IFI_data = pd.read_csv(IFI_data)
        self.gem_data = pd.read_csv(GEM_data)

        # Read in other classes
        self.geodata_class = geodata_class
        self.postprocessor = postprocessor
        
    def calculate_lifetime_abatement(self, CI_data, capacity_data, country_emissions):

        # Perform the calculation (converting from gCO2/kWh -> tCO2 / kWh requires 1e+6)
        tCO2 = CI_data / 1e+6 * capacity_data['technical_potential'] * self.lifetime * 1
        tCO2_MW = CI_data / 1e+6 * capacity_data['electricity_production'] * self.lifetime * 1
        capacity_data["abatement"] = tCO2
        capacity_data["abatement_MW"] = tCO2_MW

        # Store inputs
        capacity_data["grid_intensity"] = CI_data
        capacity_data["emissions"] = country_emissions

        return capacity_data

    def calculate_IFI_abatement(self, IFI_data, results, country_emissions, technology_type=None, additionality=None):

        # Perform the calculation (converting from gCO2/kWh -> tCO2 / kWh requires 1e+6)
        if additionality is not None:
            suffix = "_AF"
        else:
            suffix = ""
        if technology_type == "VRE":
            IFI_data["IFI_Emissions_Factor"] = IFI_data["IFI_Emissions_Factor_VRE"]
            IFI_data["IFI_Emissions_Factor_AF"] = IFI_data["IFI_Emissions_Factor_VRE_AF"]
            results["Operating_Margin"] = IFI_data["VRE_Energy"]
        else:
            IFI_data["IFI_Emissions_Factor"] = IFI_data["IFI_Emissions_Factor_Firm"]
            IFI_data["IFI_Emissions_Factor_AF"] = IFI_data["IFI_Emissions_Factor_Firm_AF"]
            results["Operating_Margin"] = IFI_data["Firm_Energy"]
        tCO2_MW = IFI_data["IFI_Emissions_Factor"] / 1e+6 * results['electricity_production'] * results["Lifetime"]
        tCO2_CI = IFI_data["Electricity_CI"] / 1e+6 * results['electricity_production'] * results["Lifetime"]

        results["abatement_MW" + suffix] = tCO2_MW
        results["abatement_CI" + suffix] = tCO2_CI
        results["IFI_Emissions_Factor"] = IFI_data["IFI_Emissions_Factor"]
        results["IFI_Emissions_Factor_AF"] = IFI_data["IFI_Emissions_Factor_AF"]
        results["Additionality_Factor"] = IFI_data["Additionality_Factor"]
        results["emissions"] = country_emissions
        results["grid_intensity"] = IFI_data["Electricity_CI"]

        return results

    def calculate_future_abatement(self, CI_data, capacity_data, country_emissions):

        # Extract current national grid electricity production
        electricity_production = self.extract_generation(self.ember_data)
        electricity_production = electricity_production["Generation"]

        # Calculate future expansion
        africa_years = [2024, 2035, 2050]
        africa_generation = [753, 1196, 2195]
        interpol_function = interpolate.interp1d(africa_years, africa_generation)

        # Calculate interpolated values
        target_years = np.arange(2025, 2025+self.lifetime, 1)
        yearly_increase = interpol_function(target_years) / africa_generation[0] - 1
        yearly_increase = yearly_increase.mean()

        # Calculate future grid carbon intensity
        CI_nat_gas = 0.49 * 1000
        factor = 0.5
        yearly_intensity = (CI_data * electricity_production + factor*(electricity_production*yearly_increase)*CI_nat_gas)/(electricity_production*yearly_increase)

        # Get list of African country indexes
        african_countries = self.postprocessor.country_mapping.loc[self.postprocessor.country_mapping["region"]=="Africa"]["index"].to_list()
        african_countries.remove(248)


        # Perform the calculation (converting from gCO2/kWh -> tCO2 / kWh requires 1e+6)
        tCO2 = capacity_data['technical_potential'] * yearly_intensity /1e+6 * self.lifetime
        tCO2_MW = capacity_data['electricity_production'] * yearly_intensity /1e+6 * self.lifetime
        capacity_data["future_abatement"] = xr.where(capacity_data["Country"].isin(african_countries),tCO2, capacity_data["abatement"])
        capacity_data["future_abatement_MW"] = xr.where(capacity_data["Country"].isin(african_countries), tCO2_MW, capacity_data["abatement_MW"])

        # Store inputs
        capacity_data["grid_intensity"] = CI_data
        capacity_data["future_grid_intensity"] = xr.where(capacity_data["Country"].isin(african_countries), yearly_intensity, CI_data)
        capacity_data["emissions"] = country_emissions


        return capacity_data



    def get_supply_curves_v3(self, data, technology, LCOE_cutoff=None):

        # Extract postprocessor
        postprocessor = self.postprocessor
        
        # Extract required parameters
        annual_production = data['electricity_production']
        latitudes = annual_production.latitude.values
        longitudes = annual_production.longitude.values

        # Get area and utilisations
        grid_areas = postprocessor.get_areas(annual_production)
        utilisation_factors = postprocessor.get_utilisations(annual_production, technology)

        # Set out constants
        if technology == "Onshore Wind":
            power_density = 6520 # kW/km2
            cutoff = 0.18
        elif technology == "Offshore Wind":
            power_density = 4000 # kW/km2
            cutoff = 0.18
        elif technology == "Solar":
            power_density = 32950  # kW/km2
            cutoff = 0.1
        installed_capacity = 1000

        # Apply cut off factors
        utilisation_factors = xr.where(data['CF']<cutoff, 0, utilisation_factors)
        if LCOE_cutoff is not None:
            data['Calculated_LCOE'] = xr.where(data['CF']<cutoff, np.nan, data['Calculated_LCOE'])
            data['Uniform_LCOE'] = xr.where(data['CF']<cutoff, np.nan, data['Uniform_LCOE'])

        # Scale annual electricity production by power density
        max_installed_capacity = power_density * grid_areas['area'] * utilisation_factors
        ratios = max_installed_capacity / installed_capacity
        technical_potential = annual_production * ratios

        # Include additional data into the dataset
        data['technical_potential'] = technical_potential
        data['capacity_GW'] = max_installed_capacity / 1e+06 # convert from kW to GW 
        if technology == "Offshore Wind":
            data['Country'] = postprocessor.country_grids['sea']
        else:
            data['Country'] = postprocessor.country_grids['land']

        return data
    
    def fill_carbon_intensity(self, ci_data_mapping):

        # Find maximums and extract
        region_max = ci_data_mapping.groupby("sub-region")["Value"].agg("mean").reset_index().rename(
            columns={"Value": "Value_mean"})

        # Merge back onto original
        ci_data_mapping = pd.merge(ci_data_mapping, region_max, how="left",
                                 on=["sub-region"])

        # Fill na
        ci_data_mapping.loc[ci_data_mapping["Value"].isnull(), "Value"] = ci_data_mapping[
            "Value_mean"]

        return ci_data_mapping

    def extract_carbon_intensity(self, ember_data):

        # Extract carbon intensity data
        ci_data = ember_data.loc[(ember_data["Unit"] == "gCO2/kWh") & (ember_data["Year"] == self.year)]

        # Get columns required
        ci_data_subset = ci_data[["Country code", "Value"]]

        # Merge onto country mapping
        ci_data_mapping = pd.merge(self.postprocessor.country_mapping, ci_data_subset, how="left", on="Country code")

        # Fill in data gaps
        ci_data_mapping = self.fill_carbon_intensity(ci_data_mapping)

        # Merge onto country grids
        country_grids = self.postprocessor.country_grids.rename({"land":"index"})
        country_df = country_grids.to_dataframe().reset_index()
        country_df = country_df.merge(ci_data_mapping.rename(columns={"Value":"Electricity_CI"}), how="left", on="index")
        country_ds = country_df.drop_duplicates(subset=["latitude", "longitude"]).set_index(["latitude", "longitude"]).to_xarray()
        country_ds = country_ds.assign_coords({"latitude":country_ds.latitude, "longitude": country_ds.longitude})

        return country_ds

    def extract_emissions(self, ember_data):

        # Extract carbon intensity data
        ci_data = ember_data.loc[(ember_data["Variable"] == "Total emissions") & (ember_data["Year"] == self.year)]

        # Get columns required
        ci_data_subset = ci_data[["Country code", "Value"]]

        # Merge onto country mapping
        ci_data_mapping = pd.merge(self.postprocessor.country_mapping, ci_data_subset, how="left", on="Country code")

        # Merge onto country grids
        country_grids = self.postprocessor.country_grids.rename({"land":"index"})
        country_df = country_grids.to_dataframe().reset_index()
        country_df = country_df.merge(ci_data_mapping.rename(columns={"Value":"emissions"}), how="left", on="index")
        country_ds = country_df.drop_duplicates(subset=["latitude", "longitude"]).set_index(["latitude", "longitude"]).to_xarray()
        country_ds = country_ds.assign_coords({"latitude":country_ds.latitude, "longitude": country_ds.longitude})

        return country_ds

    def extract_ci(self):

        def gap_fill_ci(data, ember_data, year):
            generation_subset = ember_data.loc[(ember_data["Unit"] == "gCO2/kWh") & (ember_data["Year"] == year)]
            previous_data = generation_subset[["Country code", "Value"]]

            # Merge
            previous_data = previous_data.rename(columns={"Value": "Current_Value"})
            data = data.merge(previous_data, how="left", on=["Country code"])

            # Fill with previous value
            data['Current_Value'] = data['Current_Value'].fillna(data['Value'])

            # Drop previous value column
            data = data.drop(columns="Value")
            data = data.rename(columns={"Current_Value": "Value"})
            data["Year"] = year

            return data

        # Extract carbon intensity data
        ember_data = self.ember_data
        ci_data = ember_data.loc[(ember_data["Unit"] == "gCO2/kWh") & (ember_data["Year"] == 2021)]
        ci_data = gap_fill_ci(ci_data, ember_data, 2022)
        ci_data = gap_fill_ci(ci_data, ember_data, 2023)
        ci_data = gap_fill_ci(ci_data, ember_data, 2024)
        ci_data = gap_fill_ci(ci_data, ember_data, 2025)

        extracted_data = ci_data[["Country code", "Value"]].dropna(subset="Country code").rename(columns={"Value":"Electricity_CI"})

        return extracted_data


    def extract_generation(self, ember_data):

        # Extract carbon intensity data
        ci_data = ember_data.loc[(ember_data["Variable"] == "Total Generation") & (ember_data["Year"] == 2022)]

        # Get columns required
        ci_data_subset = ci_data[["Country code", "Value"]]

        # Merge onto country mapping
        ci_data_mapping = pd.merge(self.postprocessor.country_mapping, ci_data_subset, how="left", on="Country code")

        # Merge onto country grids
        country_grids = self.postprocessor.country_grids.rename({"land":"index"})
        country_df = country_grids.to_dataframe().reset_index()
        country_df = country_df.merge(ci_data_mapping.rename(columns={"Value":"Generation"}), how="left", on="index")
        country_ds = country_df.drop_duplicates(subset=["latitude", "longitude"]).set_index(["latitude", "longitude"]).to_xarray()
        country_ds = country_ds.assign_coords({"latitude":country_ds.latitude, "longitude": country_ds.longitude})

        return country_ds

    def extract_IFI(self):

        # Extract carbon intensity data
        ifi_data = self.IFI_data[["Country code", "Firm_Energy", "VRE_Energy", "Build_Margin"]]

        # Extract data from GEM project
        gem = self.gem_data[["ISO", "Type", "Capacity (MW)"]]
        gem["Capacity (MW)"] = gem["Capacity (MW)"].astype(float)
        grouped_gem = gem .groupby(["ISO", "Type"])["Capacity (MW)"].agg("sum").reset_index()

        # Extract renewables planned capacity
        renewables_gem = grouped_gem.loc[grouped_gem["Type"].isin(["solar", "wind"])].rename(columns={"Capacity (MW)": "Renewable_Capacity"})
        renewables_gem = renewables_gem[["ISO", "Renewable_Capacity"]].groupby("ISO").agg("sum").reset_index()
        oil_gas_gem = grouped_gem.loc[grouped_gem["Type"] == "oil/gas"].rename(columns={"Capacity (MW)": "Fossil_Capacity"})

        # Merge together
        merged_gem = pd.merge(renewables_gem, oil_gas_gem, how="outer", on="ISO")
        merged_gem["Fossil_Renewable"] = merged_gem["Fossil_Capacity"] / merged_gem["Renewable_Capacity"]
        merged_gem["Country code"] = merged_gem["ISO"]

        # Merge onto IFI data
        ifi_data = ifi_data.merge(merged_gem[["Country code", "Fossil_Renewable"]], how="left", on="Country code")
        ifi_data = ifi_data.merge(merged_gem[["Country code", "Fossil_Capacity"]], how="left", on="Country code")
        ifi_data["Additionality_Factor"] = ifi_data["Fossil_Renewable"].clip(upper=1, lower=0.1).fillna(1)

        # Extract ember data for countries without data
        ember_CI = self.extract_ci()
        ifi_data = ifi_data.merge(ember_CI, how="left", on="Country code")

        # Replace VRE_Energy and Build Margin with electricity carbon intensity if below
        ifi_data.loc[ifi_data["VRE_Energy"] < 10, "VRE_Energy"] = ifi_data["Electricity_CI"]
        ifi_data.loc[ifi_data["Build_Margin"] < 10, "Build_Margin"] = np.nanmedian(ifi_data["Build_Margin"])

        # Calculate emissions factor
        vre_factor = 0.75
        firm_factor = 0.33

        # Apply without additionality
        ifi_data["IFI_Emissions_Factor_VRE"] =  (ifi_data["VRE_Energy"] * (vre_factor) + (1 - vre_factor) * ifi_data["Build_Margin"])
        ifi_data["IFI_Emissions_Factor_Firm"] = (
                    ifi_data["Firm_Energy"] * (firm_factor) + (1 - firm_factor)  * ifi_data["Build_Margin"])

        # Calculate with additionality
        ifi_data["IFI_Emissions_Factor_VRE_AF"] = ifi_data["Additionality_Factor"] * (
                    ifi_data["VRE_Energy"] * (vre_factor) + (1 - vre_factor) * ifi_data["Build_Margin"])
        ifi_data["IFI_Emissions_Factor_Firm_AF"] = ifi_data["Additionality_Factor"] * (
                ifi_data["Firm_Energy"] * (firm_factor) + (1 - firm_factor) * ifi_data["Build_Margin"])

        # Get columns required
        ifi_data_subset = ifi_data

        # Merge onto country mapping
        ifi_data_mapping = pd.merge(self.postprocessor.country_mapping, ifi_data_subset, how="left", on="Country code")

        # Merge onto country grids
        country_grids = self.postprocessor.country_grids.rename({"land":"index"})
        country_df = country_grids.to_dataframe().reset_index()
        country_df = country_df.merge(ifi_data_mapping, how="left", on="index")
        country_ds = country_df.drop_duplicates(subset=["latitude", "longitude"]).set_index(["latitude", "longitude"]).to_xarray()
        country_ds = country_ds.assign_coords({"latitude":country_ds.latitude, "longitude": country_ds.longitude})

        return country_ds
    
    
    def calculate_abatement_potential(self, results, technology_type=None):

        # 1. Calculate abatement factors at each location
        country_IFI = self.extract_IFI()

        # 2. Calculate total power sector emissions
        country_emissions = self.extract_emissions(self.carbon_intensity)
        country_emissions = country_emissions["emissions"]

        # 3. Calculate abatement costs
        results_with_abatement = self.calculate_IFI_abatement(country_IFI, results, country_emissions, technology_type=technology_type)

        # 4. Calculate with additionality abatement
        results_with_abatement = self.calculate_IFI_abatement(country_IFI, results_with_abatement, country_emissions, technology_type=technology_type, additionality="Future")

        return results_with_abatement
    
    
    def abatement_model_pipeline(self):
        
        # Calculate solar abatement
        wind_results = self.get_supply_curves_v3(self.wind_results, "Onshore Wind")
        #wind_results = self.calculate_abatement_potential(wind_results, technology_type="VRE")
        
        # Calculate wind abatement
        solar_results = self.get_supply_curves_v3(self.solar_results, "Solar")
        #solar_results = self.calculate_abatement_potential(solar_results, technology_type="VRE")
        
        return solar_results, wind_results
    
    
    
    
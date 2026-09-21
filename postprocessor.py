import numpy as np
import seaborn as sns
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from matplotlib.patches import (Circle, Patch)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox,
                                  DrawingArea, TextArea, VPacker)
import warnings
warnings.filterwarnings("ignore")
from matplotlib.colors import BoundaryNorm
from pylab import *
from bubble_plot import BubbleRiskPlot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class PostProcessor:
    def __init__(self, output_folder, solar_results, wind_results, offshore_results, GDP_capita, land_cover, land_mapping, country_grids, country_mapping, solar_cf, wind_cf, country_wacc_mapping):
        """ Initialises the PostProcessor class, which is used for geospatial mapping and postprocessing
       
        Inputs:
        Output_Folder - folder for printing outputs
        Wind_results - Dataset containing LCOE, electricity production and WACC values for Wind
        Solar_results - Dataset containing LCOE, electricity production and WACC values for Solar
        GDP_capita - CSV containing GDP per capita for all countries
        Land_cover - netcdf file with codings for land cover categories
        Land_mapping - CSV file containing land utilisation rates for land cover categories
        TIAM_regions - CSV file containing a mapping of the TIAM regions
        Country_mapping - CSV file with a mapping of country numbers to country codes
        Solar_CF - netcdf file containing average capacity factor
        Wind_CF - netcdf file containing average capacity factor
        country_wacc_mapping - Mapping of country waccs for solar PV, onshore wind and offshore wind
        
        
        """
        
        # Read in the output folder
        self.output_folder = output_folder
        
        # Read in the solar and wind parameters
        self.solar_results = solar_results
        self.wind_results = wind_results
        self.offshore_results = offshore_results
        self.solar_cf = solar_cf
        self.wind_cf = wind_cf
        
        # Collate wind results
        self.collated_wind_results = xr.merge([self.wind_results.drop(labels="capacity_GW", errors="ignore"), self.offshore_results], join="inner", compat="no_conflicts")
        
        # Read in the input parameters
        self.GDP_capita = GDP_capita
        self.land_cover = land_cover
        self.land_mapping = land_mapping
        self.country_grids = country_grids
        self.country_mapping = country_mapping
        
        # Perform merges for GDP and for country grids
        self.GDP_country_mapping = pd.merge(self.country_mapping, self.GDP_capita, on="Country code", how="left")
        self.country_wacc_mapping = country_wacc_mapping

        # IRENA stats
        self.IRENA_STATS = pd.read_csv("./DATA/IRENA_STATS.csv")
        self.unfccc_countries = pd.read_csv("./DATA/unfccc.csv")

                 
                 
                 
    def get_utilisations(self, annual_production, technology):

        latitudes = annual_production.latitude.values
        longitudes = annual_production.longitude.values
        global_cover = self.land_cover.reindex_like(annual_production, method="nearest")
        mapping = self.land_mapping

        utilisation = xr.zeros_like(global_cover['cover'])
        for i in np.arange(0, 21, 1):
            # Use xarray's where and isin functions to map land use categories to values
            if technology == "Solar":
                utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['PV LU'].iloc[i], utilisation)
            elif technology =="Onshore Wind":
                utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['Wind LU'].iloc[i], utilisation)
            elif technology == "Offshore Wind":
                utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], 1, utilisation)

        return utilisation    


    def get_areas(self, annual_production):

        latitudes = annual_production.latitude.values
        longitudes = annual_production.longitude.values

        # Add an extra value to latitude and longitude coordinates
        latitudes_extended = np.append(latitudes, latitudes[-1] + np.diff(latitudes)[-1])
        longitudes_extended = np.append(longitudes, longitudes[-1] + np.diff(longitudes)[-1])

        # Calculate the differences between consecutive latitude and longitude points
        dlat_extended = np.diff(latitudes_extended)
        dlon_extended = np.diff(longitudes_extended)

        # Calculate the Earth's radius in kms
        radius = 6371

        # Compute the mean latitude value for each grid cell
        mean_latitudes_extended = (latitudes_extended[:-1] + latitudes_extended[1:]) / 2
        mean_latitudes_2d = mean_latitudes_extended[:, np.newaxis]

        # Convert the latitude differences and longitude differences from degrees to radians
        dlat_rad_extended = np.radians(dlat_extended)
        dlon_rad_extended = np.radians(dlon_extended)

        # Compute the area of each grid cell using the Haversine formula
        areas_extended = np.outer(dlat_rad_extended, dlon_rad_extended) * (radius ** 2) * np.cos(np.radians(mean_latitudes_2d))

        # Create a dataset to store results
        area_dataset = xr.Dataset()
        area_dataset['latitude'] = latitudes
        area_dataset['longitude'] = longitudes
        area_dataset['area'] = (['latitude', 'longitude'], areas_extended, {'latitude': latitudes, 'longitude': longitudes})

        return area_dataset



    def get_supply_curves_v2(self, data, technology, offshore=None, LCOE_cutoff=None):

        # Extract required parameters
        annual_production = data['electricity_production']
        latitudes = annual_production.latitude.values
        longitudes = annual_production.longitude.values

        # Get area and utilisations
        grid_areas = self.get_areas(annual_production)
        utilisation_factors = self.get_utilisations(annual_production, technology)

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
        data['capacity_GW'] = max_installed_capacity * utilisation_factors / 1e+06 # convert from kW to GW 
        if technology == "Offshore Wind":
            data['Country'] = self.country_grids['sea']
        else:
            data['Country'] = self.country_grids['land']

        return data

    def produce_wacc_potential_curve_v2(self, supply_ds, filename=None, graphmarking=None, title=None, xlim=None, uniform_value=None, technology=None, region_code=None, subnational=None):

        # Convert the dataset into a dataframe
        supply_df = supply_ds.to_dataframe()

        # Remove locations not evaluated
        supply_df = supply_df.dropna(axis=0, subset=["Calculated_LCOE", "Country"], how="all")

        # Create two copies
        uniform_df = supply_ds.to_dataframe().dropna(axis=0, subset=["Uniform_LCOE", "Country"], how="all")
        wacc_df = supply_ds.to_dataframe().dropna(axis=0, subset=["Calculated_LCOE", "Country"], how="all")
        if subnational is not None:
            subnational_df = supply_ds.to_dataframe().dropna(axis=0, subset=["Subnational_LCOE", "Country"], how="all")
        
        # Convert units to TWh
        supply_df['technical_potential'] = supply_df['technical_potential'] / 1e+09
        uniform_df['technical_potential'] = uniform_df['technical_potential'] / 1e+09
        wacc_df['technical_potential'] = wacc_df['technical_potential'] / 1e+09
        if subnational is not None:
            subnational_df['technical_potential'] = subnational_df['technical_potential'] / 1e+09

        # For the Country WACC case, sort values and calculate cumulative sum
        supply_df = supply_df.round({'Estimated_WACC': 3})
        supply_df = supply_df.sort_values(by=['Estimated_WACC'], ascending=True)
        supply_df['cumulative_potential'] = supply_df['technical_potential'].cumsum()

        # For the WACC case, sort values and calculate cumulative sum
        wacc_sorted_df = wacc_df.sort_values(by=['Calculated_LCOE'], ascending=True)
        wacc_sorted_df['cumulative_wacc'] = wacc_sorted_df['technical_potential'].cumsum()

        # For the Uniform WACC case, sort values and calculate cumulative sum
        uniform_sorted_df = uniform_df.sort_values(by=['Uniform_LCOE'], ascending=True)
        uniform_sorted_df['cumulative_uniform'] = uniform_sorted_df['technical_potential'].cumsum()
        uniform_sorted_df = uniform_sorted_df.drop(uniform_sorted_df[uniform_sorted_df['cumulative_uniform'] > np.nanmax( supply_df['cumulative_potential'])].index)
        
        if subnational is not None:
            # For the Uniform WACC case, sort values and calculate cumulative sum
            subnational_sorted_df = subnational_df.sort_values(by=['Subnational_LCOE'], ascending=True)
            subnational_sorted_df['cumulative_uniform'] = subnational_sorted_df['technical_potential'].cumsum()
            subnational_sorted_df = subnational_sorted_df.drop(subnational_sorted_df[subnational_sorted_df['cumulative_uniform'] > np.nanmax( supply_df['cumulative_potential'])].index)
        
        # Print maximums
        print(f"The maximum for supply_df is {np.nanmax(supply_df['cumulative_potential'])}, the maximum for the uniform case is {np.nanmax(wacc_sorted_df['cumulative_wacc'])} and the max for country specific is {np.nanmax(uniform_sorted_df['cumulative_uniform'])}")


        # Plot the results
        fig, ax = plt.subplots(figsize=(20, 8))
        color_labels = {}
        cmap = mpl.colormaps['gnuplot_r']
        norm = mpl.colors.Normalize(vmin=0, vmax=50000)  # Normalize to the range of GDP


        # Iterate through each data point and create a bar with the specified width
        for index, row in supply_df.iterrows():
            width = row['technical_potential']   # Bar width, in TWh
            height = row['Estimated_WACC']  # Bar height
            country = row['Country']
            cumulative_production = row['cumulative_potential'] # Cumulative production, in TWh

            # Get GDP per capita 
            if np.isnan(country):
                gdp_per_capita = np.nan
            else:
                gdp_per_capita = self.GDP_country_mapping.loc[self.GDP_country_mapping['index'] == country, '2024'].values[0]
            if np.isnan(gdp_per_capita) or gdp_per_capita is None or gdp_per_capita == "no data":
                color = "gray"
            else:
                color = cmap(norm(gdp_per_capita))

            # Plot a bar with the specified width, height, x-position, and color
            ax.bar(cumulative_production, height, width=-1 * width, align='edge', color=color)


        def thousands_format(x, pos):
            return f'{int(x):,}'

        # Set labels
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, 30)
        ax.set_ylabel('WACC (%)', fontsize=20)
        ax.set_xlabel('Annual Electricity Potential (TWh/year)', fontsize=25)
        ax.set_title(title, fontsize=30)
        ax.xaxis.set_major_formatter(FuncFormatter(thousands_format))

        # Set the size of x and y-axis tick labels
        ax.tick_params(axis='x', labelsize=20)  # Adjust the labelsize as needed
        ax.tick_params(axis='y', labelsize=20)  # Adjust the labelsize as needed

        # Add color bar
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=[0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 450000,50000], format=',', extend="max", anchor=(0.25, 0.5))
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(thousands_format))
        cbar.set_label('GDP per capita (USD, 2024)', fontsize=20)
        cbar.ax.tick_params(labelsize=15)


        # Plot second axis
        # Create a twin Y-axis on the same figure
        ax_twin = ax.twinx()

        # Plot lines on the twin Y-axis (this axis is independent of the main y-axis)
        ax_twin.plot(wacc_sorted_df['cumulative_wacc'], wacc_sorted_df['Calculated_LCOE'], color='blue', lw=2.5, label='LCOE under Country WACCs', linestyle="--")
        ax_twin.plot(uniform_sorted_df['cumulative_uniform'], uniform_sorted_df['Uniform_LCOE'], color='red', lw=2.5, label=f'LCOE under UNFCCC Annex II Average ({uniform_value:0.1f}%)', linestyle="--")
        if subnational is not None:
            ax_twin.plot(subnational_sorted_df['cumulative_uniform'], subnational_sorted_df['Subnational_LCOE'], color='black', lw=2.5, label=f'LCOE under subnational WACCs', linestyle="--")

        # Customize the second y-axis
        ax_twin.set_ylabel('Levelised Cost (USD/MWh)', fontsize=20)
        ax_twin.legend(loc='upper center', fontsize=20)
        ax_twin.tick_params(axis="y", labelsize=20)
        ax_twin.set_ylim(0, 300)

        if graphmarking is not None:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')
            
        if region_code is not None:
            ax.text(0.15, 0.9, technology + "\n" + region_code, transform=ax.transAxes, fontsize=20, fontweight='bold', ha="center", va="center")

        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches="tight")

        plt.show()

        return supply_df
    
    
    
    
    
    
    def plot_TIAM_region(self, region_code, solar_data, wind_data, regional_solar_wacc, regional_wind_wacc):
        
        # Get Solar and Wind Datasets with technical potential
        solar_ds = self.get_supply_curves_v2(solar_data,  "Solar")
        wind_ds = self.get_supply_curves_v2(wind_data, "Wind")        
        
        # Plot region values
        wind_df = self.produce_wacc_potential_curve_v2(wind_ds, uniform_value=regional_wind_wacc, region_code=region_code, technology="Onshore\nWind")
        solar_df = self.produce_wacc_potential_curve_v2(solar_ds, uniform_value=regional_solar_wacc, region_code=region_code, technology="Solar")

    def plot_data_broken(values, latitudes, longitudes, anchor=None, filename=None, increment=None,
                                       title=None, tick_values=None, cmap=None, extend_set=None, graphmarking=None,
                                       special_value=None, hatch_label=None, hatch_label_2=None, special_value_2=None, center_norm=None):
        cmap = plt.get_cmap(cmap, len(tick_values) + 1)
    
        # create the heatmap using pcolormesh
        if anchor is None:
            anchor = 0.355
        fig = plt.figure(figsize=(30, 15), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        if center_norm is None:
            heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.Normalize(vmin=tick_values[0], vmax=tick_values[-1]), transform=ccrs.PlateCarree(), cmap=cmap)
        else:
            heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.SymLogNorm(vmin = tick_values[0], vmax=tick_values[-1], linscale
    =1, linthresh=1), transform=ccrs.PlateCarree(), cmap=cmap)

        # Check if there is a need for extension
        values_min = np.nanmin(values)
        values_max = np.nanmax(values)
        if values_min < tick_values[0]:
            extend = "min"
        elif values_max > tick_values[-1]:
            extend = "max"
        elif (values_max > tick_values[-1]) & (values_min < tick_values[0]):
            extend = "both"
        else:
            extend="neither"
        if extend_set is not None:
            extend = extend_set
        

        axins = inset_axes(
        ax,
        width="1.5%",  
        height="80%",  
        loc="lower left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
        cb = fig.colorbar(heatmap, cax=axins, shrink=0.5, ticks=tick_values, format="%0.0f", extend=extend, anchor=(0, anchor))


        cb.ax.tick_params(labelsize=20)
        if title is not None:
            cb.ax.set_title(title, fontsize=25)

        # Add the special shading
        if special_value is not None:
            special_overlay = np.where(values == special_value, 1, np.nan)
            hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['/'], colors="silver", linewidth=0.15, transform=ccrs.PlateCarree())

        if special_value_2 is not None:
            special_overlay = np.where(values == special_value_2, 1, np.nan)
            hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['\\'], colors="gold", linewidth=0.15, transform=ccrs.PlateCarree())

        # set the extent and aspect ratio of the plot
        ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
        aspect_ratio = (latitudes.max() - latitudes.min()) / (longitudes.max() - longitudes.min())
        ax.set_aspect(1)

        # add axis labels and a title
        ax.set_xlabel('Longitude', fontsize=30)
        ax.set_ylabel('Latitude', fontsize=30)
        borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none')
        ax.add_feature(borders, edgecolor='gray', linestyle=':')
        ax.coastlines()
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        ax.coastlines()
        if graphmarking is not None:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')

        hatch_patches=[]
        if special_value is not None and hatch_label is not None:
            hatch_patch_1 = Patch(facecolor='silver', edgecolor='black', hatch="/", label=hatch_label)
            hatch_patches.append(hatch_patch_1)

        if hatch_label_2 is not None:
            hatch_patch_2 = Patch(facecolor='gold', edgecolor='black', hatch="/", label=hatch_label_2)
            hatch_patches.append(hatch_patch_2)

        if hatch_patches:
            ax.legend(handles=hatch_patches, loc='lower left', fontsize=20)

        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches="tight")

        return 
    
    def get_wacc_values(self, estimated_waccs, country_mapping, technology): 

        data = country_mapping
        storage_df = xr.zeros_like(data['land'])
        if technology == "Offshore Wind":
            storage_df = xr.zeros_like(data['sea'])
        storage_df = xr.where(storage_df == 0, np.nan, np.nan)
        for i in np.arange(1, 251, 1):
            # Extract WACC
            if technology == "Solar":
                wacc = estimated_waccs[estimated_waccs['index'] == i]['solar_pv_wacc'].values[0] 
            elif technology == "Onshore Wind":
                wacc = estimated_waccs[estimated_waccs['index'] == i]['onshore_wacc'].values[0] 
            elif technology == "Offshore Wind":
                wacc = estimated_waccs[estimated_waccs['index'] == i]['offshore_wacc'].values[0] 

            # Apply mapping
            storage_df = xr.where(data['land'] == i, wacc, storage_df)

        # Extracted data
        extracted_data = storage_df

        return extracted_data
    
    def plot_wacc_values(self, geodata):
        
        # Reindex geodata
        country_geodata = geodata.reindex({"latitude":self.solar_results.latitude, "longitude":self.solar_results.longitude}, method="nearest")
        
        # Get solar WACCs
        solar_plot_waccs = self.get_wacc_values(self.country_wacc_mapping, country_geodata, "Solar")
        solar_plot_waccs = xr.where(np.isnan(self.solar_results['Calculated_LCOE']), np.nan, solar_plot_waccs)
        
        # Get onshore WACCs
        onshore_plot_waccs = self.get_wacc_values(self.country_wacc_mapping, country_geodata, "Onshore Wind")
        onshore_plot_waccs = xr.where(np.isnan(self.solar_results['Calculated_LCOE']), np.nan, onshore_plot_waccs)
        
        # Get offshore WACCs
        offshore_plot_waccs = self.get_wacc_values(self.country_wacc_mapping, country_geodata, "Offshore Wind")
        offshore_plot_waccs = xr.where(np.isnan(onshore_plot_waccs), np.nan, offshore_plot_waccs)
        
        # Get collation
        collated_wind_results = self.collated_wind_results
        solar_results = self.solar_results
        
        # Plot data
        #elf.plot_data_shading(solar_plot_waccs,solar_plot_waccs.latitude, solar_plot_waccs.longitude, tick_values = [0, 5, 10, 15, 20, 25], title="Estimated\nWACC \n (%, nom,\nafter tax)\n", cmap="YlOrRd", extend_set="neither", filename = self.output_folder + "Solar_WACC_2023", graphmarking="a")
        #elf.plot_data_shading(onshore_plot_waccs, onshore_plot_waccs.latitude, onshore_plot_waccs.longitude, tick_values = [0, 5, 10, 15, 20, 25], title="Estimated\nWACC\n (%, nom,\nafter tax)\n", cmap="YlGnBu", extend_set="neither", filename = self.output_folder + "Onshore_Wind_WACC_2023", graphmarking="b")
        #elf.plot_data_shading(offshore_plot_waccs, offshore_plot_waccs.latitude, offshore_plot_waccs.longitude, tick_values = [0, 5, 10, 15, 20, 25], title="Estimated\nWACC\n (%, nom,\nafter tax)\n", cmap="YlGnBu", extend_set="neither", filename = self.output_folder + "Offshore_Wind_WACC_2023", graphmarking="c")
        self.plot_data_shading(solar_results['Estimated_WACC'], solar_results.latitude, solar_results.longitude, tick_values = [0, 5, 10, 15, 20, 25], title="Estimated\nWACC\n (%, nom,\nafter tax)\n", cmap="YlOrRd", extend_set="neither", filename = self.output_folder + "Solar_WACC_2023", graphmarking="a")
        self.plot_data_shading(collated_wind_results['Estimated_WACC'], collated_wind_results.latitude, collated_wind_results.longitude, tick_values = [0, 5, 10, 15, 20, 25], title="Estimated\nWACC\n (%, nom,\nafter tax)\n", cmap="YlGnBu", extend_set="neither", filename = self.output_folder + "Wind_WACC_2023", graphmarking="b")
    
    
    
    def plot_supply_curve_global(self, solar_uniform, wind_uniform, offshore_uniform, subnational=None, gdp_shading=None):
        
        # Get Solar and Wind Datasets with technical potential
        solar_ds = self.get_supply_curves_v2(self.solar_results,  "Solar")
        wind_ds = self.get_supply_curves_v2(self.wind_results, "Onshore Wind")
        offshore_ds = self.get_supply_curves_v2(self.offshore_results, "Offshore Wind")
        
        # Plot solar and wind wacc potential curves
        solar_df = self.produce_potential_curve_v3(solar_ds, uniform_value=solar_uniform, technology="Solar", region_code="Global", graphmarking="a", filename=self.output_folder + "/Solar_Global", subnational=subnational, xlim=1e+06, gdp_shading=gdp_shading)
        wind_df = self.produce_potential_curve_v3(wind_ds, uniform_value=wind_uniform, technology="Onshore\nWind", region_code="Global", graphmarking="b",  filename=self.output_folder + "/Onshore_Wind_Global", subnational=subnational, xlim=1e+06, gdp_shading=gdp_shading)
        
        # Store dataframes
        self.solar_df = solar_df
        self.wind_df = wind_df

    
    def produce_scatter_plots(self, results, technology):

        unfccc = pd.read_csv("./DATA/unfccc.csv")
        unfcc_index = unfccc["index"].values.tolist()

        # Get LCOE and Abatement
        lcoe_values = results
        abatement = lcoe_values["abatement_MW"].values/1e+06
        lcoe = lcoe_values["Calculated_LCOE"].values



        # Get colors and legends
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', label='ADV',
                   markerfacecolor='green', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='China',
                   markerfacecolor='red', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='EMDE',
                   markerfacecolor='blue', markersize=10)
        ]
        color = self.country_grids["country"]
        color = xr.where(color.isin(unfcc_index), "green", xr.where(color==47, "red", "blue"))
        colors = color.values

        # Produce plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(abatement.flatten(), lcoe.flatten(), alpha=0.025, c=colors.flatten())
        ax.set_ylim([0,200])
        ax.set_ylabel("Levelised Cost (USD/MWh)")
        ax.set_xlabel("Lifetime abatement (MT CO2)")
        ax.set_title(technology + ": LCOE vs Lifetime Abatement")
        ax.legend(handles=legend_handles, loc="upper right", title="Country")
        plt.savefig(self.output_folder + technology + "_LCOE_Abatement.png")


    def produce_country_scatter(self, results, technology):

        def scatter_area_to_pixel_radius(area_points2, fig):
            """
            Convert scatter marker size (points^2) to pixel radius.
            """
            dpi = fig.dpi
            radius_points = np.sqrt(area_points2 / np.pi)
            radius_pixels = radius_points * dpi / 72
            return radius_pixels

        def draw_circles(ax, circle_radii, circle_text, padding=4):
            """
            Draw circles in an AnchoredOffsetbox without overlap.
            `circle_radii` must be in pixels (scatter marker radius).
            """
            # Compute vertical positions so circles do not touch
            y_positions = [0]  # bottom circle starts at 0
            for i in range(1, len(circle_radii)):
                prev_r = circle_radii[i - 1]
                curr_r = circle_radii[i]
                y_positions.append(
                    y_positions[-1] + prev_r + curr_r + padding
                )

            # Total height of drawing area
            height = y_positions[-1] + circle_radii[-1] + padding
            width = max(circle_radii) * 2 + padding

            area = DrawingArea(width=width, height=height)

            # Draw circles
            for r, y in zip(circle_radii, y_positions):
                area.add_artist(
                    Circle((width / 2, y), r, fc="tab:blue", alpha=0.7)
                )

            # Add the offset box into the axis
            box = AnchoredOffsetbox(
                child=area, loc="lower right", pad=0.2, frameon=False
            )
            ax.add_artist(box)

            # Optional: add text labels
            if circle_text:
                for label, y in zip(circle_text, y_positions):
                    ax.text(
                        0.98,  # x position (axes fraction)
                        ax.transAxes.inverted().transform((0, y))[1],
                        label,
                        fontsize=8,
                        ha="right",
                        va="center",
                        transform=ax.transAxes
                    )

        # Convert results to a dataframe
        results_df = results.to_dataframe()
        results_df = results_df.loc[~(results_df["Country"].isnull())]

        # Group by country and take the median result
        results_df = results_df.groupby("Country").agg("mean", numeric_only=True).reset_index()

        # Get relevant parameters
        median_lcoe = results_df["Calculated_LCOE"]
        wacc = results_df["Estimated_WACC"]
        grid_intensity = results_df["grid_intensity"]
        benchmark_price = results_df["Benchmark_Price"] *1000



        # Get country mapping and colours
        unfccc = pd.read_csv("./DATA/unfccc.csv")
        unfcc_index = unfccc["index"].values.tolist()
        results_df["Macroregion"]= results_df.apply(
            lambda row: "China" if row["Country"] == 47
            else ("ADV" if row["Country"] in unfcc_index
            else "EMDE"),
        axis=1)
        results_df = results_df.merge(self.country_wacc_mapping[["index", "Country code"]], how="left", left_on="Country", right_on="index")

        # Get capacity additions
        group_col = "Group Technology"
        if technology=="Solar":
            tech_flag = "Solar energy"
        elif technology=="Wind":
            tech_flag = "Wind energy"
        else:
            group_col = "RE or Non-RE"
            tech_flag = "Total Renewable"
        capacity_2023 = self.IRENA_STATS.loc[(self.IRENA_STATS["Year"]==2022) & (self.IRENA_STATS[group_col]==tech_flag)].rename(columns={"Electricity Installed Capacity (MW)":"2023"}).groupby(["ISO3 code", group_col]).agg("sum").reset_index()
        capacity_2024 = self.IRENA_STATS.loc[
            (self.IRENA_STATS["Year"] == 2023) & (self.IRENA_STATS[group_col]==tech_flag)].rename(columns={"Electricity Installed Capacity (MW)":"2024"}).groupby(
            ["ISO3 code", group_col]).agg("sum").reset_index()
        joint_capacity = pd.merge(capacity_2023, capacity_2024, how="left", on="ISO3 code")
        joint_capacity["Increase_2024"] = joint_capacity["2024"] - joint_capacity["2023"]
        results_df = pd.merge(results_df, joint_capacity[["ISO3 code", "Increase_2024"]], how="left", left_on="Country code", right_on="ISO3 code")

        # Calculate figures
        total_generation = results_df["Increase_2024"].sum()
        total_generation_no_china = total_generation - results_df.loc[results_df["Country code"]=="CHN"]["Increase_2024"].values[0]
        benchmark = 400
        total_generation_below = results_df.loc[results_df["grid_intensity"]<benchmark]["Increase_2024"].sum()

        # Count number of countries
        total_countries_above = len(results_df.loc[results_df["grid_intensity"]>400])
        total_countries_below = len(results_df.loc[results_df["grid_intensity"]<400])

        # Print results
        print(f"Based on generation data, {total_generation_below/total_generation*100}% (total) and {total_generation_below/total_generation_no_china*100}% (total without China) for {technology} was deployed in countries with grid intensity below {benchmark}gCO2/kWh")
        print(f"{total_countries_above} countries have grid carbon intensities above the benchmark of {benchmark}gCO2/kWh")

        # Produce plot
        colors = {"EMDE": "green", "China": "red", "ADV": "blue"}
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', label='ADV',
                   markerfacecolor='blue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='China',
                   markerfacecolor='red', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='EMDE',
                   markerfacecolor='green', markersize=10),
            Rectangle((0, 0), 1, 1, fc="grey", label="Range across\nOECD countries")
        ]


        fig, ax = plt.subplots()

        # Use seaborn scatterplot
        scatter = sns.scatterplot(
            x=grid_intensity,
            y=median_lcoe/benchmark_price*100,
            hue=results_df["Macroregion"],  # continuous variable -> color
            #style=results_df["Macroregion"],  # categorical variable -> marker style
            palette=colors,
            size=results_df["Increase_2024"]/100,  # bubble size
            sizes=(1, 10000),  # control scaling of size
            alpha=0.75,
            #palette="viridis",  # color palette for cf
            ax=ax
        )


        # Pick three meaningful reference bubble sizes
        ref_sizes = [10000, 1000, 100]  # same units as "Increase_2024"
        ref_labels = ["10 GW", "1 GW", "100 MW"]

        # Convert scatter size scale to actual marker radii
        # Seaborn sizes=(5, 5000) refers to marker *area*, so radius = sqrt(area / π)
        circle_areas = ref_sizes
        circle_radii = [scatter_area_to_pixel_radius(s, fig) for s in ref_sizes]

        # Coordinates in axis fraction (0–1 space)
        #draw_circles(ax, circle_radii, ref_labels)

        if technology == "Solar":
            graphmarking = "a"
        elif technology == "Wind":
            graphmarking = "b"
        else:
            graphmarking = "a"

        # Add colorbar for CF
        norm = plt.Normalize(wacc.min(), 20)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        ax.set_ylim([0, 100])
        ax.set_xlim([0, 1200])
        ax.axhspan(ymin=25, ymax=50, alpha=0.25, color="grey", label="Range across OECD countries")
        ax.set_ylabel("LCOE relative to electricity prices (% of price)")
        ax.set_xlabel("Grid Carbon Intensity (gCO2/kWh)")
        ax.legend(handles=legend_handles, loc="upper right", title="Country")
        ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=10, fontweight='bold')

        plt.savefig(self.output_folder + technology + "_LCOE_Abatement_Country.png")

        plt.show()


    def produce_country_scatter_v3(self, results, technology):


        # Convert results to a dataframe
        results_df = results.to_dataframe()
        results_df = results_df.loc[~(results_df["Country"].isnull())]

        # Group by country and take the median result
        results_df = results_df.groupby("Country").agg("mean", numeric_only=True).reset_index()

        # Get relevant parameters
        median_lcoe = results_df["Calculated_LCOE"]
        wacc = results_df["Estimated_WACC"]
        grid_intensity = results_df["grid_intensity"]
        benchmark_price = results_df["Benchmark_Price"] *1000
        results_df["Benchmark_Ratio"] = median_lcoe / benchmark_price * 100



        # Get country mapping and colours
        unfccc = pd.read_csv("./DATA/unfccc.csv")
        unfcc_index = unfccc["index"].values.tolist()
        results_df["Macroregion"]= results_df.apply(
            lambda row: "China" if row["Country"] == 47
            else ("ADV" if row["Country"] in unfcc_index
            else "EMDE"),
        axis=1)
        results_df = results_df.merge(self.country_wacc_mapping[["index", "Country code"]], how="left", left_on="Country", right_on="index")

        # Get capacity additions
        group_col = "Group Technology"
        if technology=="Solar":
            tech_flag = "Solar energy"
        elif technology=="Wind":
            tech_flag = "Wind energy"
        else:
            group_col = "RE or Non-RE"
            tech_flag = "Total Renewable"
        capacity_2023 = self.IRENA_STATS.loc[(self.IRENA_STATS["Year"]==2022) & (self.IRENA_STATS[group_col]==tech_flag)].rename(columns={"Electricity Installed Capacity (MW)":"2023"}).groupby(["ISO3 code", group_col]).agg("sum").reset_index()
        capacity_2024 = self.IRENA_STATS.loc[
            (self.IRENA_STATS["Year"] == 2023) & (self.IRENA_STATS[group_col]==tech_flag)].rename(columns={"Electricity Installed Capacity (MW)":"2024"}).groupby(
            ["ISO3 code", group_col]).agg("sum").reset_index()
        joint_capacity = pd.merge(capacity_2023, capacity_2024, how="left", on="ISO3 code")
        joint_capacity["Increase_2024"] = joint_capacity["2024"] - joint_capacity["2023"]
        results_df = pd.merge(results_df, joint_capacity[["ISO3 code", "Increase_2024"]], how="left", left_on="Country code", right_on="ISO3 code")
        results_df.to_csv("COUNTRY_LEVEL_SUMMARY.csv")

        # Produce plot
        plotter = BubbleRiskPlot()
        fig, ax = plotter.plot(
            df=results_df,
            yticks=np.arange(0, 200, 50),
            x_col="grid_intensity",
            y_col="Benchmark_Ratio",
            ylabel="LCOE relative to electricity prices (% of price)",
            xlabel="Grid Carbon Intensity (gCO2/kWh)",
            size_col="Increase_2024", label_col="Country code")

        if technology == "Solar":
            graphmarking = "a"
        elif technology == "Wind":
            graphmarking = "b"
        else:
            graphmarking = "a"

        # Add colorbar for CF
        ax.set_ylim([0, 400])
        ax.set_xlim([0, 1200])
        ax.axhspan(ymin=25, ymax=50, alpha=0.25, color="grey", label="Range across OECD countries")
        ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=10, fontweight='bold')

        plt.savefig(self.output_folder + technology + "_LCOE_Abatement_Country.png")

        plt.show()
        time.sleep(15)

    def produce_potential_curve_v3(self, supply_ds, filename=None, graphmarking=None, title=None, xlim=None, technology=None):

        def thousands_format(x, pos):
            return f'{int(x):,}'

        # Convert the dataset into a dataframe
        supply_df = supply_ds.to_dataframe()

        # Merge with the country mapping
        merged_supply_df = pd.merge(supply_df, self.country_mapping.rename(columns={"index": "Country"}), how="left",
                                    on="Country")

        # Merge with country_mapping to give country info
        supply_df = merged_supply_df.copy().dropna(axis=0, subset=["Calculated_LCOE", "Country"], how="all")
        supply_df.loc[supply_df["Country code"] == "CHN", "sub-region"] = "China"
        supply_df.loc[supply_df["Country code"] == "IND", "sub-region"] = "India"

        # Drop locations with 111 (code for LCOE > benchmark)
        #supply_df = supply_df.loc[supply_df["LCOE_Reductions"] != 111]
        supply_df["LCOE_Reductions"] = supply_df["Calculated_LCOE"] - supply_df["Benchmark_LCOE"]

        # Convert units to PWh
        supply_df['technical_potential'] = supply_df['technical_potential'] / 1e+012

        # Sort values by LCOE reductions
        supply_df = supply_df.round({'LCOE_Reductions': 1})
        supply_df = supply_df.sort_values(by=['LCOE_Reductions'], ascending=True)
        supply_df['cumulative_potential'] = supply_df['technical_potential'].cumsum()

        # Print share of total potential
        ref = supply_df.copy()
        len_zero = np.nanmax(ref.loc[ref["LCOE_Reductions"]>0]["technical_potential"].cumsum())
        max = np.nanmax(supply_df['cumulative_potential'])
        percentage = len_zero / max * 100

        # Work out LCOE reductions for EMDEs
        reduction = supply_df.copy()
        average_reduction = np.nanmedian(
            reduction.loc[~(reduction["Country"].isin(self.unfccc_countries["index"].tolist()))]["LCOE_Reductions"])

        # Print maximums
        print(
            f"The maximum for supply_df is {max} PWh, with  {percentage}% where the cost is below the benchmark and a median reduction in EMDEs of {average_reduction}USD/MWh")

        # Set regional colour scheme
        region_colors_old = {
            "India": "yellow",
            "China": "darkred",
            "Southern Europe": "#4f81bd",  # medium blue
            "Western Europe": "#2e5fa4",  # dark blue
            "Northern Europe": "#7ea6e0",  # light blue
            "Eastern Europe": "#3b6fb6",  # steel blu
            "Western Asia": "#f46d43",  # orange-red
            "Southern Asia": "#d73027",  # strong red
            "South-eastern Asia": "#fdae61",  # orange
            "Eastern Asia": "#e6550d",  # burnt orange
            "Central Asia": "#fb8c00",  # deep orange
            "Northern Africa": "#66a61e",  # olive green
            "Sub-Saharan Africa": "#1b9e77",  # deep green
            "Latin America and the Caribbean": "#8e44ad",  # purple
            "Northern America": "#5e3c99",  # dark purple
            "Oceania": "#1f9fb5",  # teal
            "Australia and New Zealand": "#2bb1c1",  # light teal
            "Unclassified": "#bdbdbd"  # grey
        }
        region_colors = {
            "Europe": "#4f81bd",  # blue
            "Asia": "#e6550d",  # orange-red
            "Latin America and the Caribbean": "#8e44ad",  # purple
            "Sub-Saharan Africa": "#1b9e77",  # green
            "Oceania": "#1f9fb5",  # teal
            "North America": "#5e3c99",  # dark purple
            "North Africa": "#66a61e",  # olive green
            "Unclassified": "#bdbdbd"  # grey
        }

        # Plot the results
        fig, ax = plt.subplots(figsize=(20, 8))

        # Iterate through each data point and create a bar with the specified width
        for index, row in supply_df.iterrows():
            width = row['technical_potential']  # Bar width, in TWh
            height = row['LCOE_Reductions']  # Bar height
            country = row['Country']
            region = row['region']  # Cumulative production, in TWh
            cumulative_production = row['cumulative_potential']

            # Get color
            region_label = supply_df.loc[supply_df['Country'] == country, 'region'].values[0]
            region_color = region_colors.get(region_label, "k")

            # Plot a bar with the specified width, height, x-position, and color
            ax.bar(cumulative_production, height, width=-1 * width, align='edge', color=region_color)

        # Create the legend
        region_colors = {k: region_colors.get(k, "grey") for k in supply_df["region"].unique().tolist()}
        handles = [plt.Line2D([0], [0], color=color, lw=4, label=region) for region, color in region_colors.items()]

        # Set labels
        ax.set_ylim(-150, 150)
        ax.legend(handles=handles, title="Regions", loc="lower center", ncol=4, fontsize=12, title_fontsize=15)
        ax.set_ylabel('Required LCOE reductions (USD/MWh)', fontsize=20)
        ax.set_xlabel('Annual Electricity Potential (PWh/year)', fontsize=20)
        ax.set_title(title, fontsize=30)
        ax.xaxis.set_major_formatter(FuncFormatter(thousands_format))

        # Set the size of x and y-axis tick labels
        ax.tick_params(axis='x', labelsize=20)  # Adjust the labelsize as needed
        ax.tick_params(axis='y', labelsize=20)  # Adjust the labelsize as needed
        ax.text(0.15, 0.9, technology, transform=ax.transAxes, fontsize=20, fontweight='bold',
                ha="center", va="center")

        if xlim is not None:
            ax.set_xlim([0, xlim])
            ax.xaxis.set_ticks(np.arange(0, xlim + 100, 100))

        if graphmarking is not None:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')

        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches="tight", dpi=500)

        plt.show()

        return supply_df

    def produce_potential_curve_v4(
            self,
            supply_ds,
            filename=None,
            graphmarking=None,
            title=None,
            xlim=None,
            technology=None
    ):

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        def thousands_format(x, pos):
            return f"{int(x):,}"

        # Convert the dataset into a dataframe
        supply_df = supply_ds.to_dataframe()

        # Merge with the country mapping
        merged_supply_df = pd.merge(
            supply_df,
            self.country_mapping.rename(columns={"index": "Country"}),
            how="left",
            on="Country"
        )

        # Merge with country_mapping to give country info
        supply_df = merged_supply_df.copy().dropna(
            axis=0,
            subset=["Calculated_LCOE", "Country"],
            how="all"
        )

        supply_df.loc[supply_df["Country code"] == "CHN", "sub-region"] = "China"
        supply_df.loc[supply_df["Country code"] == "IND", "sub-region"] = "India"

        # Calculate reductions
        supply_df["LCOE_Reductions"] = (
                supply_df["Calculated_LCOE"] - supply_df["Benchmark_LCOE"]
        )

        # Convert units to PWh
        supply_df["technical_potential"] = supply_df["technical_potential"] / 1e12

        # Sort by reduction and compute cumulative potential
        supply_df = supply_df.round({"LCOE_Reductions": 1})
        supply_df = supply_df.sort_values(by=["LCOE_Reductions"], ascending=True)
        supply_df["cumulative_potential"] = supply_df["technical_potential"].cumsum()

        # Print summary stats
        ref = supply_df.copy()
        len_zero = np.nanmax(
            ref.loc[ref["LCOE_Reductions"] > 0, "technical_potential"].cumsum()
        )
        total_max = np.nanmax(supply_df["cumulative_potential"])
        percentage = len_zero / total_max * 100 if total_max > 0 else np.nan

        reduction = supply_df.copy()
        average_reduction = np.nanmedian(
            reduction.loc[
                ~(reduction["Country"].isin(self.unfccc_countries["index"].tolist())),
                "LCOE_Reductions"
            ]
        )

        print(
            f"The maximum for supply_df is {total_max} PWh, with {percentage}% "
            f"where the cost is below the benchmark and a median reduction in "
            f"EMDEs of {average_reduction} USD/MWh"
        )

        # Region colours to define which regions to plot
        region_colors = {
            "Europe": "#4f81bd",
            "Asia": "#e6550d",
            "Latin America and the Caribbean": "#8e44ad",
            "Sub-Saharan Africa": "#1b9e77",
            "Oceania": "#1f9fb5",
            "North America": "#5e3c99",
            "North Africa": "#66a61e",
            "Unclassified": "#bdbdbd"
        }

        # Keep only requested regions that are actually present
        region_order = [r for r in region_colors.keys() if r in supply_df["region"].dropna().unique()]

        if len(region_order) == 0:
            raise ValueError("No matching regions found in supply_df['region'].")

        # Build cumulative positions within each region
        regional_dfs = []
        for region in region_order:
            region_df = supply_df.loc[supply_df["region"] == region].copy()
            region_df = region_df.sort_values(by=["LCOE_Reductions"], ascending=True)
            region_df["cumulative_potential"] = region_df["technical_potential"].cumsum()
            regional_dfs.append(region_df)

        # Create vertically stacked subplots
        fig, axes = plt.subplots(
            nrows=len(region_order),
            ncols=1,
            figsize=(20, 3.5 * len(region_order)),
            sharex=True
        )

        if len(region_order) == 1:
            axes = [axes]

        # Plot each region in its own subplot
        for i, (region, region_df) in enumerate(zip(region_order, regional_dfs)):
            ax = axes[i]
            color = region_colors[region]

            for _, row in region_df.iterrows():
                width = row["technical_potential"]
                height = row["LCOE_Reductions"]
                cumulative_production = row["cumulative_potential"]

                ax.bar(
                    cumulative_production,
                    height,
                    width=-1 * width,
                    align="edge",
                    color=color,
                    edgecolor="none"
                )

            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylim(-150, 150)
            ax.set_ylabel(region, fontsize=14, rotation=0, labelpad=60, va="center")
            ax.tick_params(axis="y", labelsize=12)
            ax.tick_params(axis="x", labelsize=14)
            ax.xaxis.set_major_formatter(FuncFormatter(thousands_format))

            if graphmarking is not None and i == 0:
                ax.text(
                    0.02, 0.94, graphmarking,
                    transform=ax.transAxes,
                    fontsize=20,
                    fontweight="bold"
                )

            if technology is not None and i == 0:
                ax.text(
                    0.15, 0.9, technology,
                    transform=ax.transAxes,
                    fontsize=20,
                    fontweight="bold",
                    ha="center",
                    va="center"
                )

        # Shared labels and title
        axes[-1].set_xlabel("Annual Electricity Potential (PWh/year)", fontsize=18)
        fig.supylabel("Required LCOE reductions (USD/MWh)", fontsize=18)

        if title is not None:
            fig.suptitle(title, fontsize=24, y=0.995)

        if xlim is not None:
            axes[-1].set_xlim([0, xlim])
            axes[-1].set_xticks(np.arange(0, xlim + 100, 100))

        plt.tight_layout(rect=[0.06, 0.04, 1, 0.97])

        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches="tight", dpi=500)

        plt.show()

        return supply_df
    
    
    def produce_cf_curve_v3(self, supply_ds, filename=None, graphmarking=None, title=None, xlim=None, technology=None):

        def thousands_format(x, pos):
            return f'{int(x):,}'

        # Convert the dataset into a dataframe
        supply_df = supply_ds.to_dataframe()

        # Calculate LCOE reductions
        supply_df["LCOE_Reductions"] = supply_df["Calculated_LCOE"] - supply_df["Benchmark_LCOE"]

        # Convert units to PWh
        supply_df['technical_potential'] = supply_df['technical_potential'] / 1e+012

        # Sort values by LCOE reductions
        supply_df = supply_df.round({'LCOE_Reductions': 1})
        supply_df = supply_df.sort_values(by=['LCOE_Reductions'], ascending=True)
        supply_df['cumulative_potential'] = supply_df['technical_potential'].cumsum()

        # Plot the results
        fig, ax = plt.subplots(figsize=(20, 8))
        color_labels = {}
        cmap = mpl.colormaps['gnuplot_r']
        if technology == "Solar":
            vmax = 30
            vmin = 10
        else:
            vmax = 60
            vmin = 20
        ticks = np.arange(vmin, vmax + 5, 5)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)  # Normalize to the range of GDP


        # Iterate through each data point and create a bar with the specified width
        for index, row in supply_df.iterrows():
            width = row['technical_potential']  # Bar width, in TWh
            height = row['LCOE_Reductions']  # Bar height
            cf = row['CF'] * 100
            cumulative_production = row['cumulative_potential']

            # Get color
            color = cmap(norm(cf))

            # Plot a bar with the specified width, height, x-position, and color
            ax.bar(cumulative_production, height, width=-1 * width, align='edge', color=color)

        # Set labels
        ax.set_ylim(-150, 150)
        ax.set_ylabel('Required LCOE reductions (USD/MWh)', fontsize=20)
        ax.set_xlabel('Annual Electricity Potential (PWh/year)', fontsize=20)
        ax.set_title(title, fontsize=30)

        # Add color bar
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                            ticks=ticks, orientation="horizontal", location="bottom", fraction=0.15, pad=0.15)
        cbar.set_label('Average capacity factor (%)', fontsize=20)
        cbar.ax.tick_params(labelsize=15)

        # Set the size of x and y-axis tick labels
        ax.tick_params(axis='x', labelsize=20)  # Adjust the labelsize as needed
        ax.tick_params(axis='y', labelsize=20)  # Adjust the labelsize as needed
        ax.text(0.15, 0.9, technology, transform=ax.transAxes, fontsize=20, fontweight='bold',
                ha="center", va="center")

        if xlim is not None:
            ax.set_xlim([0, xlim])
            ax.xaxis.set_ticks(np.arange(0, xlim + 100, 100))

        if graphmarking is not None:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')

        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches="tight", dpi=500)

        plt.show()

        return supply_df




    def produce_mac_curve(self, results, filename=None, graphmarking=None, title=None, ylim=None):

        # Convert results to a dataframe
        results_df = results.to_dataframe()
        results_df = results_df.loc[~(results_df["Country"].isnull())]

        # Remove unfccc countries
        unfccc = pd.read_csv("./DATA/unfccc.csv")
        unfccc_index = unfccc["index"].values.tolist()
        results_df = results_df.loc[~(results_df["Country"].isin(unfccc_index))]


        # Group by country and take the median result
        results_df = results_df.groupby("Country").agg("mean").reset_index()
        results_df = results_df.loc[(results_df["abatement_cost"] != 999) & ~(results_df["abatement_cost"].isnull())]

        # Merge on countries
        results_df = results_df.merge(self.country_mapping, how="left", left_on="Country", right_on="index")
        results_df.loc[results_df["Region"]=="CHN", "sub-region"] = "China"
        results_df.loc[results_df["Region"] == "IND", "sub-region"] = "India"


        # Sort the values and calculate the cumulative emissions sum
        results_df = results_df.round({'abatement_cost': 3})
        results_df = results_df.sort_values(by=['abatement_cost'], ascending=True)
        results_df['cumulative_emissions'] = results_df['emissions'].cumsum()

        # Plot the results
        fig, ax = plt.subplots(figsize=(20, 8))
        color_labels = {}
        cmap = mpl.colormaps['gnuplot_r']
        norm = mpl.colors.Normalize(vmin=0, vmax=30)  # Normalize to the range of GDP

        # Set region colormapping
        region_colors = {
            "China": "red",
            "Southern Europe": "#1f77b4",  # blue
            "Western Asia": "#ff7f0e",  # orange
            "Southern Asia": "#2ca02c",  # green
            "Latin America and the Caribbean": "#d62728",  # red
            "Sub-Saharan Africa": "#9467bd",  # purple
            "Western Europe": "#e377c2",  # pink
            "Australia and New Zealand": "#7f7f7f",  # gray
            "Northern Europe": "#bcbd22",  # olive
            "Eastern Europe": "#17becf",  # teal
            "Northern America": "#aec7e8",  # light blue
            "South-eastern Asia": "#ffbb78",  # light orange
            "Eastern Asia": "#98df8a",  # light green
            "Northern Africa": "#ff9896",  # light red
            "Oceania": "#c5b0d5",  # light purple
            "Central Asia": "#f7b6d2",  # light pink
            "India": "#ffcc00"
        }

        # Iterate through each data point and create a bar with the specified width
        for index, row in results_df.iterrows():
            width = row['emissions']  # Bar width, in TWh
            height = row['abatement_cost']  # Bar height
            country = row['Country']
            cumulative_production = row['cumulative_emissions']  # Cumulative emissions, in MtCO2

            # Get color
            region_label = results_df.loc[results_df['index'] == country, 'sub-region'].values[0]
            region_color = region_colors.get(region_label)

            # Plot a bar with the specified width, height, x-position, and color
            ax.bar(cumulative_production, height, width=-1 * width, align='edge', color=region_color, label=region_label)

        def thousands_format(x, pos):
            return f'{int(x):,}'


        legend_handles = [
    Line2D([0], [0], color=color, linewidth=4, label=label,
           markerfacecolor=color, markersize=10)
    for label, color in region_colors.items()]

        # Set labels
        ax.set_ylabel('Marginal abatement cost (USD/tCO2)', fontsize=20)
        ax.set_xlabel('Annual Power Sector Emissions (MtCO2/year)', fontsize=25)
        ax.set_title(title, fontsize=30)
        ax.xaxis.set_major_formatter(FuncFormatter(thousands_format))

        # Set the size of x and y-axis tick labels
        ax.tick_params(axis='x', labelsize=20)  # Adjust the labelsize as needed
        ax.tick_params(axis='y', labelsize=20) # Adjust the labelsize as needed
        if ylim is None:
            ylim=1000
        ax.set_ylim([0, ylim])

        # Add color bar
        #cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                            #ticks=[0, 5, 10, 15, 20, 25, 30], format=',',
                            #extend="max", anchor=(0.25, 0.5))
        #cbar.ax.yaxis.set_major_formatter(FuncFormatter(thousands_format))
        #cbar.set_label('Cost of capital (%, 2024)', fontsize=20)
        #cbar.ax.tick_params(labelsize=15)

        # Add in a legend
        ax.legend(handles=legend_handles, loc="upper center", ncols=3)

        if graphmarking is not None:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')

        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches="tight")

        plt.show()


    def produce_mac_curve_v2(self, results, filename=None, graphmarking=None, title=None, ylim=None, future=None):

        # Convert results to a dataframe
        results_df = results.to_dataframe()
        results_df = results_df.loc[~(results_df["Country"].isnull())]

        # Remove unfccc countries
        unfccc = pd.read_csv("./DATA/unfccc.csv")
        unfccc_index = unfccc["index"].values.tolist()
        results_df = results_df.loc[~(results_df["Country"].isin(unfccc_index))]

        # Merge with the country mapping
        results_df = results_df.merge(self.country_mapping, how="left", left_on="Country", right_on="index")
        #results_df.loc[results_df["Region"] == "CHN", "sub-region"] = "China"
        #results_df.loc[results_df["Region"] == "IND", "sub-region"] = "India"

        # Drop locations with negative abatement cost
        if future is not None:
            results_df["abatement_cost"] = results_df["future_abatement_cost"]
        results_df = results_df.loc[results_df["abatement_cost"] != -111]


        # Convert units to TWh
        results_df['technical_potential'] = results_df['technical_potential'] / 1e+012

        # Sort the values and calculate the cumulative emissions sum
        results_df = results_df.round({'abatement_cost': 0})
        results_df = results_df.sort_values(by=['abatement_cost'], ascending=True)
        results_df['cumulative_potential'] = results_df['technical_potential'].cumsum()

        # Plot the results
        fig, ax = plt.subplots(figsize=(20, 8))
        color_labels = {}
        cmap = mpl.colormaps['gnuplot_r']
        norm = mpl.colors.Normalize(vmin=0, vmax=30)  # Normalize to the range of GDP

        # Set region colormapping
        region_colors_old = {
            "India": "yellow",
            "China": "darkred",
            "Southern Europe": "#4f81bd",  # medium blue
            "Western Europe": "#2e5fa4",  # dark blue
            "Northern Europe": "#7ea6e0",  # light blue
            "Eastern Europe": "#3b6fb6",  # steel blu
            "Western Asia": "#f46d43",  # orange-red
            "Southern Asia": "#d73027",  # strong red
            "South-eastern Asia": "#fdae61",  # orange
            "Eastern Asia": "#e6550d",  # burnt orange
            "Central Asia": "#fb8c00",  # deep orange
            "Northern Africa": "#66a61e",  # olive green
            "Sub-Saharan Africa": "#1b9e77",  # deep green
            "Latin America and the Caribbean": "#8e44ad",  # purple
            "Northern America": "#5e3c99",  # dark purple
            "Oceania": "#1f9fb5",  # teal
            "Australia and New Zealand": "#2bb1c1",  # light teal
            "Unclassified": "#bdbdbd"  # grey
        }

        region_colors = {
            "Europe": "#4f81bd",  # blue
            "Asia": "#e6550d",  # orange-red
            "Latin America and the Caribbean": "#8e44ad",  # purple
            "Sub-Saharan Africa": "#1b9e77",  # green
            "Oceania": "#1f9fb5",  # teal
            "North America": "#5e3c99",  # dark purple
            "North Africa": "#66a61e",  # olive green
            "Unclassified": "#bdbdbd"  # grey
        }

        # Iterate through each data point and create a bar with the specified width
        for index, row in results_df.iterrows():
            width = row['technical_potential']  # Bar width, in TWh
            height = row['abatement_cost']  # Bar height
            country = row['Country']
            cumulative_production = row['cumulative_potential']

            # Get color
            region_label = results_df.loc[results_df['index'] == country, 'region'].values[0]
            region_color = region_colors.get(region_label)

            # Plot a bar with the specified width, height, x-position, and color
            ax.bar(cumulative_production, height, width=-1 * width, align='edge', color=region_color, label=region_label)

        def thousands_format(x, pos):
            return f'{int(x):,}'

        region_colors = {k: region_colors.get(k, "grey") for k in results_df["region"].unique().tolist()}
        legend_handles = [Line2D([0], [0], color=color, linewidth=4, label=label,
           markerfacecolor=color, markersize=10) for label, color in region_colors.items()]

        # Set labels
        ax.set_ylabel('Marginal abatement cost (USD/tCO2)', fontsize=20)
        ax.set_xlabel('Technical potential (PWh p.a.)', fontsize=25)
        ax.set_title(title, fontsize=30)
        ax.xaxis.set_major_formatter(FuncFormatter(thousands_format))

        # Set the size of x and y-axis tick labels
        ax.tick_params(axis='x', labelsize=20)  # Adjust the labelsize as needed
        ax.tick_params(axis='y', labelsize=20) # Adjust the labelsize as needed
        if ylim is None:
            ylim=1000
        ax.set_ylim([0, ylim])

        # Add in a legend
        ax.legend(handles=legend_handles, loc="upper center", ncols=3)

        if graphmarking is not None:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')

        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches="tight")

        plt.show()

    def plot_grant_support_distribution(
            self,
            technology,
            fig=None,
            ax=None,
            title=None,
            filename=None,
            cmap="seismic",
            vmin=-100,
            vmax=100
    ):

        # Calculate values
        lcoe = technology["Calculated_LCOE"]
        reductions = technology["Calculated_LCOE"] - technology["Benchmark_LCOE"]
        value = (reductions / lcoe) * 100

        # Keep only valid cells
        valid_mask = np.isfinite(lcoe.values) & np.isfinite(value.values)
        values_sorted = np.sort(value.values[valid_mask])

        n = len(values_sorted)
        if n == 0:
            raise ValueError("No valid values to plot.")

        # Width sums to 100% of valid cells
        width = 100.0 / n
        x = np.arange(1, n + 1) * width

        # Colormap normalization centered on 0
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        bar_colors = cmap_obj(norm(values_sorted))

        # Plot bars
        ax.bar(
            x,
            values_sorted,
            width=-width,
            align="edge",
            color=bar_colors,
            edgecolor="none"
        )

        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlim(0, 100)
        ax.set_ylabel("Required grant support (%)", fontsize=16)
        ax.set_xlabel("Share of valid locations (%)", fontsize=16)

        if title is not None:
            ax.set_title(title, fontsize=18)

        ax.tick_params(axis="both", labelsize=14)

        # Optional colorbar
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])

        axins = inset_axes(
            ax,
            width="100%",
            height="5%",
            loc="lower center",
            bbox_to_anchor=(0, 1.05, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        cb = fig.colorbar(sm, cax=axins, orientation="horizontal")
        cb.set_label("Grant support (%)", fontsize=14)
        cb.ax.tick_params(labelsize=12)

        if created_new_figure:
            plt.tight_layout()

        if filename is not None and created_new_figure:
            plt.savefig(filename + ".png", bbox_inches="tight", dpi=300)

        if created_new_figure:
            plt.show()

        return fig, ax
    
    def plot_technology_panels(self, solar, wind, hydro, geothermal, bioenergy):
        
        # Do a for loop
        technology_labels = ["Solar", "Wind", "Hydro", "Geothermal", "Bioenergy"]
        technologies = [solar, wind, hydro, geothermal, bioenergy]
        for i, technology in enumerate(technologies):
            fig, axes = plt.subplots(
                3, 1, figsize=(18, 24),
                subplot_kw={"projection": ccrs.PlateCarree()},
                facecolor="white"
            )

            label = technology_labels[i]
            self.plot_data_shading(technology['Calculated_LCOE'], technology.latitude, technology.longitude,
                               tick_values=[0, 25, 50, 75, 100, 125, 150], cmap="YlOrRd",
                               title=label +" Levelised Cost: " + label + " (USD/MWh) ",
                               filename=self.output_folder + label + "_LCOE", fig=fig, ax=axes[0])
            technology["LCOE_Reductions"] = technology["Calculated_LCOE"] - technology["Benchmark_LCOE"]
            self.plot_data_shading(technology["LCOE_Reductions"],technology.latitude.values,  technology.longitude.values,
                                   tick_values=[-100, -75, -50, -25, 0, 25, 50, 75, 100], cmap="seismic",
                                   filename=self.output_folder + "Price_Reductions_"+ label +"_Full", fig=fig, ax=axes[1],
                                   title="Required LCOE reductions: " + label + " (USD/MWh)"
                                   )
            technology["LCOE_Reductions"] = xr.where(
                technology["Calculated_LCOE"] < technology["Benchmark_LCOE"],
                99,
                technology["Calculated_LCOE"] - technology["Benchmark_LCOE"])
            self.plot_grant_support_distribution(
                technology,
                fig=fig,
                ax=axes[2],
                title="Required grant support: " + label + " (%)"
            )
            pos1 = axes[0].get_position()
            pos3 = axes[2].get_position()
            #axes[2].set_position([pos1.x0, pos3.y0, pos1.width, pos3.height])
            plt.savefig(self.output_folder + f"{label}_panel.png", bbox_inches="tight")
            plt.show()

        return
    
    def plot_cheapest_technology_global(
            self,
            cheapest_tech,  # xr.DataArray OR xr.Dataset
            var_name="Cheapest_Technology",
            lat_name="latitude",
            lon_name="longitude",
            filename=None,
            title="Cheapest Technology",
            graphmarking=None,
            cmap_name="Paired"
    ):
        """
        Plot global categorical map with legend only (no colorbar), preserving xarray workflow.
        """

        # 1) Ensure DataArray
        if isinstance(cheapest_tech, xr.Dataset):
            if var_name not in cheapest_tech:
                raise KeyError(f"'{var_name}' not found in dataset variables: {list(cheapest_tech.data_vars)}")
            da = cheapest_tech[var_name]
        elif isinstance(cheapest_tech, xr.DataArray):
            da = cheapest_tech
        else:
            raise TypeError("cheapest_tech must be an xarray.DataArray or xarray.Dataset")

        # 2) Check coords
        if lat_name not in da.coords or lon_name not in da.coords:
            raise KeyError(f"Expected coords '{lat_name}' and '{lon_name}' in DataArray coords: {list(da.coords)}")

        # 3) Build unique category list from xarray values (non-null)
        tech_order = ["Solar", "Onshore Wind", "Hydroelectric", "Bioenergy", "Geothermal"]
        unique_tech = tech_order
        ntech = len(unique_tech)
        if ntech == 0:
            raise ValueError("No valid technology labels found in DataArray.")

        # 4) Map labels -> integer codes as DataArray (still xarray-centric)
        tech_to_code = {name: i for i, name in enumerate(unique_tech)}

        code_da = xr.full_like(da, fill_value=np.nan, dtype=float)
        for name, code in tech_to_code.items():
            code_da = xr.where(da == name, code, code_da)

        # 6) Plot
        fig = plt.figure(figsize=(18, 9), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        plot_vals = code_da.astype(float).values
        present_codes = np.unique(plot_vals[~np.isnan(plot_vals)]).astype(int)

        # Build colormap using full code range so indices remain valid
        max_code = int(present_codes.max())
        ncolors = max_code + 1
        cmap = plt.get_cmap(cmap_name, ncolors)
        #cmap.set_bad(color="lightgray")
        bounds = np.arange(-0.5, ncolors + 0.5, 1)
        norm = BoundaryNorm(bounds, ncolors=ncolors, clip=True)
        u = np.unique(plot_vals[~np.isnan(plot_vals)])
        print("unique plotted codes:", u[:20], "count:", len(u))
        print("expected codes:", np.arange(ncolors))
        mesh = ax.pcolormesh(
            code_da[lon_name].values,
            code_da[lat_name].values,
            plot_vals,
            cmap=cmap,
            norm=norm,
            shading="auto",
            transform=ccrs.PlateCarree()
        )

        ax.set_global()
        ax.set_ylim([-65, 90])
        ax.coastlines(linewidth=0.8)
        borders = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_0_boundary_lines_land",
            scale="110m",
            facecolor="none"
        )
        ax.add_feature(borders, edgecolor="gray", linestyle=":", linewidth=0.5)

        # 7) Legend only
        legend_handles = [
            Patch(facecolor=cmap(i), edgecolor="black", label=tech_name)
            for i, tech_name in enumerate(unique_tech)
        ]

        if bool(code_da.isnull().any()):
            legend_handles.append(Patch(facecolor="lightgray", edgecolor="black", label="No data"))

        ax.legend(
            handles=legend_handles,
            loc="lower left",
            fontsize=11,
            frameon=True,
            title="Technology",
            title_fontsize=12
        )

        if graphmarking is not None:
            ax.text(0.02, 0.96, graphmarking, transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="top")

        if filename is not None:
            plt.savefig(filename + ".png", dpi=300, bbox_inches="tight")

        plt.show()

        
    def plot_data_shading(self, values, latitudes, longitudes, anchor=None, filename=None, increment=None, title=None, tick_values=None, cmap=None, extend_set=None, graphmarking=None, special_value=None,
                          hatch_label=None, hatch_label_2=None, special_value_2=None, center_norm=None, ylim=None, fig=None, ax=None):
    
        # create the heatmap using pcolormesh
        if anchor is None:
            anchor = 0.355
        created_new_figure= False
        if ax is None:
            fig = plt.figure(figsize=(30, 15), facecolor="white")
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            created_new_figure = True
        if center_norm is None:
            heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.Normalize(vmin=tick_values[0], vmax=tick_values[-1]), transform=ccrs.PlateCarree(), cmap=cmap)
        else:
            heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.TwoSlopeNorm(vmin=tick_values[0], vcenter=center_norm, vmax=tick_values[-1]), cmap=cmap)

        # Check if there is a need for extension
        values_min = np.nanmin(values)
        values_max = np.nanmax(values)
        if values_min < tick_values[0]:
            extend = "min"
        elif values_max > tick_values[-1]:
            extend = "max"
        else:
            extend = "neither"
        if extend_set is not None:
            extend = extend_set
        if (values_max > tick_values[-1]) & (values_min < tick_values[0]):
            extend = "both"

        axins = inset_axes(
            ax,
            width="100%",  # wide to span across the top
            height="5%",  # thin since it's horizontal
            loc="lower center",
            bbox_to_anchor=(0, 1.05, 1, 1),  # positions it above the plot
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cb = fig.colorbar(heatmap, cax=axins, ticks=tick_values, format="%0.0f",
                          extend=extend, orientation="horizontal")


        cb.ax.tick_params(labelsize=20)
        if title is not None:
            cb.ax.set_title(title, fontsize=25)

        # Add the special shading
        if special_value is not None:
            special_overlay = np.where(values == special_value, 1, np.nan)
            hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['/'], colors="silver", linewidth=0.15, transform=ccrs.PlateCarree())

        if special_value_2 is not None:
            special_overlay = np.where(values == special_value_2, 1, np.nan)
            hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['/'], colors="gold", linewidth=0.15, transform=ccrs.PlateCarree())

        # set the extent and aspect ratio of the plot
        ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
        aspect_ratio = (latitudes.max() - latitudes.min()) / (longitudes.max() - longitudes.min())
        ax.set_aspect(1)

        # add axis labels and a title
        ax.set_xlabel('Longitude', fontsize=30)
        ax.set_ylabel('Latitude', fontsize=30)
        borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none')
        ax.add_feature(borders, edgecolor='gray', linestyle=':')
        ax.coastlines()
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        ax.coastlines()
        if graphmarking is not None:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')

        hatch_patches=[]
        if special_value is not None and hatch_label is not None:
            hatch_patch_1 = Patch(facecolor='silver', edgecolor='black', hatch="/", label=hatch_label)
            hatch_patches.append(hatch_patch_1)

        if hatch_label_2 is not None:
            hatch_patch_2 = Patch(facecolor='gold', edgecolor='black', hatch="/", label=hatch_label_2)
            hatch_patches.append(hatch_patch_2)

        if hatch_patches:
            ax.legend(handles=hatch_patches, loc='lower left', fontsize=20)

        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches="tight")

        return 
    
    def plot_data_discrete_shading(self, values, latitudes, longitudes, anchor=None, filename=None, increment=None, title=None, tick_values=None, cmap=None, extend_set=None, graphmarking=None, special_value=None, hatch_label=None, hatch_label_2=None, special_value_2=None):
    
        cmap = plt.get_cmap(cmap, len(tick_values)+1)

        if anchor is None:
            anchor = 0.355

        fig = plt.figure(figsize=(30, 15), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Ensure tick_values are passed and sorted
        if tick_values is None or len(tick_values) < 2:
            raise ValueError("You must provide at least two tick values for discrete boundaries.")

        # Discrete normalization
        norm = BoundaryNorm(boundaries=tick_values, ncolors=len(tick_values) - 1)
        heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=norm, cmap=cmap, transform=ccrs.PlateCarree())



        # Special hatching
        if special_value is not None:
            special_overlay = np.where(values == special_value, 1, np.nan)
            ax.contourf(longitudes, latitudes, special_overlay, hatches=['/'], colors="silver", linewidth=0.15, transform=ccrs.PlateCarree())
        if special_value_2 is not None:
            special_overlay = np.where(values == special_value_2, 1, np.nan)
            ax.contourf(longitudes, latitudes, special_overlay, hatches=['/'], colors="gold", linewidth=0.15, transform=ccrs.PlateCarree())

        # Map setup
        ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
        ax.set_aspect(1)
        ax.set_xlabel('Longitude', fontsize=30)
        ax.set_ylabel('Latitude', fontsize=30)
        borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none')
        ax.add_feature(borders, edgecolor='gray', linestyle=':')
        ax.coastlines()


        if graphmarking is not None:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')

        # Legend for hatching
        hatch_patches = []
        if special_value is not None and hatch_label is not None:
            hatch_patches.append(Patch(facecolor='silver', edgecolor='black', hatch="/", label=hatch_label))
        if special_value_2 is not None and hatch_label_2 is not None:
            hatch_patches.append(Patch(facecolor='gold', edgecolor='black', hatch="/", label=hatch_label_2))
        if hatch_patches:
            ax.legend(handles=hatch_patches, loc='lower left', fontsize=20)


        # Create discrete legend instead of colorbar
        legend_patches = []
        for i in range(len(tick_values) - 1):
            color = cmap(i+1)
            label = f"{tick_values[i]*100}–{tick_values[i+1]*100}%"
            patch = Patch(facecolor=color, edgecolor='black', label=label)
            legend_patches.append(patch)

        # Add patch for values above the last bin
        values_max = np.nanmax(values)
        if values_max > tick_values[-1]:
            # Choose the overflow color — you can make it the same as last or custom
            overflow_color = cmap(len(tick_values))
            overflow_patch = Patch(facecolor=overflow_color, edgecolor='black', label=f"> {tick_values[-1]*100}%")
            legend_patches.append(overflow_patch)

        values_min = np.nanmin(values)
        if values_min < tick_values[0]:
            underflow_color = cmap(0)
            underflow_patch = Patch(facecolor=underflow_color, edgecolor='black', label=f"< {tick_values[0]*100}%")
            legend_patches.insert(0, underflow_patch)

        # Add legend to the plot
        ax.legend(
            handles=legend_patches,
            loc='lower left',
            title="LCOE to\nElectricity Price\nRatio",
            fontsize=20,
            title_fontsize=22,
            frameon=True,
            alignment="center"
        )

        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches="tight")
    
        return

    def plot_data_three_shading(self, values, X, Y, color_a, color_b, label_a, label_b, filename=None,
                                title=None, graphmarking=None):
        # Get latitudes and longitudes
        latitudes = values.latitude
        longitudes = values.longitude

        # Create categorical mask: 0 for <X, 1 for X–Y, 2 for >Y
        masked = np.full(values.shape, np.nan)
        masked[values == X] = 0
        masked[values == Y] = 1

        # Define colormap
        cmap = ListedColormap([color_a, color_b])

        # Plot setup
        fig = plt.figure(figsize=(30, 15), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        heatmap = ax.pcolormesh(longitudes, latitudes, masked, cmap=cmap, transform=ccrs.PlateCarree())

        # Colorbar
        axins = inset_axes(ax, width="1.5%", height="80%", loc="lower left",
                           bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cb = fig.colorbar(heatmap, cax=axins, ticks=[0.5, 1.5], format="%0.0f")
        cb.ax.set_yticklabels([label_a, label_b])
        cb.ax.tick_params(labelsize=20)
        if title:
            cb.ax.set_title(title, fontsize=25)

        # Map features
        ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
        ax.set_aspect(1)
        ax.set_xlabel('Longitude', fontsize=30)
        ax.set_ylabel('Latitude', fontsize=30)
        borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land',
                                               scale='10m', facecolor='none')
        ax.add_feature(borders, edgecolor='black', linestyle=':')
        ax.coastlines()

        if graphmarking:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')

        # Legend
        legend_patches = [
            Patch(facecolor=color_a, edgecolor='black', label=label_a),
            Patch(facecolor=color_b, edgecolor='black', label=label_b),
        ]
        ax.legend(handles=legend_patches, loc='lower left', fontsize=20)

        # Save
        if filename:
            plt.savefig(filename + ".png", bbox_inches="tight")


    def extract_csv(self, data, name):

        selected_data = data.to_dataframe()[["Country", "CF", "Estimated_WACC", "Calculated_LCOE",
                              "Benchmark_LCOE", "Benchmark_Price", "Grant_Equivalent_%", "abatement_cost",
                              "Build_Margin", "Operating_Margin", "IFI_Emissions_Factor", "Additionality_Factor"]]
        selected_data["Benchmark_Price"] = selected_data["Benchmark_Price"] * 1000
        selected_data["Grant_Equivalent_%"] = selected_data["Grant_Equivalent_%"] * 100
        selected_data["CF"] = selected_data["CF"] * 100
        selected_data["Additionality_Factor"] = selected_data["Additionality_Factor"] * 100
        selected_data = selected_data.reset_index().dropna(subset="Calculated_LCOE")

        # Merge on country info
        selected_data = selected_data.merge(self.country_mapping[["index", "Country code", "name", "region"]], how="left", right_on="index", left_on="Country")

        selected_data = selected_data.rename(columns={"CF": "Capacity Factor (avg, %)",
                                                      "name": "Country name",
                                                      "region":"Region",
                                                      "Estimated_WACC": "Estimated Cost of Capital (%)",
                                                      "Calculated_LCOE": "LCOE (USD/MWh)",
                                                      "Benchmark_LCOE": "Required LCOE (USD/MWh)",
                                                      "Benchmark_Price": "Electricity Price (USD/MWh)",
                                                      "Grant_Equivalent_%":"Required Grant Support (%, CAPEX)",
                                                      "abatement_cost": "Marginal abatement cost (USD/tCO2)",
                                                      "Build_Margin": "Build Margin (gCO2/kWh)",
                                                      "Operating_Margin": "Operating Margin (gCO2/kWh)",
                                                      "Additionality_Factor": "Probability of Additionality (%)",
                                                      "IFI_Emissions_Factor": "IFI Emissions Factor (gCO2/kWh)"
                                                      })

        selected_data.to_csv(name + ".csv")

        return selected_data
    
    def visualiser_pipeline(self): 
        
        # Specify results
        wind_results = self.wind_results
        solar_results = self.solar_results

        # Save subset of results
        self.extract_csv(wind_results, "WIND_RESULTS_SUMMARY")
        self.extract_csv(solar_results, "SOLAR_RESULTS_SUMMARY")
        
        # Produce LCOE to Electricity Price ratios
        lcoe_electricity = input("Plot LCOE to Electricity Price ratios? Y/N")
        if lcoe_electricity == "Y":
            self.plot_data_discrete_shading(solar_results["Calculated_LCOE"]/solar_results["Benchmark_Price"]/1000, solar_results.latitude.values, solar_results.longitude.values, tick_values=[0.1, 0.2, 0.3, 0.4, 0.5], cmap="YlOrRd", graphmarking="a", filename=self.output_folder +"Electricity_Price_LCOE_Solar")
            self.plot_data_discrete_shading(wind_results["Calculated_LCOE"]/wind_results["Benchmark_Price"]/1000, wind_results.latitude.values, wind_results.longitude.values, tick_values=[0.1, 0.2, 0.3, 0.4, 0.5], cmap="YlGnBu", graphmarking="b", filename=self.output_folder +"Electricity_Price_LCOE_Wind")
            self.plot_data_shading(solar_results['Estimated_WACC'], solar_results.latitude, solar_results.longitude,
                                   tick_values=[0, 5, 10, 15, 20], cmap="YlOrRd",
                                   title="Solar PV:\nEstimated\nWACC\n(%, real)\n",
                                   filename=self.output_folder + "Solar_WACC", graphmarking="a")
            self.plot_data_shading(wind_results['Estimated_WACC'], solar_results.latitude, solar_results.longitude,
                                   tick_values=[0, 5, 10, 15, 20], cmap="YlGnBu",
                                   title="Onshore Wind:\nEstimated\nWACC\n(%, real)\n",
                                   filename=self.output_folder + "Wind_WACC", graphmarking="b")
            self.plot_data_shading(solar_results['Calculated_LCOE'], solar_results.latitude, solar_results.longitude,
                                   tick_values=[0, 25, 50, 75, 100, 125, 150], cmap="YlOrRd",
                                   title="Solar PV:\nLevelised\nCost\n(USD/MWh)\n",
                                   filename=self.output_folder + "Solar_LCOE", graphmarking="a")
            self.plot_data_shading(wind_results['Calculated_LCOE'], solar_results.latitude, solar_results.longitude,
                                   tick_values=[0, 25, 50, 75, 100, 125, 150], cmap="YlGnBu",
                                   title="Onshore Wind:\nLevelised\nCost\n(USD/MWh)\n",
                                   filename=self.output_folder + "Wind_LCOE", graphmarking="b")
        # Evaluate how far the LCOE needs to fall to reach the benchmark
        wind_results["LCOE_Reductions"] = wind_results["Calculated_LCOE"] - wind_results["Benchmark_LCOE"]
        solar_results["LCOE_Reductions"] = solar_results["Calculated_LCOE"] - solar_results["Benchmark_LCOE"]
        lcoe_reductions = input("Plot LCOE reductions? Y/N")
        if lcoe_reductions == "Y":
            self.plot_data_shading(solar_results["LCOE_Reductions"],
                                        solar_results.latitude.values, solar_results.longitude.values,
                                        tick_values=[-100, -75, -50, -25, 0, 25, 50, 75, 100], cmap="seismic", graphmarking="a",
                                        filename=self.output_folder + "Price_Reductions_Solar_Full",
                                        title="Required LCOE\nreductions:\nSolar\n(USD/MWh)\n"
                                        )
            self.plot_data_shading(wind_results["LCOE_Reductions"],
                                        wind_results.latitude.values, wind_results.longitude.values,
                                        tick_values=[-100, -75,-50, -25, 0, 25, 50, 75, 100], cmap="seismic", graphmarking="a",
                                        filename=self.output_folder + "Price_Reductions_Wind_Full",
                                        title="Required LCOE\nreductions:\nWind\n(USD/MWh)\n"
                                        )
            solar_results["LCOE_Reductions"] = xr.where(solar_results["Calculated_LCOE"] < solar_results["Benchmark_LCOE"], 99,
            solar_results["Calculated_LCOE"] - solar_results["Benchmark_LCOE"])
            wind_results["LCOE_Reductions"] = xr.where(wind_results["Calculated_LCOE"] < wind_results["Benchmark_LCOE"], 99,
            wind_results["Calculated_LCOE"] - wind_results["Benchmark_LCOE"])
            self.plot_data_shading(xr.where(solar_results["LCOE_Reductions"] == 99, 99, solar_results["LCOE_Reductions"] / solar_results["Calculated_LCOE"]*100),
                               solar_results.latitude.values, solar_results.longitude.values,
                               tick_values=[0, 25, 50, 75, 100], cmap="YlOrRd", graphmarking="a",
                               filename=self.output_folder + "Price_Reductions_SolarFull%",
                               special_value=99, hatch_label="Below benchmark",
                               title="Required grant\nsupport:\nSolar\n(%)\n"
                               )
            self.plot_data_shading(xr.where(wind_results["LCOE_Reductions"] ==99, 99, wind_results["LCOE_Reductions"] / wind_results["Calculated_LCOE"]*100),
                               wind_results.latitude.values, wind_results.longitude.values,
                               tick_values=[0, 25, 50, 75, 100], cmap="YlGnBu", graphmarking="b",
                               filename=self.output_folder + "Price_Reductions_WindFull%",
                               special_value=99, hatch_label="Below benchmark",
                               title="Required grant\nsupport:\nWind\n(%)\n"
                               )

            self.produce_potential_curve_v3(solar_results, xlim=1e+03, technology="Solar",
                                            filename=self.output_folder + "Reduction_Potential_Solar", graphmarking="b")
            self.produce_potential_curve_v3(wind_results, xlim=1e+03, technology="Onshore\nWind",
                                            filename=self.output_folder + "Reduction_Potential_Wind",
                                            graphmarking="b")
            #self.produce_cf_curve_v3(solar_results, xlim=1e+03, technology="Solar",
                                            #filename=self.output_folder + "Reduction_Potential_Solar_CF", graphmarking="c")
            #self.produce_cf_curve_v3(wind_results, xlim=1e+03, technology="Onshore\nWind",
                                            #filename=self.output_folder + "Reduction_Potential_Wind_CF",
                                            #graphmarking="c")


        # Depict the LCOE ratio versus grid carbon intensity at a country level
        country = input("Plot country results? Y/N")
        if country == "Y":
            self.produce_country_scatter(solar_results, "Solar")
            self.produce_country_scatter(wind_results, "Wind")

            self.plot_data_shading(wind_results['Additionality_Factor']*100, solar_results.latitude, solar_results.longitude,
                                   tick_values=[0, 25, 50, 75, 100], cmap="Purples",
                                   title="Additionality\nFactor\n(probability, %)\n",
                                   filename=self.output_folder + "Additionality_Factor", graphmarking="b")

        # Produce the MAC curve
        self.produce_mac_curve_v2(solar_results, filename=self.output_folder + "MAC_Curve_Solar", graphmarking="a", ylim=250)
        self.produce_mac_curve_v2(wind_results, filename=self.output_folder + "MAC_Curve_Wind", graphmarking="b", ylim=250)
        #self.produce_mac_curve_v2(solar_results, filename=self.output_folder + "MAC_Curve_Solar_Future",
                                  #graphmarking="a",
                                  #ylim=250, future="True")
        #self.produce_mac_curve_v2(wind_results, filename=self.output_folder + "MAC_Curve_Wind_Future", graphmarking="b",
                                  #ylim=250, future="True")
        
        # Produce the abatement cost in USD per tCO2
        self.plot_data_shading(xr.where(solar_results["abatement_cost"].isnull() & ~np.isnan(solar_results["Calculated_LCOE"]), 999, solar_results['abatement_cost']), solar_results.latitude,
                               solar_results.longitude, tick_values=[0, 50, 100, 150, 200], cmap="YlOrRd",
                               title="Solar PV:\nAbatement\nCost\n(USD/tCO2)\n", filename=self.output_folder +"Solar_Concessional_MAC",
                               graphmarking="a", special_value=999, hatch_label="Unachievable", special_value_2=-111, hatch_label_2="Below benchmark")
        self.plot_data_shading(xr.where(wind_results["abatement_cost"].isnull() & ~np.isnan(wind_results["Calculated_LCOE"]), 999, wind_results['abatement_cost']), solar_results.latitude,
                               solar_results.longitude, tick_values=[0, 50, 100, 150, 200], cmap="YlGnBu", title="Onshore Wind:\n Abatement\nCost\n(USD/tCO2)\n", filename=self.output_folder +"Wind_Concessional_MAC",
                               graphmarking="b", special_value=999, hatch_label="Unachievable", special_value_2=-111, hatch_label_2="Below benchmark")


        solar_change = xr.where(solar_results["abatement_cost"].isnull() & ~np.isnan(solar_results["Calculated_LCOE"]), 999,
                                (solar_results['abatement_cost'] - solar_results['future_abatement_cost'])/solar_results["abatement_cost"]*100)
        solar_change = xr.where(solar_results["abatement_cost"]==-111, -111, solar_change)
        wind_change = xr.where(wind_results["abatement_cost"].isnull() & ~np.isnan(wind_results["Calculated_LCOE"]),
                                999,
                               (wind_results['abatement_cost'] - wind_results['future_abatement_cost'])/wind_results['abatement_cost']*100)
        wind_change = xr.where(wind_results["abatement_cost"] == -111, -111, wind_change)

        # Produce future abatement costs
        self.plot_data_shading(solar_change , solar_results.latitude,
                               solar_results.longitude, tick_values=[0, 20, 40, 60, 80, 100], cmap="YlOrRd",
                               title="Solar PV:\nChange in\nAbatement\nCost\n(USD/tCO2)\n",
                               filename=self.output_folder + "Future_Solar_Concessional_MAC", graphmarking="a",
                               special_value=999, hatch_label="Unachievable", special_value_2=-111, hatch_label_2="Below benchmark")
        self.plot_data_shading(wind_change, solar_results.latitude,
                               solar_results.longitude, tick_values=[0, 20, 40, 60, 80, 100], cmap="YlGnBu",
                               title="Onshore Wind:\nChange in\nAbatement\nCost\n(USD/tCO2)\n",
                               filename=self.output_folder + "Future_Wind_Concessional_MAC", graphmarking="b",
                               special_value=999, hatch_label="Unachievable", special_value_2=-111, hatch_label_2="Below benchmark")




        # Plot benchmark price
        benchmark = input("Plot benchmark price and cheapest technology results? Y/N")
        if benchmark == "Y":
            benchmark_price = xr.where(np.isnan(solar_results["Calculated_LCOE"]), np.nan, solar_results['Benchmark_Price']*1000)
            self.plot_data_shading(benchmark_price, solar_results.latitude, solar_results.longitude,
                               tick_values=[0, 50, 100, 150, 200, 250, 300], cmap="Reds",
                               title="Electricity:\nPrice\nBenchmark\n(USD/MWh)\n",
                               filename=self.output_folder + "Electricity_Price",
                               graphmarking="b")
            benchmark_lcoe = xr.where(np.isnan(solar_results["Calculated_LCOE"]), np.nan,
                                   solar_results['Benchmark_LCOE'])
            self.plot_data_shading(benchmark_lcoe, solar_results.latitude, solar_results.longitude,
                               tick_values=[0, 25, 50, 75, 100], cmap="Reds",
                               title="Electricity:\nPrice\nBenchmark\n(USD/MWh)\n",
                               filename=self.output_folder + "Electricity_LCOE")


            # Get cheapest technology
            cheapest_tech = xr.where(solar_results["Calculated_LCOE"] < wind_results["Calculated_LCOE"], "Solar", "Wind")
            cheapest_tech = xr.where(np.isnan(solar_results["Calculated_LCOE"]), np.nan,
                                     cheapest_tech)
            cheapest_lcoe = xr.where(solar_results["Calculated_LCOE"] < wind_results["Calculated_LCOE"],
                                     solar_results["Calculated_LCOE"], wind_results["Calculated_LCOE"])
            cheapest_ratio = cheapest_lcoe / solar_results["Benchmark_Price"] / 1000


            # Plot cheapest technology
            self.plot_data_shading(cheapest_lcoe, solar_results.latitude, solar_results.longitude,
                                   tick_values=[0, 25, 50, 75, 100, 125, 150], cmap="plasma_r",
                                   title="Cheapest:\nLevelised\nCost\n(USD/MWh)\n",
                                   filename=self.output_folder + "Renewable_LCOE", graphmarking="a")
            self.plot_data_discrete_shading(cheapest_ratio,wind_results.latitude.values,
                                            wind_results.longitude.values, tick_values=[0.1, 0.2, 0.3, 0.4, 0.5],
                                            cmap="plasma_r", graphmarking="b",
                                            filename=self.output_folder +"Electricity_Price_LCOE_Renewable")
            self.plot_data_three_shading(cheapest_tech, "Solar", "Wind", "yellow", "lightblue", "Solar PV", "Onshore Wind", filename=self.output_folder +"Cheapest_Technology",
                                    graphmarking="c")
            renewable_reductions = xr.where(
                solar_results['abatement_cost'] < wind_results['abatement_cost'],
                solar_results["Calculated_LCOE"] - solar_results["Benchmark_LCOE"],
                wind_results["Calculated_LCOE"] - wind_results["Benchmark_LCOE"])
            renewable_reductions = xr.where(cheapest_lcoe < solar_results["Benchmark_LCOE"], 111,
                                            renewable_reductions)
            self.plot_data_shading(renewable_reductions,
                                   solar_results.latitude.values, solar_results.longitude.values,
                                   tick_values=[0, 10, 20, 30, 40, 50], cmap="Purples",
                                   filename=self.output_folder + "Price_Reductions_Renewable",
                                   special_value=111, hatch_label="Below benchmark",
                                   title="Required LCOE\nreductions:\nCheapest\n(USD/MWh)\n"
                                   )

        # Produce plots for ED
        graphmarkings = ["a", "b", "c", "d", "e", "f"]
        for i, benchmark in enumerate([5, 10, 15, 20, 30]):

            # Get graphmarking
            benchmark = str(benchmark)
            graphmarking = graphmarkings[i] + " (" + benchmark + "%)"

            # Calculate cheapest abatement for ED Figure 7
            cheapest_sensitivity = xr.where(solar_results['abatement_cost' + benchmark] < wind_results['abatement_cost'+ benchmark], solar_results['abatement_cost' + benchmark], wind_results['abatement_cost' + benchmark])
            cheapest_sensitivity = xr.where(np.isnan(solar_results["Calculated_LCOE"]), np.nan,
                                     cheapest_sensitivity)
            self.plot_data_shading(xr.where(solar_results["Country"].isin(self.unfccc_countries["index"]), 999,
                                            cheapest_sensitivity), solar_results.latitude,
                                   solar_results.longitude, tick_values=[0, 50, 100, 150, 200], cmap="Greens",
                                   title="Renewables:\nAbatement\nCost\n(USD/tCO2)\n",
                                   filename=self.output_folder + "Renewable_Concessional_MAC_"+ benchmark,
                                   special_value=999, hatch_label="Annex II", special_value_2=-111,
                                   hatch_label_2="Below benchmark", graphmarking=graphmarking)


            # Calculate required reductions in electricity LCOEs
            solar_reductions = xr.where(solar_results["Calculated_LCOE"] < solar_results["Benchmark_LCOE" + benchmark], 111,
                                        solar_results["Calculated_LCOE"] - solar_results["Benchmark_LCOE"+ benchmark])
            wind_reductions = xr.where(solar_results["Calculated_LCOE"] < solar_results["Benchmark_LCOE"], 111,
                                       solar_results["Calculated_LCOE"] - solar_results["Benchmark_LCOE"])

            self.plot_data_shading(solar_reductions,
                                   solar_results.latitude.values, solar_results.longitude.values,
                                   tick_values=[0, 10, 20, 30, 40, 50], cmap="YlOrRd", graphmarking=graphmarking,
                                   filename=self.output_folder + "Price_Reductions_Solar_"+benchmark,
                                   special_value=111, hatch_label="Below benchmark",
                                   title="Required LCOE\nreductions:\nSolar\n(USD/MWh)\n"
                                   )
            self.plot_data_shading(wind_reductions,
                                   wind_results.latitude.values, wind_results.longitude.values,
                                   tick_values=[0, 10, 20, 30, 40, 50], cmap="YlGnBu", graphmarking=graphmarking,
                                   filename=self.output_folder + "Price_Reductions_Wind_"+benchmark,
                                   special_value=111, hatch_label="Below benchmark",
                                   title="Required LCOE\nreductions:\nWind\n(USD/MWh)\n"
                                   )




            # Calculate WACC reductions for ED Figure 6 for cheapest abatement
            cheapest_grants = xr.where(
                solar_results['abatement_cost' + benchmark] < wind_results['abatement_cost' + benchmark],
                solar_results['Grant_Equivalent_%' + benchmark]*100, wind_results['Grant_Equivalent_%' + benchmark]*100)
            cheapest_grants = xr.where(np.isnan(solar_results["Calculated_LCOE"]), np.nan,
                                            cheapest_grants)
            cheapest_grants = xr.where(cheapest_grants < 0, -111, cheapest_grants)
            self.plot_data_shading(xr.where(solar_results["Country"].isin(self.unfccc_countries["index"]), 999,
                                            cheapest_grants), solar_results.latitude, solar_results.longitude,
                                   tick_values=[0, 20, 40, 60, 80, 100], cmap="Purples",
                                   title="Renewables:\nRequired\nGrant\nEquivalence\n(%)\n",
                                   filename=self.output_folder + "Renewable_Grants_Required_" + benchmark,
                                   special_value=999, hatch_label="Annex II", special_value_2=-111,
                                   hatch_label_2="Below benchmark", graphmarking=graphmarking)

            renewable_reductions = xr.where(
                solar_results['abatement_cost' + benchmark] < wind_results['abatement_cost' + benchmark],
                solar_results["Calculated_LCOE"] - solar_results["Benchmark_LCOE"],
                wind_results["Calculated_LCOE"] - wind_results["Benchmark_LCOE"])
            renewable_reductions = xr.where(cheapest_lcoe < solar_results["Benchmark_LCOE"], 111,
                                            renewable_reductions)
            self.plot_data_shading(renewable_reductions,
                                   solar_results.latitude.values, solar_results.longitude.values,
                                   tick_values=[0, 10, 20, 30, 40, 50], cmap="Purples",
                                   filename=self.output_folder + "Price_Reductions_Renewable_"+benchmark,
                                   special_value=111, hatch_label="Below benchmark",
                                   title="Required LCOE\nreductions:\nCheapest\n(USD/MWh)\n"
                                   )

    def visualiser_cheapest_pipeline(self, results):

        # Save subset of results
        #self.extract_csv(results, "CHEAPEST_RESULTS_SUMMARY")
        results.to_dataframe().to_csv("CHEAPEST_RESULTS_SUMMARY.csv")

        # Produce LCOE to Electricity Price ratios
        self.plot_cheapest_technology_global(results, title="Cheapest Technology",
                                                           graphmarking="b", filename="CheapestTechnologyLocation")
        lcoe_electricity = input("Plot LCOE to Electricity Price ratios? Y/N")
        if lcoe_electricity == "Y":
            self.plot_data_shading(results['Calculated_LCOE'], results.latitude,
                                   results.longitude,
                                   tick_values=[0, 25, 50, 75, 100], cmap="plasma_r",
                                   title="Cheapest Renewables: Levelised Cost (USD/MWh) ",
                                   filename=self.output_folder + "Cheapest_LCOE", graphmarking="a")
            self.plot_data_discrete_shading(
                results["Calculated_LCOE"] / results["Benchmark_Price"] / 1000,
                results.latitude.values, results.longitude.values,
                tick_values=[0.1, 0.2, 0.3, 0.4, 0.5], cmap="plasma_r", graphmarking="c",
                filename=self.output_folder + "Electricity_Price_LCOE_Cheapest")
            self.plot_data_shading(results['Estimated_WACC'], results.latitude,
                                   results.longitude,
                                   tick_values=[0, 5, 10, 15], cmap="plasma",
                                   title="Cheapest Renewables: Estimated WACC (%, real)",
                                   filename=self.output_folder + "Cheapest_WACC", graphmarking="a")
        # Evaluate how far the LCOE needs to fall to reach the benchmark
        results["LCOE_Reductions"] = results["Calculated_LCOE"] - results["Benchmark_LCOE"]
        lcoe_reductions = input("Plot LCOE reductions? Y/N")
        if lcoe_reductions == "Y":
            self.plot_data_shading(results["LCOE_Reductions"],
                                   results.latitude.values, results.longitude.values,
                                   tick_values=[-100, -75, -50, -25, 0, 25, 50, 75, 100], cmap="seismic",
                                   graphmarking="a",
                                   filename=self.output_folder + "Price_Reductions_Cheapest_Full",
                                   title="Required LCOE reductions: Cheapest Tech (USD/MWh)"
                                   )
            results["LCOE_Reductions"] = xr.where(
                results["Calculated_LCOE"] < results["Benchmark_LCOE"], 99,
                results["Calculated_LCOE"] - results["Benchmark_LCOE"])
            self.plot_data_shading(xr.where(results["LCOE_Reductions"] == 99, 99,
                                            results["LCOE_Reductions"] / results[
                                                "Calculated_LCOE"] * 100),
                                   results.latitude.values, results.longitude.values,
                                   tick_values=[0, 25, 50, 75, 100], cmap="YlOrRd", graphmarking="a",
                                   filename=self.output_folder + "Price_Reductions_CheapestFull%",
                                   special_value=99, hatch_label="Below benchmark",
                                   title="Required grant support (%)"
                                   )
        self.produce_country_scatter_v3(results, "Cheapest")


    
    

 
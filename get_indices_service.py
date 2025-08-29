import os
import datetime
import numpy as np
import geojson
from tqdm.notebook import tqdm

# Sentinel Hub and eo-learn imports
from sentinelhub import (
    SHConfig,
    DataCollection,
    BBox,
    CRS,
)
from eolearn.core import EOTask, FeatureType
from eolearn.io import (
    SentinelHubInputTask,
)  # Using SentinelHubInputTask for idiomatic data fetching
from eolearn.features.ndi import NormalizedDifferenceIndexTask  # Used for NDVI and NDMI

# For plotting maps with polygons
import matplotlib.pyplot as plt
from shapely.geometry import shape


# Define output directory
output_dir = "farm_analysis_results_eolearn_best_practice"
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved in: {os.path.abspath(output_dir)}")


# 2. Core Functions
def authorize_sentinelhub():
    """
    Sets up and verifies Sentinel Hub configuration.
    """
    config = SHConfig()
    if not config.sh_client_id or not config.sh_client_secret:
        print("Sentinel Hub credentials not found in config or environment variables.")
        print("Please ensure SH_CLIENT_ID and SH_CLIENT_SECRET are set.")
        raise SystemExit("Exiting: Sentinel Hub credentials are required.")
    print("Sentinel Hub configuration loaded successfully.")
    return config


def get_or_create_farm_polygon(polygon_path="farm_polygon.geojson"):
    """
    Retrieves a farm polygon from a GeoJSON file or creates a sample one if it doesn't exist.
    Uses the provided 'Test_area_Gedaref' GeoJSON as the sample.
    Returns the GeoJSON geometry object and a Sentinel Hub BBox object.
    """
    if not os.path.exists(polygon_path):
        sample_farm_geojson = {
            "type": "FeatureCollection",
            "name": "Test_area_Gedaref",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
            },
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": 1},
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": [
                            [
                                [
                                    [35.059705226904711, 12.986947971349911],
                                    [35.118561634226467, 13.039248827197715],
                                    [35.194094730861465, 12.98417757904055],
                                    [35.132480092482808, 12.927274341651565],
                                    [35.059705226904711, 12.986947971349911],
                                ]
                            ]
                        ],
                    },
                }
            ],
        }
        with open(polygon_path, "w") as f:
            geojson.dump(sample_farm_geojson, f)
        print(f"Created a sample farm polygon GeoJSON at: {polygon_path}")
        print(
            "NOTE: This uses the 'Test_area_Gedaref' polygon as the default. Modify 'farm_polygon.geojson' for your specific farm."
        )

    with open(polygon_path, "r") as f:
        farm_geojson = geojson.load(f)

    farm_geometry = farm_geojson["features"][0]["geometry"]
    shapely_polygon = shape(farm_geometry)
    minx, miny, maxx, maxy = shapely_polygon.bounds
    bbox_sh = BBox(bbox=(minx, miny, maxx, maxy), crs=CRS.WGS84)

    print(f"Farm polygon loaded/created. Bounding Box: {bbox_sh}")
    return farm_geometry, bbox_sh


class CalculateAllIndices(EOTask):
    """
    EO-Learn Task to calculate NDVI, EVI, MSAVI, and NDMI.
    EVI and MSAVI are calculated manually as their dedicated EOTasks are missing.
    """

    def execute(self, eopatch):
        if FeatureType.DATA not in eopatch or "BANDS" not in eopatch[FeatureType.DATA]:
            raise ValueError(
                "BANDS feature not found in EOPatch. Cannot calculate indices."
            )

        # Sentinel-2 L2A band indices based on the order requested from SentinelHubInputTask (B2, B3, B4, B8, B11)
        # Note: BANDS data is already scaled to reflectance [0-1] by SentinelHubInputTask
        bands = eopatch.data["BANDS"]  # Shape: (time, height, width, bands_count)

        # Extract bands for calculations
        # B2: Blue (idx 0), B3: Green (idx 1), B4: Red (idx 2), B8: NIR (idx 3), B11: SWIR1 (idx 4)
        B2 = bands[..., 0]  # Blue
        B3 = bands[..., 1]  # Green
        B4 = bands[..., 2]  # Red
        B8 = bands[..., 3]  # NIR
        B11 = bands[..., 4]  # SWIR1

        # NDVI: (B8 - B4) / (B8 + B4)
        # Using NormalizedDifferenceIndexTask as it was confirmed to be available.
        ndvi = (
            NormalizedDifferenceIndexTask((FeatureType.DATA, "BANDS", (3, 2)))
            .execute(eopatch)
            .data["NDVI"]
        )
        eopatch.add_feature(FeatureType.DATA, "NDVI", ndvi)

        # MSAVI: (2 * NIR + 1 - sqrt((2 * NIR + 1)^2 - 8 * (NIR - Red))) / 2
        # Insight: Manually calculating MSAVI because the EOTask for it is missing.
        with np.errstate(divide="ignore", invalid="ignore"):
            term_under_sqrt = (2 * B8 + 1) ** 2 - 8 * (B8 - B4)
            msavi = np.where(
                term_under_sqrt >= 0,
                (2 * B8 + 1 - np.sqrt(term_under_sqrt)) / 2,
                np.nan,
            )
        eopatch.add_feature(
            FeatureType.DATA, "MSAVI", msavi[..., np.newaxis]
        )  # Add newaxis to maintain (h,w,1) or (t,h,w,1) shape

        # EVI: 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))
        # Insight: Manually calculating EVI because the EOTask for it is missing.
        with np.errstate(divide="ignore", invalid="ignore"):
            numerator_evi = B8 - B4
            denominator_evi = B8 + 6 * B4 - 7.5 * B2 + 1
            evi = 2.5 * np.where(
                denominator_evi != 0, numerator_evi / denominator_evi, np.nan
            )
        eopatch.add_feature(
            FeatureType.DATA, "EVI", evi[..., np.newaxis]
        )  # Add newaxis to maintain (h,w,1) or (t,h,w,1) shape

        # NDMI (Normalized Difference Moisture Index) / NDWI (Gao's version)
        # Formula: (NIR - SWIR1) / (NIR + SWIR1)
        # Using NormalizedDifferenceIndexTask as it was confirmed to be available.
        ndmi = (
            NormalizedDifferenceIndexTask((FeatureType.DATA, "BANDS", (3, 4)))
            .execute(eopatch)
            .data["NDMI"]
        )
        eopatch.add_feature(FeatureType.DATA, "NDMI", ndmi)

        print("  All specified indices calculated within EOPatch.")
        return eopatch


def run_full_eolearn_processing(bbox, time_interval_str, config, size=(100, 100)):
    """
    Runs a full EO-Learn processing pipeline for a given time interval and bounding box.
    Uses SentinelHubInputTask to fetch data, calculates indices, applies cloud masking,
    and computes median/max values.
    Returns the processed EOPatch.
    """
    start_date, end_date = time_interval_str
    print(f"  Processing EOPatch for interval: {start_date} to {end_date}...")

    input_task = SentinelHubInputTask(
        data_collection=DataCollection.SENTINEL2_L2A,
        bands=["B02", "B03", "B04", "B08", "B11", "SCL"],  # Removed B5 (Red Edge 1)
        bands_feature=(FeatureType.DATA, "BANDS"),
        mosaicking_order="leastRecent",
        resolution=10,
        time_difference=datetime.timedelta(minutes=60),
        config=config,
    )

    calculate_indices_task = CalculateAllIndices()

    # Insight: Reverted to direct task chaining, as LinearWorkflow caused ModuleNotFoundError.
    # This is a robust alternative for sequential operations.
    try:
        # Step 1: Execute input_task to get the EOPatch with raw bands
        eopatch = input_task.execute(bbox=bbox, time_interval=time_interval_str)

        if not eopatch.timestamps:
            print(
                f"  No valid timestamps/scenes found for this interval. Returning None."
            )
            return None

        print(f"  Retrieved {len(eopatch.timestamps)} images in EOPatch.")

        # Step 2: Execute calculate_indices_task on the obtained EOPatch
        eopatch = calculate_indices_task.execute(eopatch)

        # Compute Median and Maximum Values per Index (over time axis) with cloud masking
        print("  Calculating median and maximum values with cloud masking...")

        # SCL (Scene Classification Layer) is extracted from the 'BANDS' feature (index 5)
        # and used for robust cloud/shadow masking.
        scl_mask = eopatch.data["BANDS"][:, :, :, 5].astype(
            np.uint8
        )  # SCL is at index 5 in the BANDS list

        # SCL values to mask: 0-No data, 3-Cloud shadows, 8-Cloud medium probability, 9-Cloud high probability, 10-Thin cirrus
        valid_pixel_mask = ~(
            (scl_mask == 0)
            | (scl_mask == 3)
            | (scl_mask == 8)
            | (scl_mask == 9)
            | (scl_mask == 10)
        )
        valid_pixel_mask = valid_pixel_mask[:, :, :, np.newaxis]

        indices_to_analyze = [
            "NDVI",
            "MSAVI",
            "EVI",
            "NDMI",
        ]  # Insight: List of indices to analyze

        for index_name in indices_to_analyze:
            index_data = eopatch.data[index_name]
            masked_index_data = np.where(valid_pixel_mask, index_data, np.nan)
            median_value = np.nanmedian(masked_index_data, axis=0)
            max_value = np.nanmax(masked_index_data, axis=0)
            eopatch.add_feature(
                FeatureType.DATA,
                f"{index_name}_MEDIAN",
                median_value.astype(np.float32),
            )
            eopatch.add_feature(
                FeatureType.DATA, f"{index_name}_MAX", max_value.astype(np.float32)
            )

            # Insight: Print numerical median and max values to console as requested.
            # Using np.nanmedian for a summary of the whole map's median/max values.
            overall_median = np.nanmedian(median_value)
            overall_max = np.nanmax(max_value)
            print(
                f"    {index_name} ({time_interval_str[0]} to {time_interval_str[1]}): Overall Median={overall_median:.4f}, Overall Max={overall_max:.4f}"
            )

        print(
            "  Median and maximum values calculated for all indices and added to EOPatch."
        )
        return eopatch

    except Exception as e:
        print(
            f"  An error occurred during EO-Learn workflow execution for {time_interval_str}: {e}"
        )
        return None


def define_time_periods():
    """
    Defines and returns time intervals for analysis.
    """
    today = datetime.date.today()
    last_5_days_start = today - datetime.timedelta(days=5)
    last_10_days_start = today - datetime.timedelta(days=10)
    past_3_months_start = today - datetime.timedelta(days=90)

    time_intervals_for_maps = {
        "last_5_days": (last_5_days_start, today),
        "last_10_days": (last_10_days_start, today),
        "past_3_months": (past_3_months_start, today),
    }

    time_intervals_str_for_maps = {
        name: (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        for name, (start, end) in time_intervals_for_maps.items()
    }

    past_3_months_interval_str = (
        past_3_months_start.strftime("%Y-%m-%d"),
        today.strftime("%Y-%m-%d"),
    )

    return time_intervals_str_for_maps, past_3_months_interval_str


def plot_map_with_polygon(
    eopatch, feature_name, farm_geometry, title="", cmap="viridis", vmin=None, vmax=None
):
    """
    Plots a specific feature map directly from an EOPatch, with the farm polygon overlay, saving as PNG.
    Insight: Modified to plot directly from EOPatch data (in-memory), no GeoTIFF export/load needed.
    """
    if FeatureType.DATA not in eopatch or feature_name not in eopatch[FeatureType.DATA]:
        print(f"  Feature '{feature_name}' not found in EOPatch. Cannot plot map.")
        return

    data_to_plot = eopatch.data[feature_name]

    # Ensure data is 2D (height, width) for plotting, remove singleton last dim if present
    if data_to_plot.ndim == 3 and data_to_plot.shape[-1] == 1:
        data_to_plot = data_to_plot[:, :, 0]
    elif data_to_plot.ndim != 2:
        print(
            f"  Error: Data for feature '{feature_name}' has unexpected shape {data_to_plot.shape}. Expected 2D or 3D with last dim 1."
        )
        return

    # Derive extent from EOPatch bbox
    bbox = eopatch.bbox
    extent = [bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y]

    plt.figure(figsize=(10, 10))
    im = plt.imshow(
        data_to_plot, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, origin="upper"
    )

    from matplotlib.patches import Polygon as MplPolygon

    # Robustly handle both Polygon and MultiPolygon geometry types for plotting.
    if farm_geometry["type"] == "MultiPolygon":
        for poly_coords_list in farm_geometry["coordinates"]:
            outer_ring_coords = poly_coords_list[0]
            poly_patch = MplPolygon(
                outer_ring_coords,
                closed=True,
                edgecolor="red",
                facecolor="none",
                linewidth=2,
                alpha=0.7,
            )
            plt.gca().add_patch(poly_patch)
    else:  # Assume Polygon
        polygon_coords = farm_geometry["coordinates"][0]
        poly_patch = MplPolygon(
            polygon_coords,
            closed=True,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
            alpha=0.7,
        )
        plt.gca().add_patch(poly_patch)

    plt.colorbar(im, label="Index Value")
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()

    output_fig = os.path.join(
        output_dir, f"{title.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    )

    # Using current date and time in filename to avoid overwriting previous runs
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_fig_dated = output_fig.replace(".png", f"_{timestamp_str}.png")

    plt.savefig(output_fig_dated, dpi=300)
    print(f"  Saved figure to: {output_fig_dated}")
    plt.close()


def plot_time_series(eopatch_for_ts, output_dir, index_name="NDVI"):
    """
    Generates a time series plot of median values for a specified index
    from an already processed EOPatch (for the entire period).
    Applies cloud masking to each time slice before computing median.
    Insight: This function efficiently extracts time series data from an existing EOPatch,
    avoiding redundant Sentinel Hub requests for each time chunk.
    """
    if (
        not eopatch_for_ts
        or FeatureType.DATA not in eopatch_for_ts
        or index_name not in eopatch_for_ts.data
    ):
        print(
            f"  EOPatch or '{index_name}' data not available for time series plotting."
        )
        return

    print(f"\n--- Generating {index_name} Time Series Plot from EOPatch ---")

    time_series_values = []
    date_labels = []

    index_data_all_times = eopatch_for_ts.data[
        index_name
    ]  # Shape: (time, height, width, 1)
    # SCL is last band in 'BANDS' feature as retrieved by SentinelHubInputTask
    scl_data_all_times = eopatch_for_ts.data["BANDS"][:, :, :, 5].astype(
        np.uint8
    )  # SCL is at index 5 in the BANDS list

    for i, timestamp in enumerate(eopatch_for_ts.timestamps):
        current_index_map = index_data_all_times[
            i, :, :, 0
        ]  # Remove last dimension if present
        current_scl_map = scl_data_all_times[i, :, :]

        # SCL values to mask: 0-No data, 3-Cloud shadows, 8-Cloud medium probability, 9-Cloud high probability, 10-Thin cirrus
        valid_mask = ~(
            (current_scl_map == 0)
            | (current_scl_map == 3)
            | (current_scl_map == 8)
            | (current_scl_map == 9)
            | (current_scl_map == 10)
        )

        masked_index_data = np.where(valid_mask, current_index_map, np.nan)

        if not np.all(np.isnan(masked_index_data)):
            median_val = np.nanmedian(masked_index_data)
            time_series_values.append(median_val)
        else:
            time_series_values.append(np.nan)  # Append NaN if entire image is masked

        date_labels.append(timestamp.strftime("%Y-%m-%d"))

    if time_series_values:
        plt.figure(figsize=(12, 5))
        # Use different colors for different indices in time series
        if index_name == "NDVI":
            color = "green"
        elif index_name == "MSAVI":
            color = "purple"
        elif index_name == "EVI":
            color = "blue"
        elif index_name == "NDMI":
            color = "cyan"
        else:
            color = "gray"  # Fallback for any other index

        plt.plot(
            date_labels, time_series_values, marker="o", linestyle="-", color=color
        )
        plt.title(f"{index_name} Time Series (Median per Scene - Full Period)")
        plt.xlabel("Date")
        plt.ylabel(f"Median {index_name}")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        output_fig_path = os.path.join(
            output_dir, f"{index_name}_time_series_full_period.png"
        )

        # Using current date and time in filename to avoid overwriting previous runs
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_fig_dated = output_fig_path.replace(".png", f"_{timestamp_str}.png")

        plt.savefig(output_fig_dated, dpi=300)
        print(f"Saved {index_name} time series plot to: {output_fig_dated}")
        plt.close()
    else:
        print(f"No valid data points collected for {index_name} time series plot.")


# 3. Main Execution Workflow

# 1. Authorize Sentinel Hub
sh_config = authorize_sentinelhub()

# 2. Get or Create Farm Polygon
farm_geometry, farm_bbox = get_or_create_farm_polygon()

# 3. Define Time Periods
time_intervals_str_for_maps, past_3_months_interval_str = define_time_periods()

# Insight: List of indices to analyze (NDVI, MSAVI, EVI, NDMI) based on conditional
indices_to_analyze = ["NDVI", "MSAVI", "EVI", "NDMI"]

# Store EOPatches for potential later use
processed_eopatches = {}

# 4. Process Indices and Generate Maps for Each Timeframe
print("\n--- Starting Main Index Processing and Map Generation ---")
for timeframe_name, interval_str in tqdm(
    time_intervals_str_for_maps.items(), desc="Overall Processing"
):
    print(f"\nProcessing data for {timeframe_name}...")
    # Call the main EO-Learn processing function which fetches data and calculates all
    # indices efficiently into a single EOPatch for the entire interval.
    eopatch_result = run_full_eolearn_processing(farm_bbox, interval_str, sh_config)

    if eopatch_result:
        processed_eopatches[timeframe_name] = eopatch_result
        # Generate maps (PNG figures) for each index
        for index_name in indices_to_analyze:
            median_feature_name = f"{index_name}_MEDIAN"
            max_feature_name = f"{index_name}_MAX"

            # Plot Median Map (PNG)
            plot_map_with_polygon(
                eopatch_result,  # Pass EOPatch directly
                median_feature_name,  # Pass feature name
                farm_geometry,
                title=f"{index_name} Median ({timeframe_name}) with Farm Polygon",
                # Adjust vmin/vmax for specific indices if needed
                vmin=0.0 if index_name in ["NDVI", "EVI", "MSAVI"] else -1.0,
                vmax=1.0,
            )

            # Plot Max Map (PNG)
            plot_map_with_polygon(
                eopatch_result,  # Pass EOPatch directly
                max_feature_name,  # Pass feature name
                farm_geometry,
                title=f"{index_name} Max ({timeframe_name}) with Farm Polygon",
                vmin=0.0 if index_name in ["NDVI", "EVI", "MSAVI"] else -1.0,
                vmax=1.0,
            )
    else:
        print(f"Skipping map generation for {timeframe_name} due to processing issues.")

# 5. Generate Time Series Plot (for NDVI over past 3 months as requested)
print("\n--- Starting Time Series Plot Generation ---")
# The time series plot directly uses the already processed EOPatch
# for the 'past_3_months' period, avoiding redundant Sentinel Hub calls.
# You can change 'NDVI' to 'MSAVI', 'EVI', or 'NDMI' to plot their time series.
if (
    "past_3_months" in processed_eopatches
    and processed_eopatches["past_3_months"] is not None
):
    plot_time_series(
        processed_eopatches["past_3_months"], output_dir, index_name="NDVI"
    )
    # Optional: Plot other indices time series as well, for example:
    # plot_time_series(processed_eopatches["past_3_months"], output_dir, index_name='MSAVI')
    # plot_time_series(processed_eopatches["past_3_months"], output_dir, index_name='EVI')
    # plot_time_series(processed_eopatches["past_3_months"], output_dir, index_name='NDMI')
else:
    print(
        "Cannot generate time series plot: No valid EOPatch available for 'past_3_months'."
    )


print(
    "\n--- All processing complete! Check the 'farm_analysis_results_eolearn_best_practice' directory. ---"
)

import pandas as pd
import folium
from folium.plugins import MarkerCluster
import webbrowser
from typing import Tuple


def load_station_csv(path="station.csv"):
    """Load the station CSV into a pandas DataFrame."""
    df = pd.read_csv(path, low_memory=False)
    return df


def filter_water_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame for likely water-quality monitoring sites and remove duplicates.

    Heuristics used:
    - Keep rows where `ProviderName` looks like a water-quality provider (STORET, WQX, NWIS).
    - Or where the organization name contains 'Water'.
    - Remove duplicate stations using `MonitoringLocationIdentifier` when available.
    """
    # Defensive: ensure columns exist
    cols = df.columns
    provider = df["ProviderName"] if "ProviderName" in cols else pd.Series([], dtype=object)
    orgname = df["OrganizationFormalName"] if "OrganizationFormalName" in cols else pd.Series([], dtype=object)

    mask = (
        provider.astype(str).str.contains(r"STORET|WQX|WQ|NWIS", case=False, na=False)
        | orgname.astype(str).str.contains(r"Water", case=False, na=False)
    )

    filtered = df[mask].copy()

    # Normalize latitude/longitude column names used here
    lat_col = "LatitudeMeasure"
    lon_col = "LongitudeMeasure"

    # Drop rows without coordinates
    if lat_col in filtered.columns and lon_col in filtered.columns:
        filtered = filtered.dropna(subset=[lat_col, lon_col])
        filtered[lat_col] = pd.to_numeric(filtered[lat_col], errors="coerce")
        filtered[lon_col] = pd.to_numeric(filtered[lon_col], errors="coerce")
        filtered = filtered.dropna(subset=[lat_col, lon_col])

    # Remove duplicate stations by identifier if present, else by (lat, lon, name)
    if "MonitoringLocationIdentifier" in filtered.columns:
        filtered = filtered.drop_duplicates(subset=["MonitoringLocationIdentifier"])
    else:
        subset = [c for c in [lat_col, lon_col, "MonitoringLocationName"] if c in filtered.columns]
        if subset:
            filtered = filtered.drop_duplicates(subset=subset)

    return filtered


def get_station_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of unique station locations with columns:
    `MonitoringLocationIdentifier`, `MonitoringLocationName`, `Latitude`, `Longitude`.

    Ensures each station appears once by using `MonitoringLocationIdentifier` when
    available, otherwise deduplicating by name+lat+lon.
    """
    lat_col = "LatitudeMeasure"
    lon_col = "LongitudeMeasure"
    id_col = "MonitoringLocationIdentifier"
    name_col = "MonitoringLocationName"

    # Ensure latitude/longitude exist
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError("DataFrame must contain LatitudeMeasure and LongitudeMeasure columns")

    cols = [c for c in [id_col, name_col, lat_col, lon_col] if c in df.columns]
    locations = df[cols].copy()

    # Normalize numeric types and drop missing coords
    locations[lat_col] = pd.to_numeric(locations[lat_col], errors="coerce")
    locations[lon_col] = pd.to_numeric(locations[lon_col], errors="coerce")
    locations = locations.dropna(subset=[lat_col, lon_col])

    # Deduplicate
    if id_col in locations.columns and locations[id_col].notna().any():
        locations = locations.drop_duplicates(subset=[id_col])
    else:
        dedup_subset = [c for c in [name_col, lat_col, lon_col] if c in locations.columns]
        locations = locations.drop_duplicates(subset=dedup_subset)

    # Rename lat/lon to simple names
    locations = locations.rename(columns={lat_col: "Latitude", lon_col: "Longitude"})
    return locations[[c for c in [id_col, name_col, "Latitude", "Longitude"] if c in locations.columns]]


def plot_and_show_map(locations: pd.DataFrame, out_html: str = "station_map.html", open_in_browser: bool = True):
    """Create a folium map from `locations`, save it to `out_html`, and display inline if
    running inside IPython/Jupyter. Falls back to opening the HTML in the default browser.
    `locations` must contain `Latitude` and `Longitude` columns.
    """
    if "Latitude" not in locations.columns or "Longitude" not in locations.columns:
        raise ValueError("locations DataFrame must contain 'Latitude' and 'Longitude' columns")

    center_lat = locations["Latitude"].astype(float).mean()
    center_lon = locations["Longitude"].astype(float).mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    cluster = MarkerCluster().add_to(m)

    id_col = "MonitoringLocationIdentifier" if "MonitoringLocationIdentifier" in locations.columns else None
    name_col = "MonitoringLocationName" if "MonitoringLocationName" in locations.columns else None

    for _, row in locations.iterrows():
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        name = row.get(name_col, "(no name)") if name_col else "(no name)"
        mid = row.get(id_col, "") if id_col else ""
        popup = f"{name} <br> {mid}"
        folium.Marker(location=[lat, lon], popup=popup).add_to(cluster)

    m.save(out_html)

    # Try to display inline in IPython environments (Jupyter/Colab). If not available,
    # open the saved HTML in the default browser.
    try:
        from IPython.display import HTML, display  # type: ignore

        display(HTML(m._repr_html_()))
    except Exception:
        if open_in_browser:
            webbrowser.open(out_html)

    return m


def make_station_map(df: pd.DataFrame, out_html: str = "station_map.html") -> folium.Map:
    """Create a folium map with station markers and save to `out_html`. Returns the folium.Map object."""
    lat_col = "LatitudeMeasure"
    lon_col = "LongitudeMeasure"

    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError("Latitude or longitude columns not found in DataFrame")

    # Compute center
    center_lat = df[lat_col].astype(float).mean()
    center_lon = df[lon_col].astype(float).mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        name = row.get("MonitoringLocationName", "(no name)")
        mid = row.get("MonitoringLocationIdentifier", "")
        popup = f"{name} <br> {mid}"
        folium.Marker(location=[lat, lon], popup=popup).add_to(marker_cluster)

    m.save(out_html)
    return m


def main():
    df = load_station_csv("station.csv")

    # Display first 10 rows and column names
    print("Columns:")
    print(list(df.columns))
    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    filtered = filter_water_quality(df)
    print(f"\nFiltered water-quality stations: {len(filtered)} rows")

    out_html = "station_map.html"
    make_station_map(filtered, out_html=out_html)
    print(f"Map saved to {out_html}")


if __name__ == "__main__":
    main()

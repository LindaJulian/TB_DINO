from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tb_dino.config import load_config
from tb_dino.utils import write_json


def _load_aoi(ee, path: str):
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as file:
        geojson = json.load(file)
    geo_type = geojson.get("type")
    if geo_type == "FeatureCollection":
        return ee.FeatureCollection(geojson).geometry()
    if geo_type == "Feature":
        return ee.Feature(geojson).geometry()
    return ee.Geometry(geojson)


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaEarth")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    export = config.get("alphaearth_export", config)
    try:
        import ee
    except ImportError as exc:
        raise SystemExit(
            "The Earth Engine API has not been installed"
        ) from exc

    initialize_kwargs = {}
    if export.get("project"):
        initialize_kwargs["project"] = export["project"]
    try:
        ee.Initialize(**initialize_kwargs)
    except Exception as exc:
        raise SystemExit(
            "Earth Engine Initialization failed"
            "And confirm that the project configuration is correct"
        ) from exc

    geometry = _load_aoi(ee, export["aoi"])
    collection_id = export.get(
        "collection", "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    )
    bands = [f"A{index:02d}" for index in range(64)]
    destination = str(export.get("destination", "drive")).lower()
    tasks = []
    for year in [int(value) for value in export["years"]]:
        image = (
            ee.ImageCollection(collection_id)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .mosaic()
            .select(bands)
            .clip(geometry)
        )
        common = {
            "image": image,
            "description": f"alphaearth_{year}",
            "fileNamePrefix": f"alphaearth_{year}",
            "region": geometry,
            "scale": float(export.get("scale", 10)),
            "maxPixels": int(export.get("max_pixels", 1_000_000_000_000)),
            "fileFormat": "GeoTIFF",
        }
        if export.get("crs"):
            common["crs"] = export["crs"]
        if destination == "drive":
            task = ee.batch.Export.image.toDrive(
                folder=export.get("drive_folder", "TB_DINO_AlphaEarth"),
                **common,
            )
        elif destination in {"gcs", "cloud_storage"}:
            task = ee.batch.Export.image.toCloudStorage(
                bucket=export["bucket"],
                **common,
            )
        else:
            raise ValueError("destination  drive or gcs")
        task.start()
        tasks.append({"year": year, "task_id": task.id, "state": "SUBMITTED"})
        print(f"Submitted {year}: {task.id}")
    task_file = Path(export.get("task_file", "alphaearth_tasks.json"))
    write_json(tasks, task_file)


if __name__ == "__main__":
    main()


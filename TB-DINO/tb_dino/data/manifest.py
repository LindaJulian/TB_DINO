from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class ManifestRecord:
    id: str
    images: list[str]
    timestamps: list[str]
    mask: str
    alpha_by_year: dict[str, str] = field(default_factory=dict)
    valid_mask: str | None = None
    bands: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any], base_dir: Path) -> "ManifestRecord":
        required = ("id", "images", "timestamps", "mask")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"manifest 记录缺少字段 {missing}: {value}")
        images = list(value["images"])
        timestamps = [str(item) for item in value["timestamps"]]
        if not images:
            raise ValueError(f"记录 {value['id']} 不包含影像")
        if len(images) != len(timestamps):
            raise ValueError(
                f"记录 {value['id']} 的 images 与 timestamps 数量不一致"
            )

        def resolve(path: str | None) -> str | None:
            if path is None:
                return None
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            return str(candidate.resolve())

        alpha = {
            str(year): str(resolve(path))
            for year, path in (value.get("alpha_by_year") or {}).items()
        }
        known = {
            "id",
            "images",
            "timestamps",
            "mask",
            "alpha_by_year",
            "valid_mask",
            "bands",
        }
        metadata = {key: item for key, item in value.items() if key not in known}
        return cls(
            id=str(value["id"]),
            images=[str(resolve(path)) for path in images],
            timestamps=timestamps,
            mask=str(resolve(value["mask"])),
            alpha_by_year=alpha,
            valid_mask=resolve(value.get("valid_mask")),
            bands=list(value["bands"]) if value.get("bands") is not None else None,
            metadata=metadata,
        )

    def to_dict(self, relative_to: Path | None = None) -> dict[str, Any]:
        def serialize(path: str | None) -> str | None:
            if path is None:
                return None
            source = Path(path)
            if relative_to is not None:
                try:
                    return source.resolve().relative_to(relative_to.resolve()).as_posix()
                except ValueError:
                    pass
            return str(source)

        result: dict[str, Any] = {
            "id": self.id,
            "images": [serialize(path) for path in self.images],
            "timestamps": self.timestamps,
            "mask": serialize(self.mask),
        }
        if self.alpha_by_year:
            result["alpha_by_year"] = {
                year: serialize(path) for year, path in self.alpha_by_year.items()
            }
        if self.valid_mask:
            result["valid_mask"] = serialize(self.valid_mask)
        if self.bands is not None:
            result["bands"] = self.bands
        result.update(self.metadata)
        return result

    def validate_files(self, require_alpha: bool = False) -> None:
        paths = [*self.images, self.mask]
        if self.valid_mask:
            paths.append(self.valid_mask)
        missing = [path for path in paths if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"记录 {self.id} 存在缺失文件: {missing}")
        if require_alpha:
            years = {timestamp[:4] for timestamp in self.timestamps}
            absent = sorted(year for year in years if year not in self.alpha_by_year)
            if absent:
                raise ValueError(f"记录 {self.id} 缺少年度 AlphaEarth: {absent}")
        alpha_missing = [
            path for path in self.alpha_by_year.values() if not Path(path).exists()
        ]
        if alpha_missing:
            raise FileNotFoundError(
                f"记录 {self.id} 存在缺失 AlphaEarth 文件: {alpha_missing}"
            )


def read_manifest(
    path: str | Path,
    validate_files: bool = False,
    require_alpha: bool = False,
) -> list[ManifestRecord]:
    path = Path(path).expanduser().resolve()
    records: list[ManifestRecord] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                record = ManifestRecord.from_dict(raw, path.parent)
            except Exception as exc:
                raise ValueError(f"无法解析 {path}:{line_number}: {exc}") from exc
            if validate_files:
                record.validate_files(require_alpha=require_alpha)
            records.append(record)
    if not records:
        raise ValueError(f"manifest 为空: {path}")
    return records


def write_manifest(
    records: Iterable[ManifestRecord],
    path: str | Path,
    relative_paths: bool = True,
) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            payload = record.to_dict(path.parent if relative_paths else None)
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


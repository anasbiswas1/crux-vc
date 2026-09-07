"""Path discovery and repository layout."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls, start: str | Path | None = None) -> "ProjectPaths":
        candidates: list[Path] = []
        env = os.environ.get("CRUX_REPO_ROOT")
        if env:
            candidates.append(Path(env))
        if start is not None:
            candidates.append(Path(start))
        candidates.append(Path.cwd())
        candidates.append(Path("/content/drive/MyDrive/CRUX_Research/crux-vc"))

        checked: set[Path] = set()
        for candidate in candidates:
            try:
                candidate = candidate.expanduser().resolve()
            except FileNotFoundError:
                candidate = candidate.expanduser().absolute()
            for current in (candidate, *candidate.parents):
                if current in checked:
                    continue
                checked.add(current)
                if (current / ".cruxvc-root").exists():
                    return cls(current)
        raise FileNotFoundError(
            "CRUX-VC repository root not found. Set CRUX_REPO_ROOT or place the repository "
            "at /content/drive/MyDrive/CRUX_Research/crux-vc."
        )

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def protocol(self) -> Path:
        return self.root / "protocol"

    @property
    def locks(self) -> Path:
        return self.protocol / "locks"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def external(self) -> Path:
        return self.data / "external"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def manifests(self) -> Path:
        return self.results / "manifests"

    @property
    def audits(self) -> Path:
        return self.results / "audits"

    @property
    def models(self) -> Path:
        return self.results / "models"

    @property
    def predictions(self) -> Path:
        return self.results / "predictions"

    @property
    def attributions(self) -> Path:
        return self.results / "attributions"

    @property
    def inference(self) -> Path:
        return self.results / "inference"

    @property
    def controls(self) -> Path:
        return self.results / "controls"

    @property
    def decision(self) -> Path:
        return self.results / "decision"

    @property
    def selective(self) -> Path:
        return self.results / "selective"

    @property
    def release(self) -> Path:
        return self.results / "release"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    def ensure(self) -> "ProjectPaths":
        for path in (
            self.config,
            self.docs,
            self.protocol,
            self.locks,
            self.raw,
            self.interim,
            self.processed,
            self.external,
            self.manifests,
            self.audits,
            self.models,
            self.predictions,
            self.attributions,
            self.inference,
            self.controls,
            self.decision,
            self.selective,
            self.release,
            self.figures,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def relative(self, path: str | Path) -> str:
        path = Path(path).resolve()
        return str(path.relative_to(self.root.resolve()))

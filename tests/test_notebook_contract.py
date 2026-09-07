from __future__ import annotations

import ast
from pathlib import Path

import nbformat
import yaml

EXPECTED = [f"{index:02d}_" for index in range(21)]


def test_all_notebooks_have_clean_machine_readable_contract_metadata():
    notebooks = sorted(Path("notebooks").glob("*.ipynb"))
    contracts = yaml.safe_load(Path("config/notebook_contracts.yaml").read_text(encoding="utf-8"))
    assert len(notebooks) == 21
    for index, path in enumerate(notebooks):
        stage = f"{index:02d}"
        assert path.name.startswith(EXPECTED[index])
        notebook = nbformat.read(path, as_version=4)
        contract = notebook.metadata.get("cruxvc", {})
        expected = contracts[stage]
        assert contract.get("stage_id") == stage
        assert "final_test_access" in contract
        assert contract.get("contract_ref") == f"config/notebook_contracts.yaml#{stage}"
        assert list(contract.get("permitted_blocks", [])) == list(expected["permitted_blocks"])
        assert list(contract.get("declared_inputs", [])) == list(expected["inputs"])
        assert list(contract.get("declared_outputs", [])) == list(expected["outputs"])
        assert list(contract.get("assertions", [])) == list(expected["assertions"])
        for cell in notebook.cells:
            assert cell.get("id")
            if cell.cell_type == "code":
                assert cell.execution_count is None
                assert not cell.outputs
                ast.parse(cell.source)

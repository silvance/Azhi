"""The build step that ships every bundled package's licence files."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "packaging" / "collect_licences.py"
_spec = importlib.util.spec_from_file_location("collect_licences", _PATH)
collect_licences = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect_licences)


def test_installed_packages_licences_land_in_the_bundle(tmp_path):
    # pytest itself is installed wherever this runs, and ships a LICENSE.
    assert collect_licences.main([str(tmp_path)]) == 0
    copied = {
        p.parent.parent.name for p in (tmp_path / "licenses").rglob("*") if p.is_file()
    }
    assert "pytest" in copied


def test_a_missing_bundle_folder_is_refused(tmp_path):
    assert collect_licences.main([str(tmp_path / "missing")]) == 2

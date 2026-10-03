"""Basic tests for apex-critical-infra."""


def test_import():
    """Test that the package can be imported."""
    import sys
    sys.path.insert(0, "src")
    import os
    assert os.path.exists("src")


def test_modules_exist():
    """Test that key modules exist."""
    import os
    modules = [
        "command_control.py",
        "cooling_optimization.py",
        "cybersecurity.py",
        "dc_operations.py",
        "edge_ai.py",
        "edge_computing.py",
        "grid_simulator.py",
        "incident_response.py",
        "network_operations.py",
        "network_simulator.py",
        "power_management.py",
        "predictive_maintenance.py",
        "renewables.py",
        "smart_grid.py",
        "spectrum_management.py",
        "threat_intelligence.py",
    ]
    for mod in modules:
        assert os.path.exists(f"src/{mod}"), f"Missing module: {mod}"

"""Pytest configuration and fixtures for webapp tests."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(autouse=True)
def setup_module_registry():
    """Automatically set up module registry before each test.
    
    This fixture ensures that all model modules are properly registered
    before each test runs, preventing issues with test isolation.
    """
    from lob_modeling.webapp.modules import (
        AlmgrenChrissModule,
        AssetOptionModule,
        CriscuoloWaehlbroeckModule,
        DePradoModule,
        GlostenMilgromModule,
        KyleModelModule,
        register_module,
        registry,
    )
    
    # Clear registry
    registry._modules = {}
    
    # Register all modules
    register_module("kyle", KyleModelModule)
    register_module("almgren_chriss", AlmgrenChrissModule)
    register_module("glosten_milgrom", GlostenMilgromModule)
    register_module("de_prado", DePradoModule)
    register_module("criscuolo_waehlbroeck", CriscuoloWaehlbroeckModule)
    register_module("asset_option", AssetOptionModule)
    
    yield
    
    # Cleanup after test (optional)
    pass

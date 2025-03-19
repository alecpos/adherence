"""Basic tests for the adherence package."""

def test_import():
    """Test that the package can be imported."""
    try:
        import adherence
        assert True
    except ImportError:
        assert False, "Failed to import adherence package" 
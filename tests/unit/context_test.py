from cyantic.context import ValidationContext


def test_validation_context():
    """Test basic ValidationContext functionality."""
    test_data = {"foo": "bar", "nested": {"value": 42}}

    # Context shouldn't exist outside the manager
    assert ValidationContext.get_root_data() is None

    with ValidationContext.root_data(test_data):
        # Inside the context manager, we can access the data
        assert ValidationContext.get_root_data() == test_data

        # Test nested value access
        assert ValidationContext.get_nested_value("foo") == "bar"
        assert ValidationContext.get_nested_value("nested.value") == 42

        # Test path management
        assert ValidationContext.get_current_path() == ""

        ValidationContext.push_path("model")
        assert ValidationContext.get_current_path() == "model"

        ValidationContext.push_path("field")
        assert ValidationContext.get_current_path() == "model.field"

        ValidationContext.pop_path()
        assert ValidationContext.get_current_path() == "model"

        ValidationContext.pop_path()
        assert ValidationContext.get_current_path() == ""

    # After the context manager exits, the context is cleaned up
    assert ValidationContext.get_root_data() is None

    # Test errors outside context
    try:
        ValidationContext.get_nested_value("foo")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

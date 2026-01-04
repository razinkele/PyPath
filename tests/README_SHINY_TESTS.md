# PyPath Shiny Dashboard Tests

Comprehensive test suite for the PyPath Shiny dashboard application.

## Test Files

### `test_shiny_app.py`
Core application structure and integration tests:
- **TestAppStructure**: App imports, constants, and static assets
- **TestUIComponents**: UI layout, navigation, custom CSS, Bootstrap Icons
- **TestServerLogic**: Server function, SharedData class, reactive state management
- **TestErrorHandling**: Error recovery in server initialization
- **TestDataFlow**: Data propagation between model_data and sim_results
- **TestNavigationStructure**: Page navigation and module structure
- **TestThemeAndSettings**: Theme picker and settings functionality
- **TestDocumentation**: Code documentation and docstrings
- **TestIntegrationScenarios**: End-to-end workflow tests

### `test_shiny_pages.py`
Individual page module tests:
- **TestHomePage**: Home page UI and server functions
- **TestDataImportPage**: Data import page structure
- **TestEcopathPage**: Ecopath model page
- **TestEcosimPage**: Ecosim simulation page
- **TestResultsPage**: Results visualization page
- **TestAnalysisPage**: Analysis page
- **TestAboutPage**: About/documentation page
- **TestMultiStanzaPage**: Multi-stanza groups feature
- **TestEcospacePage**: Ecospace spatial modeling
- **TestDemoPages**: Demonstration pages (forcing, diet rewiring, optimization)
- **TestPageConsistency**: Naming conventions and consistency
- **TestUtilsModule**: Shared utilities
- **TestPageInteractions**: Data flow between pages

### `test_shiny_reactive.py`
Reactive behavior and state management tests:
- **TestReactiveValues**: Basic reactive value creation and updates
- **TestSharedDataReactivity**: SharedData reactivity patterns
- **TestDataPropagation**: Data propagation through reactive values
- **TestReactiveIsolation**: Independence of reactive values
- **TestComplexDataStructures**: Complex data in reactive values
- **TestReactiveErrorHandling**: Error handling in reactive contexts
- **TestMultipleReactiveEffects**: Multiple watchers on same value
- **TestReactivePerformance**: Performance with large data

## Running Tests

### Run all Shiny tests:
```bash
pytest tests/test_shiny_*.py -v
```

### Run specific test file:
```bash
pytest tests/test_shiny_app.py -v
pytest tests/test_shiny_pages.py -v
pytest tests/test_shiny_reactive.py -v
```

### Run specific test class:
```bash
pytest tests/test_shiny_app.py::TestAppStructure -v
pytest tests/test_shiny_pages.py::TestHomePage -v
pytest tests/test_shiny_reactive.py::TestReactiveValues -v
```

### Run specific test:
```bash
pytest tests/test_shiny_app.py::TestAppStructure::test_app_imports -v
```

### Run with coverage:
```bash
pytest tests/test_shiny_*.py --cov=app --cov-report=html
```

## Test Dependencies

These tests require:
- `pytest` - Testing framework
- `shiny` - Shiny for Python
- `shinyswatch` - Theme picker
- `pandas` - Data structures
- `numpy` - Numerical operations

Most tests will skip gracefully if Shiny is not installed.

## Test Strategy

### Unit Tests
- Test individual components in isolation
- Mock dependencies where needed
- Fast execution, high coverage

### Integration Tests
- Test data flow between pages
- Test reactive state management
- Test typical user workflows

### Structural Tests
- Verify naming conventions
- Check function signatures
- Ensure consistent patterns

### Performance Tests
- Test with large DataFrames
- Test frequent updates
- Verify reasonable performance

## Test Coverage

### What's Covered
✅ App structure and imports
✅ UI component generation
✅ Server initialization
✅ Reactive value behavior
✅ Data flow between pages
✅ SharedData synchronization
✅ Error handling
✅ Theme and settings
✅ Navigation structure
✅ Page module consistency
✅ Complex data structures
✅ Performance characteristics

### What's Not Covered (Browser-level testing)
- Actual browser rendering
- User interactions (clicks, form inputs)
- JavaScript behavior
- Real-time reactivity in browser
- Visual regression

For browser-level testing, consider using:
- Playwright for Python
- Selenium
- Shiny's upcoming testing tools

## Writing New Tests

### Test Template
```python
class TestNewFeature:
    """Tests for new feature."""

    def test_feature_exists(self):
        """Test that feature exists."""
        try:
            from pages import new_feature
            assert hasattr(new_feature, 'feature_ui')
            assert hasattr(new_feature, 'feature_server')
        except ImportError:
            pytest.skip("Module not available")

    def test_feature_behavior(self):
        """Test feature behavior."""
        try:
            from shiny import reactive

            # Create test data
            data = reactive.Value(None)

            # Test behavior
            data.set("test")
            assert data() == "test"
        except ImportError:
            pytest.skip("Shiny not installed")
```

### Best Practices
1. **Use `pytest.skip`** for missing dependencies
2. **Test both UI and server** functions
3. **Mock external dependencies** (databases, APIs)
4. **Test error conditions** not just happy paths
5. **Keep tests focused** - one concept per test
6. **Use descriptive names** - test names explain what they test
7. **Add docstrings** - explain what the test verifies

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Shiny Dashboard Tests
  run: |
    pip install -e .[dev]
    pytest tests/test_shiny_*.py -v --cov=app
```

## Troubleshooting

### "Shiny not installed" errors
Install dependencies:
```bash
pip install shiny shinyswatch
```

### Import errors for page modules
Make sure you're running from the project root:
```bash
cd /path/to/PyPath
pytest tests/test_shiny_app.py
```

### Tests pass locally but fail in CI
Check Python version compatibility and ensure all dependencies are in `requirements.txt`

## Future Enhancements

Potential additions to test suite:
- [ ] Browser-based integration tests with Playwright
- [ ] Visual regression tests
- [ ] Accessibility tests
- [ ] Performance benchmarks
- [ ] Load testing for concurrent users
- [ ] API endpoint tests (if added)
- [ ] Database integration tests
- [ ] User authentication tests (if added)

## Related Documentation

- [Shiny for Python Docs](https://shiny.posit.co/py/)
- [pytest Documentation](https://docs.pytest.org/)
- [PyPath Main README](../README.md)
- [Deployment Guide](../deploy/README.md)

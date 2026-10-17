import pytest


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Report a fixture's deliberate pytest.fail (missing results.json, stub not implemented) as FAILED, not ERROR."""
    report = yield
    if report.when == "setup" and report.failed and call.excinfo and call.excinfo.errisinstance(pytest.fail.Exception):
        report.when = "call"
    return report

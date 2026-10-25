def pytest_configure(config):
    config.addinivalue_line("markers", "slow: долгие тесты (замер скорости)")

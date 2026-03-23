# Selenium GUI test - skipped in CI
import pytest

class Test11:
    @pytest.mark.skip(reason="Selenium GUI - CI not supported")
    def test_11(self):
        pass

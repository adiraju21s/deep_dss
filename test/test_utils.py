import pytest

from deep_dss.utils import path_cl


class TestUtils:
    def test_path_cl(self):
        assert path_cl(0.5) == "../data/flask/input/dss-20-0.28-0.5-1.54Cl-f1z1f1z1.dat"
        assert path_cl(0.856999999999, "f2z1f2z2") == "../data/flask/input/dss-20-0.28-0.857-1.54Cl-f2z1f2z2.dat"


if __name__ == '__main__':
    pytest.main()

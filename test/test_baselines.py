import pytest

from deep_dss.utils import path_to_cl


class TestUtils:
    def test_path_to_cl(self):
        assert path_to_cl(0.5) == "../data/flask/input/dss-20-0.28-0.5-1.54Cl-f1z1f1z1.dat"
        assert path_to_cl(0.856999999999, name="f2z1f2z2") == "../data/flask/input/dss-20-0.28-0.857-1.54Cl-f2z1f2z2.dat"


if __name__ == '__main__':
    pytest.main()

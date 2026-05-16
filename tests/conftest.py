import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def sample_student_info():
    import pandas as pd
    return pd.DataFrame({
        "code_module":       ["AAA", "AAA", "BBB"],
        "code_presentation": ["2014J", "2014J", "2014B"],
        "id_student":        [1, 2, 3],
        "gender":            ["M", "F", "M"],
        "region":            ["Scotland", "London Region", "North Region"],
        "highest_education": ["A Level or Equivalent", "HE Qualification", "Lower Than A Level"],
        "imd_band":          ["0-10%", "10-20", "20-30%"],
        "age_band":          ["0-35", "35-55", "0-35"],
        "num_of_prev_attempts": [0, 1, 0],
        "studied_credits":   [60, 120, 60],
        "disability":        ["N", "N", "Y"],
        "final_result":      ["Pass", "Withdrawn", "Distinction"],
    })


@pytest.fixture
def sample_vle():
    import pandas as pd
    return pd.DataFrame({
        "code_module":       ["AAA"]*6,
        "code_presentation": ["2014J"]*6,
        "id_student":        [1, 1, 1, 2, 2, 2],
        "id_site":           [100, 101, 102, 100, 100, 103],
        "date":              [1, 4, 9, 2, 5, 12],
        "sum_click":         [3, 5, 2, 7, 1, 4],
    })

import pandas as pd
import pytest

from src.preprocessing import Cleaner, deduplicate, load_schema, SchemaValidator
from src.preprocessing.cleaner import encode_final_result


def test_trim_whitespace_and_drop_nulls():
    df = pd.DataFrame({"id_student": [1, None, 3], "name": [" alice ", "bob", "  cara"]})
    cleaner = Cleaner(critical_cols=["id_student"])
    out = cleaner.apply(df)
    assert len(out) == 2
    assert out["name"].iloc[0] == "alice"


def test_deduplicate_keeps_last():
    df = pd.DataFrame({"k": [1, 1, 2], "v": ["a", "b", "c"]})
    out = deduplicate(df, keys=["k"], keep="last")
    assert len(out) == 2
    assert out.loc[out.k == 1, "v"].iloc[0] == "b"


def test_encode_final_result(sample_student_info):
    out = encode_final_result(sample_student_info)
    expected = [0, 1, 0]
    assert out["final_result_binary"].tolist() == expected


def test_schema_validator_passes(sample_student_info):
    schema = load_schema("configs/schemas/student_info.yaml")
    SchemaValidator(schema).validate(sample_student_info)


def test_schema_validator_rejects_bad_allowed_value(sample_student_info):
    bad = sample_student_info.copy()
    bad.loc[0, "gender"] = "X"
    schema = load_schema("configs/schemas/student_info.yaml")
    with pytest.raises(ValueError):
        SchemaValidator(schema).validate(bad)

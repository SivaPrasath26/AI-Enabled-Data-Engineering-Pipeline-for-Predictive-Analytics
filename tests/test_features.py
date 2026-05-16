import pandas as pd

from src.features.factory import FeatureFactory
from src.preprocessing.cleaner import encode_final_result


def _empty_assessments():
    return pd.DataFrame(columns=["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"])


def _empty_student_assessment():
    return pd.DataFrame(columns=["id_assessment", "id_student", "date_submitted", "is_banked", "score"])


def test_factory_builds_feature_matrix(sample_student_info, sample_vle):
    si = encode_final_result(sample_student_info)
    reg = sample_student_info[["id_student", "code_module", "code_presentation"]].copy()
    reg["date_registration"] = -30
    reg["date_unregistration"] = None

    factory = FeatureFactory(target_horizon_weeks=4)
    out = factory.build(
        student_info=si,
        registration=reg,
        assessments=_empty_assessments(),
        student_assessment=_empty_student_assessment(),
        vle=pd.DataFrame({"id_site": [100, 101, 102, 103], "activity_type": ["url"]*4}),
        student_vle=sample_vle,
    )
    assert "total_clicks" in out.columns
    assert "clicks_week_1" in out.columns
    assert out["total_clicks"].sum() > 0


def test_clicks_week_columns_present(sample_student_info, sample_vle):
    si = encode_final_result(sample_student_info)
    reg = sample_student_info[["id_student", "code_module", "code_presentation"]].copy()
    factory = FeatureFactory(target_horizon_weeks=4)
    out = factory.build(
        student_info=si,
        registration=reg,
        assessments=_empty_assessments(),
        student_assessment=_empty_student_assessment(),
        vle=pd.DataFrame({"id_site": [100, 101, 102, 103], "activity_type": ["url"]*4}),
        student_vle=sample_vle,
    )
    for w in range(1, 5):
        assert f"clicks_week_{w}" in out.columns

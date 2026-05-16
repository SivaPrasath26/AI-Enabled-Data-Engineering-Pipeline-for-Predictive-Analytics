"""Generate a small synthetic OULAD-like dataset for smoke testing.

Useful when running the pipeline without internet access to the real OULAD
release. Output volume: ~2 000 students, ~150 000 VLE rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

GENDERS = ["M", "F"]
REGIONS = ["Scotland", "London Region", "South East Region", "North Region", "East Anglian Region"]
EDU = ["A Level or Equivalent", "HE Qualification", "Lower Than A Level", "Post Graduate Qualification"]
IMD = ["0-10%", "10-20", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
AGES = ["0-35", "35-55", "55<="]
MODULES = ["AAA", "BBB", "CCC", "DDD"]
PRESENTATIONS = ["2014J", "2014B", "2013J"]
FINAL = ["Pass", "Fail", "Withdrawn", "Distinction"]
ASMT_TYPES = ["TMA", "CMA", "Exam"]


def gen_students(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "id_student": np.arange(10000, 10000 + n),
        "gender": rng.choice(GENDERS, n),
        "region": rng.choice(REGIONS, n),
        "highest_education": rng.choice(EDU, n),
        "imd_band": rng.choice(IMD + [None], n, p=[0.1]*10 + [0.0]),
        "age_band": rng.choice(AGES, n, p=[0.55, 0.40, 0.05]),
        "num_of_prev_attempts": rng.integers(0, 4, n),
        "studied_credits": rng.integers(30, 360, n),
        "disability": rng.choice(["N", "Y"], n, p=[0.92, 0.08]),
        "code_module": rng.choice(MODULES, n),
        "code_presentation": rng.choice(PRESENTATIONS, n),
        "final_result": rng.choice(FINAL, n, p=[0.45, 0.18, 0.22, 0.15]),
    })


def gen_registration(students: pd.DataFrame) -> pd.DataFrame:
    return students[["id_student", "code_module", "code_presentation"]].assign(
        date_registration=rng.integers(-60, 0, len(students)),
        date_unregistration=rng.choice([np.nan] + list(range(-5, 250)), len(students), p=[0.78] + [0.22/255]*255),
    )


def gen_assessments() -> pd.DataFrame:
    rows = []
    aid = 1
    for m in MODULES:
        for p in PRESENTATIONS:
            for i, atype in enumerate(["TMA", "TMA", "TMA", "CMA", "Exam"]):
                rows.append({
                    "code_module": m,
                    "code_presentation": p,
                    "id_assessment": aid,
                    "assessment_type": atype,
                    "date": (i + 1) * 40 if atype != "Exam" else 230,
                    "weight": 20.0 if atype != "Exam" else 100.0,
                })
                aid += 1
    return pd.DataFrame(rows)


def gen_student_assessment(students: pd.DataFrame, asmt: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, s in students.iterrows():
        relevant = asmt[(asmt.code_module == s.code_module) & (asmt.code_presentation == s.code_presentation)]
        for _, a in relevant.iterrows():
            if rng.random() < 0.85:  # 85% submission rate
                out.append({
                    "id_assessment": a.id_assessment,
                    "id_student": s.id_student,
                    "date_submitted": a.date + int(rng.normal(0, 5)),
                    "is_banked": 0,
                    "score": float(np.clip(rng.normal(65, 20), 0, 100)),
                })
    return pd.DataFrame(out)


def gen_vle() -> pd.DataFrame:
    rows = []
    sid = 100000
    for m in MODULES:
        for p in PRESENTATIONS:
            for atype in ["resource", "url", "forumng", "oucontent", "subpage", "quiz", "homepage"]:
                for _ in range(20):
                    rows.append({
                        "id_site": sid,
                        "code_module": m,
                        "code_presentation": p,
                        "activity_type": atype,
                        "week_from": rng.integers(0, 30),
                        "week_to": rng.integers(0, 30),
                    })
                    sid += 1
    return pd.DataFrame(rows)


def gen_student_vle(students: pd.DataFrame, vle: pd.DataFrame, per_student: int = 80) -> pd.DataFrame:
    rows = []
    for _, s in students.iterrows():
        sites = vle[(vle.code_module == s.code_module) & (vle.code_presentation == s.code_presentation)]
        if sites.empty:
            continue
        ids = sites.id_site.sample(min(per_student, len(sites)), replace=True, random_state=int(s.id_student) % 10000)
        days = rng.integers(-7, 260, len(ids))
        clicks = np.clip(rng.poisson(4, len(ids)), 1, None)
        for sid_, d, c in zip(ids, days, clicks):
            rows.append({
                "code_module": s.code_module,
                "code_presentation": s.code_presentation,
                "id_student": s.id_student,
                "id_site": sid_,
                "date": int(d),
                "sum_click": int(c),
            })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="data/raw")
    p.add_argument("--students", type=int, default=2000)
    args = p.parse_args()
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    students = gen_students(args.students)
    students.to_csv(target / "studentInfo.csv", index=False)

    reg = gen_registration(students)
    reg.to_csv(target / "studentRegistration.csv", index=False)

    asmt = gen_assessments()
    asmt.to_csv(target / "assessments.csv", index=False)

    sasmt = gen_student_assessment(students, asmt)
    sasmt.to_csv(target / "studentAssessment.csv", index=False)

    vle = gen_vle()
    vle.to_csv(target / "vle.csv", index=False)

    svle = gen_student_vle(students, vle)
    svle.to_csv(target / "studentVle.csv", index=False)

    courses = pd.DataFrame([
        {"code_module": m, "code_presentation": p, "module_presentation_length": 268}
        for m in MODULES for p in PRESENTATIONS
    ])
    courses.to_csv(target / "courses.csv", index=False)

    print(f"wrote synthetic OULAD-like data to {target}")
    print(f"  students={len(students):,}  assessments={len(sasmt):,}  vle_events={len(svle):,}")


if __name__ == "__main__":
    main()

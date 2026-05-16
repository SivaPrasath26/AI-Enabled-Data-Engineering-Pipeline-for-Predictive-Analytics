# Data Dictionary

## OULAD source tables (Bronze)

### studentInfo

| Field | Type | Description |
|-------|------|-------------|
| code_module | string | Module identifier (AAA…GGG) |
| code_presentation | string | Presentation, e.g. 2014J |
| id_student | integer | Anonymised student id |
| gender | string | M / F |
| region | string | UK region of residence |
| highest_education | string | Highest prior qualification |
| imd_band | string | Index of Multiple Deprivation band |
| age_band | string | 0-35 / 35-55 / 55<= |
| num_of_prev_attempts | integer | Times the module was attempted previously |
| studied_credits | integer | Credit-weighted study load |
| disability | string | Y / N |
| final_result | string | Pass / Fail / Distinction / Withdrawn |

### studentVle

| Field | Type | Description |
|-------|------|-------------|
| code_module | string | |
| code_presentation | string | |
| id_student | integer | |
| id_site | integer | VLE resource identifier |
| date | integer | Days from start of presentation (-50…300) |
| sum_click | integer | Total clicks on that resource that day |

### studentAssessment

| Field | Type | Description |
|-------|------|-------------|
| id_assessment | integer | |
| id_student | integer | |
| date_submitted | integer | Day of submission |
| is_banked | integer | 1 if banked from a previous presentation |
| score | double | Mark out of 100 |

### assessments

| Field | Type | Description |
|-------|------|-------------|
| code_module | string | |
| code_presentation | string | |
| id_assessment | integer | |
| assessment_type | string | TMA / CMA / Exam |
| date | integer | Day on which assessment is due |
| weight | double | Weight in % of the final mark |

## Gold-tier feature matrix

| Feature | Type | Group | Description |
|---------|------|-------|-------------|
| total_clicks | double | vle_activity | Cumulative VLE clicks |
| active_days | double | vle_activity | Distinct days with activity |
| mean_clicks_per_day | double | vle_activity | |
| max_clicks_in_day | double | vle_activity | |
| n_resource_types | double | vle_activity | Distinct activity types accessed |
| clicks_week_1..4 | double | temporal_dynamics | Weekly click totals over the first 4 weeks |
| click_growth_rate | double | temporal_dynamics | Mean week-over-week growth |
| days_since_last_activity | double | temporal_dynamics | Days since the most recent click |
| session_duration_variance | double | temporal_dynamics | Std-dev of daily clicks |
| n_assessments_submitted | double | assessment_history | |
| n_assessments_missed | double | assessment_history | |
| mean_assessment_score | double | assessment_history | |
| std_assessment_score | double | assessment_history | |
| median_days_to_submission | double | assessment_history | |
| late_submission_ratio | double | assessment_history | |
| days_to_registration | double | registration | |
| days_to_unregistration | double | registration | |
| gender, region, … (one-hot) | binary | demographic | One-hot encoded categorical attributes |
| final_result_binary | int | target | 1 = at-risk (Fail / Withdrawn), 0 = on-track |

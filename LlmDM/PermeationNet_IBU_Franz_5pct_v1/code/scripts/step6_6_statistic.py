import pandas as pd

df = pd.read_csv("GenAI/outputs/step6_verified_records_v1_2.csv")

print("Total verified:", len(df))
print("pass_v1:", df["pass_v1"].sum())

# 失败分桶（最关键）
def bucket(r):
    if r["franz_confirmed"] != "yes":
        return "not_franz"
    if r["ibuprofen_5pct_w_w"] != "yes":
        return "not_5pct"
    if pd.isna(r["endpoint_value"]) or pd.isna(r["endpoint_time_h"]):
        return "missing_endpoint_or_time"
    if r["endpoint_kind"] == "percent":
        return "endpoint_is_percent"
    if pd.isna(r["endpoint_value_ug_cm2"]):
        # 常见：total amount 但缺 diffusion area，或单位不支持
        if r["endpoint_kind"] == "amount_total" and pd.isna(r["diffusion_area_cm2"]):
            return "total_amount_missing_area"
        return "unit_or_conversion_issue"
    return "other"

df["fail_bucket"] = df.apply(bucket, axis=1)
print(df.loc[df["pass_v1"]==False, "fail_bucket"].value_counts())

# 通过集基本统计
v1 = df[df["pass_v1"]==True].copy()
print("V1 endpoint_kind:", v1["endpoint_kind"].value_counts(dropna=False))
print("V1 barrier:", v1.get("barrier_category", pd.Series()).value_counts(dropna=False) if "barrier_category" in v1.columns else "barrier_category not in verifier file")
print("V1 time(h) describe:\n", v1["endpoint_time_h"].describe())

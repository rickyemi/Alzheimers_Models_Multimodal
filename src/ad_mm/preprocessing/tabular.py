"""Tabular (demographic, cognitive, volumetric, plasma biomarker) cleaning and feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

DX_MAP = {"CN": "CN", "Normal": "CN", "SMC": "SMC", "SCD": "SMC", "EMCI": "EMCI", "E-MCI": "EMCI",
          "LMCI": "LMCI", "L-MCI": "LMCI", "AD": "AD", "Dementia": "AD"}
SEX_MAP = {"m": 1, "male": 1, "f": 0, "female": 0}
PLAUSIBLE = {"age": (40, 100), "mmse": (0, 30), "moca": (0, 30), "cdr_sb": (0, 18), "adas_cog13": (0, 85),
             "faq": (0, 30), "education_yrs": (0, 25), "hippocampus_vol_mm3": (2000, 12000)}
LOG_FEATURES = ["ptau217_pg_ml", "nfl_pg_ml", "gfap_pg_ml"]
MISSING_FLAG = ["ptau217_pg_ml", "ab42_40_ratio", "nfl_pg_ml", "gfap_pg_ml"]
NUMERIC = ["age", "sex", "education_yrs", "apoe4_copies", "mmse", "moca", "cdr_sb", "adas_cog13", "faq",
           "ecog_self", "ecog_partner", "hippocampus_vol_mm3", "entorhinal_thick_mm", "ventricles_vol_mm3",
           "ptau217_pg_ml", "ab42_40_ratio", "nfl_pg_ml", "gfap_pg_ml"]


def clean_tabular(df: pd.DataFrame, id_fn) -> tuple[pd.DataFrame, dict]:
    log = {"rows_raw": len(df)}
    t = df.copy()
    t["patient_id"] = t.patient_id.map(id_fn)
    before = len(t)
    t = t.drop_duplicates()
    t = t.drop_duplicates(subset="patient_id", keep="first")
    log["duplicates_removed"] = before - len(t)
    t["dx"] = t.dx.map(DX_MAP)
    log["unmapped_dx"] = int(t.dx.isna().sum())
    t["sex"] = t.sex.astype(str).str.lower().map(SEX_MAP)
    # unit harmonization: hippocampal volume recorded in cm3 at one site
    unit_err = t.hippocampus_vol_mm3 < 100
    log["hippocampus_cm3_to_mm3"] = int(unit_err.sum())
    t.loc[unit_err, "hippocampus_vol_mm3"] *= 1000
    # data-entry typos: decimal shift
    age_err = t.age < 20
    log["age_decimal_fix"] = int(age_err.sum())
    t.loc[age_err, "age"] *= 10
    mmse_err = t.mmse > 30
    log["mmse_decimal_fix"] = int(mmse_err.sum())
    t.loc[mmse_err, "mmse"] /= 10
    # remaining implausible values -> NaN (imputed later)
    n_implaus = 0
    for c, (lo, hi) in PLAUSIBLE.items():
        bad = (t[c] < lo) | (t[c] > hi)
        n_implaus += int(bad.sum())
        t.loc[bad, c] = np.nan
    log["implausible_set_nan"] = n_implaus
    log["missing_cells"] = int(t[NUMERIC].isna().sum().sum())
    log["rows_clean"] = len(t)
    return t.reset_index(drop=True), log


class TabularTransformer:
    """Fit-on-train transformer: missing indicators, median impute, log transform, z-score."""

    def fit(self, df: pd.DataFrame) -> "TabularTransformer":
        x = self._base(df)
        self.medians = x.median()
        x = x.fillna(self.medians)
        self.mean, self.std = x.mean(), x.std().replace(0, 1)
        self.columns = list(x.columns)
        return self

    def _base(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[NUMERIC].astype(float).copy()
        for c in MISSING_FLAG:
            x[f"{c}_missing"] = x[c].isna().astype(float)
        for c in LOG_FEATURES:
            x[c] = np.log1p(x[c])
        return x

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self._base(df).fillna(self.medians)
        flags = [c for c in x.columns if c.endswith("_missing")]
        z = (x - self.mean) / self.std
        z[flags] = x[flags]                   # keep indicators as 0/1
        return z[self.columns].astype(np.float32)

    def to_dict(self) -> dict:
        return {"columns": self.columns, "medians": self.medians.to_dict(), "mean": self.mean.to_dict(),
                "std": self.std.to_dict(), "log_features": LOG_FEATURES, "missing_flags": MISSING_FLAG}

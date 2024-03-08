import pandas as pd
import numpy as np


def recode_income(spc: pd.DataFrame) -> pd.DataFrame:
    """Recodes yearly income."""
    # Household Income
    # --- Get sum of spc.salary_yearly per household
    spc["salary_yearly_hh"] = spc.groupby("household")["salary_yearly"].transform("sum")

    # --- transform into categorical column so that it matches the reported NTS values
    # Define the bins
    bins = [0, 24999, 49999, np.inf]
    # Define the labels for the bins
    labels = ["0-25k", "25k-50k", "50k+"]

    spc["salary_yearly_hh_cat"] = pd.cut(
        spc["salary_yearly_hh"], bins=bins, labels=labels
    )

    return spc

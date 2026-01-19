# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 15:58:26 2026

@author: aksha
"""

import numpy as np
import pandas as pd

class SchemaValidator:
    def __init__(self, expected_cols):
        self.expected_cols = expected_cols

    def validate(self, data: dict) -> pd.DataFrame:
        df = pd.DataFrame([data])
        for col in self.expected_cols:
            if col not in df.columns:
                df[col] = np.nan
        return df[self.expected_cols]

# -*- coding: utf-8 -*-
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from collections import defaultdict

@st.cache_resource
class ShapExplainer:
    def __init__(self, model):
        self.preprocessor = model.named_steps['preprocessing']
        self.model = model.named_steps['rf_bal']
        self.explainer = shap.Explainer(
            self.model, feature_perturbation="tree_path_dependent"
        )

    def plot(self, df):
        df_transformed = self.preprocessor.transform(df)
        new_df = pd.DataFrame(df_transformed, columns=self.preprocessor.get_feature_names_out())
        shap_values = self.explainer(new_df)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0, :, 1])
        return fig






























# -*- coding: utf-8 -*-
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from collections import defaultdict

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
        friendly_names = [name.split("_")[1] for name in self.preprocessor.get_feature_names_out()]
        shap_row = shap_values[0, :, 1]  # one sample, one class

        groups = defaultdict(list)
        for i, name in enumerate(friendly_names):
            groups[name].append(i)
        
        merged_shap = {base: np.sum(shap_row[idxs]) for base, idxs in groups.items()}
        values = np.array(list(merged_shap.values()))
        names = list(merged_shap.keys())
        merged_explanation = shap.Explanation(values=values, feature_names=names)
        fig, ax = plt.subplots()
        shap.plots.waterfall(merged_explanation)
        return fig



























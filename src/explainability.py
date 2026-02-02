# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 16:02:10 2026

@author: aksha
"""
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
        shap.plots.waterfall(shap_values[0, :, 0], max_display=10)
        try:
            feature_names = new_df.columns
            mean_abs_shap = pd.DataFrame({
                'Feature': feature_names,
                'Mean |SHAP|': np.abs(shap_values).mean(axis=0)
            }).sort_values(by='Mean |SHAP|', ascending=False).head(top_n)
    
            lines = [f"### 🧠 SHAP Insight Summary (Top {top_n} Features)\n"]
            for _, row in mean_abs_shap.iterrows():
                lines.append(f"- **{row['Feature']}**: Avg contribution = {row['Mean |SHAP|']:.4f}")
            exp_lines =  "\n".join(lines)
        except Exception as e:
            exp_lines = f"⚠️ Unable to generate SHAP insights: {e}"
        return fig, exp_lines













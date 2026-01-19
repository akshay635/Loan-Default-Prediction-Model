# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 16:02:10 2026

@author: aksha
"""

import shap
import matplotlib.pyplot as plt

class ShapExplainer:
    def __init__(self, model):
        self.explainer = shap.TreeExplainer(
            model, feature_perturbation="tree_path_dependent"
        )

    def plot(self, df):
        shap_values = self.explainer(df)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10)
        return fig

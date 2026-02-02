import pandas as pd
import numpy as np
import shap

def generate_feature_insight(df, importances, top_n = 5):
    top_features = importances.sort_values(by='Importances', ascending=False).head(top_n)
    lines = [f"### 🧠 Top {top_n} Feature Insights\n"]

    for _, row in top_features.iterrows():
        col = row['Features']
        imp = row['Importances']
        if col not in df.columns:
            lines.append(f"- **{col}**: ⚠️ Not found in input data.")
            continue

        series = df[col].dropna()
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() <= 1:
                lines.append(f"- **{col}** (Importance: {imp:.2f}): Constant value → {series.iloc[0] if not series.empty else 'N/A'}")
                continue
            try:
                desc = series.describe()
                skew = series.skew()
                skewness = (
                    "right-skewed" if skew > 1 else
                    "left-skewed" if skew < -1 else
                    "approximately symmetric"
                )
                lines.append(
                    f"- **{col}** (Importance: {imp:.2f}): Mean = {desc['mean']:.2f}, "
                    f"Std = {desc['std']:.2f}, Skew = {skew:.2f} → *{skewness}*"
                )
            except Exception as e:
                lines.append(f"- **{col}** (Importance: {imp:.2f}): ⚠️ Could not compute skew ({e})")
        else:
            top_vals = series.value_counts().head(3).to_dict()
            if top_vals:
                top_str = ', '.join([f"{k} ({v})" for k, v in top_vals.items()])
                lines.append(f"- **{col}** (Importance: {imp:.2f}): Top values → {top_str}")
            else:
                lines.append(f"- **{col}** (Importance: {imp:.2f}): No non-null values to summarize.")

    return "\n".join(lines)


def generate_shap_insight(df, shap_values, top_n = 5):
    try:
        feature_names = df.columns
        mean_abs_shap = pd.DataFrame({
            'Feature': feature_names,
            'Mean |SHAP|': np.abs(shap_values).mean(axis=0)
        }).sort_values(by='Mean |SHAP|', ascending=False).head(top_n)

        lines = [f"### 🧠 SHAP Insight Summary (Top {top_n} Features)\n"]
        for _, row in mean_abs_shap.iterrows():
            lines.append(f"- **{row['Feature']}**: Avg contribution = {row['Mean |SHAP|']:.4f}")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Unable to generate SHAP insights: {e}"

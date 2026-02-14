# imporrting Pandas
import pandas as pd
import plotly.express as px

class Feature_Engineering:
  # Feature engineering
  def derived_features(self, df):
    df['EMI/Income_ratio'] = round((df['EMI'] / df['Monthly_Income']), 2)
    df['Post_DTI'] = df['DTIRatio'] + df['EMI/Income_ratio']
    df['age_post_dti'] = df['Age'] * df['Post_DTI']
    df['tenure_age_ratio'] = df['MonthsEmployed'] / (df['Age'] + 1e-6)
    df['debt_stress'] = df['EMI/Income_ratio'] * df['DTIRatio']

    return df
  # Feature Importances
  def Feature_IMP(self, df):
    fig = px.bar(
                df.head(10),
                x="Importances",
                y="Features",
                title="Top Features Driving Model Decisions",
                text_auto=True)
    
    fig.update_layout(yaxis=dict(autorange="reversed"))
  
    return fig

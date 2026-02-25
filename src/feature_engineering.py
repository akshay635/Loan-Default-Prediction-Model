# imporrting Pandas
import pandas as pd
import plotly.express as px

class FeatureEngineering:
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

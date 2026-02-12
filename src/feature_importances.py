import pandas as pd
import plotly.express as px

# Feature Importances
def Feature_IMP(df):
  fig = px.bar(
              df.head(10),
              x="Importances",
              y="Features",
              title="Top Features Driving Model Decisions",
              text_auto=True)
  
  fig.update_layout(yaxis=dict(autorange="reversed"))

  return fig

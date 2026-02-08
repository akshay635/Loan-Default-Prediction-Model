# Exploration Tab

def tab2():
  with tab2:
      st.header("Explore Model Insights")
      st.markdown(
          """
          This section helps you understand the **bigger picture**:  
          - Which features matter most overall  
          - How borrowers compare across different profiles  
          - Narrative insights into repayment stability
          """
      )
  
      feature_imp_df = pd.read_csv(RiskConfig.FEATURE_IMP_PATH)
      st.subheader("📊 Global Feature Importance")
      fig = px.bar(
          feature_imp_df,
          x="Importances",
          y="Features",
          title="Overall Feature Importance",
          text_auto=True
      )
      fig.update_layout(yaxis=dict(autorange="reversed"))
      st.plotly_chart(fig, use_container_width=True)
  
      top_features = feature_imp_df.sort_values("Importances", ascending=False).head(3)
      st.markdown(
          f"""
          🗣️ **Dynamic Insights:**  
          Right now, the model is most influenced by **{top_features.iloc[0]['Features']}**, 
          followed by **{top_features.iloc[1]['Features']}** and **{top_features.iloc[2]['Features']}**.  
          This means these features are the strongest drivers of repayment readiness in your profile.
          """
      )

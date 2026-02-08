def tab1():
  with tab1:
    st.header("Your repayment risk assessment")

    if st.button("🔍 Assess Risk"):
        df, issues = validator.validate_inference(user_data)

        if issues:
            st.error(" ".join([i for i in issues]))
            st.stop()
        # Feature engineering
        df['EMI/Income_ratio'] = round((df['EMI'] / df['Monthly_Income']), 2)
        df['Post_DTI'] = df['DTIRatio'] + df['EMI/Income_ratio']
        df['age_post_dti'] = df['Age'] * df['Post_DTI']
        df['tenure_age_ratio'] = df['MonthsEmployed'] / (df['Age'] + 1e-6)
        df['debt_stress'] = df['EMI/Income_ratio'] * df['DTIRatio']

        df = df[RiskConfig.EXPECTED_COLS]
        prob = model.predict_proba(df)

        # Narrative risk output
        st.subheader("📈 Repayment Risk assessment")
        risk, action = decision_engine.decide(prob)
        if risk == "HIGH":
            st.error(f"❌ High repayment risk ({prob:.2%})")
            st.markdown("""This application shows a higher-than-average probability of repayment
                            difficulty based on financial indicators such as income stability and
                            debt obligations. The customer has higher chances to stop repayments 
                            and default the loan.""")
        elif risk == "MEDIUM":
            st.warning(f"⚠️ Moderate repayment risk ({prob:.2%})")
            st.markdown("""This assessment indicates a moderate probability (35%-60%) of repayment difficulty
                           based on the available financial information, suggesting that further review may be 
                           appropriate.""")
        else:
            st.success(f"✅ Low risk of repayment ({prob:.2%})")
            st.markdown("""This assessment indicates a lower probability of repayment difficulty,
                           suggesting comparatively lower risk based on the available information.""")

        st.markdown(f"**Suggested Action:** {action}")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Global Feature Importance")
            feature_imp_df = pd.read_csv(RiskConfig.FEATURE_IMP_PATH)
            fig = px.bar(
                feature_imp_df.head(10),
                x="Importances",
                y="Features",
                title="Top Features Driving Model Decisions",
                text_auto=True
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=False)
            with st.expander('Feature Summary'):
                st.markdown(generate_feature_insight(df, feature_imp_df, top_n = 5))

        with col2:
            st.subheader("Personalized SHAP Explanation")
            fig = explainer.plot(df)
            st.pyplot(fig, use_container_width=False)
            st.markdown(
            """Features pushing the risk higher are shown in red, 
               while features reducing risk are shown in blue.""")

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 16:01:25 2026

@author: aksha
"""

class RiskDecisionEngine:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def decide(self, prob):
        if prob >= self.high:
            return "HIGH", "Reject or apply stricter loan terms"
        elif prob >= self.low:
            return "MEDIUM", "Manual review recommended"
        else:
            return "LOW", "Future loan can be approved if applied"

        risk = prob * 100

        st.metric(
            label="Default Risk",
            value=f"{risk:.1f}%",
            delta="High Risk" if risk > 35 else "Low Risk"
        )
        
        decision = "REJECT" if risk > 35 else "APPROVE"
        
        st.subheader(f"Decision: {decision}")

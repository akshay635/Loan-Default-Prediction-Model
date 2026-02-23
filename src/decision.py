# -*- coding: utf-8 -*-

class RiskDecisionEngine:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def decide(self, prob):
        if prob >= self.high:
            return "HIGH", "Reject or apply stricter loan terms"
        elif prob > self.low and prob < self.high:
            return "MEDIUM", "Manual review recommended"
        else:
            return "LOW", "Future loan can be approved if applied"





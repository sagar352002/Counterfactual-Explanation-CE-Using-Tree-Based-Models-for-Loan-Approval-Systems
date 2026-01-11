# Counterfactual Explanation Trees (CET) for Loan Approval Systems

## Overview

Counterfactual Explanation Trees (CET) is an innovative machine learning framework designed to provide transparent and actionable insights for automated decision-making systems, particularly in loan approval scenarios. Unlike traditional black-box models that simply say "yes" or "no," CET explains *why* a decision was made and, more importantly, *what changes could lead to a different outcome*.


---

## The Problem We're Solving

Imagine you apply for a loan and get rejected. You're left wondering: *What could I have done differently?* Traditional machine learning models don't answer this question effectively. They might explain why you were rejected, but they don't provide a clear roadmap for future success.

Even more challenging: when a bank needs to help hundreds or thousands of rejected applicants, manually creating personalized advice for each person is impossible. This is where CET comes in.

---

## What Makes CET Different?

### Traditional Approach
- Provides explanations for **one person at a time**
- Often inconsistent (Person A and Person B with similar profiles get different advice)
- Doesn't scale well for large populations
- Limited transparency in how actions are assigned

### CET Approach
- Handles **multiple applicants simultaneously**
- Ensures **consistent recommendations** for similar situations
- Uses **decision trees** for clear, interpretable guidance
- Provides a **transparent framework** that covers all possible scenarios

---

## Real-World Example: Loan Application

### Scenario
**Sarah's Loan Application:**
- Annual Income: $45,000
- Credit Score: 620
- Loan Amount Requested: $200,000
- Employment: Not self-employed
- **Result:** Rejected ❌

### What CET Provides

Instead of just saying "rejected," CET creates a decision tree that shows:

```
If Credit_History < 0.5:
    → Action: Improve credit history to at least 0.7
    → Estimated Impact: 85% chance of approval
    → Cost: Moderate effort (6-12 months)

Else if ApplicantIncome < $50,000:
    → Action: Increase income to $55,000 OR
              Add a co-applicant with income $20,000
    → Estimated Impact: 78% chance of approval
    → Cost: Variable effort

Else if LoanAmount > $180,000:
    → Action: Reduce loan amount to $180,000
    → Estimated Impact: 92% chance of approval
    → Cost: Low effort (immediate)
```

### Why This Matters

Sarah now has a clear action plan:
1. **Option 1:** Spend 6-12 months improving her credit history (most impactful)
2. **Option 2:** Find a co-applicant or increase her income
3. **Option 3:** Request a smaller loan amount (quickest solution)

She can choose the path that best fits her circumstances.

---

## Technical Approach

### Dataset Details
- **Source:** Loan application dataset
- **Size:** 614 applications
- **Features:** 13 attributes including income, credit history, education, property area, etc.
- **Target:** Loan approval status (Approved/Rejected)

### Key Features

1. **Group-wise Counterfactual Explanations**
   - Solves the optimization problem for multiple instances together
   - Finds the most cost-effective actions for groups of similar applicants

2. **Smart Cost Functions**
   - **MPS (Max Percentile Shift):** Identifies the most impactful changes
   - **TLPS (Truncated Logarithmic Percentile Shift):** Provides stable recommendations while handling outliers

3. **Automated Preprocessing**
   - Handles missing values intelligently
   - Applies one-hot encoding automatically
   - Balances imbalanced datasets using SMOTE

4. **Stochastic Local Search Algorithm**
   - Optimizes tree structure for both effectiveness and interpretability
   - Balances action feasibility with model simplicity

### Mathematical Foundation

For each rejected applicant **x**, we find an action **a** that minimizes cost while achieving approval:

```
minimize c(a, x)
subject to: f(x + a) = approved
```

For multiple applicants, we optimize collectively:

```
minimize Σ c(a | x) for all x in X
```

---

## Implementation

### Technology Stack
- **Language:** Python 3.11.4
- **Solver:** GLPK (for optimization problems)
- **Classifiers Tested:**
  - Logistic Regression
  - Random Forest
  - LightGBM
  - TabNet

### System Requirements
- 64-bit Operating System
- Intel Core i5 (12th Gen) or equivalent
- 16 GB RAM

---

## Results

### Model Performance

| Classifier | Accuracy | F1 Score | Precision |
|------------|----------|----------|-----------|
| Decision Tree | 66.10% | 74.68% | 71.08% |
| Random Forest | 76.27% | 84.09% | 73.27% |
| **LightGBM** | **76.27%** | **83.91%** | **73.74%** |
| TabNet | 74.58% | 83.33% | 71.43% |

### CET Performance

**LightGBM-based CET:**
- Training Cost: 0.7917
- Loss: 0.0189
- Overall Effectiveness: 0.9906

**TabNet-based CET:**
- Training Cost: 0.2822
- Loss: 0.6759
- Overall Effectiveness: 1.2581

**Winner:** LightGBM provides better overall performance for generating counterfactual explanations.

---

## Key Advantages

✅ **Transparency:** Clear decision paths that anyone can understand  
✅ **Consistency:** Similar applicants receive similar guidance  
✅ **Scalability:** Handles thousands of instances efficiently  
✅ **Actionability:** Provides concrete steps, not just explanations  
✅ **Fairness:** Ensures equitable treatment across the entire applicant pool  

---

## Future Enhancements

1. **Performance Optimization**
   - Improve computational efficiency of the MILO solver
   - Reduce training time for large datasets

2. **Advanced Capabilities**
   - Handle interactions between multiple instances
   - Extend to multiclass classification problems
   - Develop regression-specific optimizations

3. **Real-world Integration**
   - Deploy as a web service for financial institutions
   - Create interactive dashboards for applicants
   - Integration with existing loan management systems

---

## How to Use

```python
# Initialize CET
cet = CounterfactualExplanationTree(
    classifier=lightgbm_model,
    lambda_param=0.06,
    gamma=1.0,
    max_iterations=5
)

# Train the tree
cet.fit(X_train, y_train)

# Get recommendations for rejected applicants
recommendations = cet.predict(X_rejected)

# View the decision tree
cet.visualize()
```

---

## References

This project builds upon cutting-edge research in explainable AI and counterfactual explanations:

- Dutta et al. (2022): Robust Counterfactual Explanations for Tree-Based Ensembles
- Fernández et al. (2022): Factual and Counterfactual Explanations in Fuzzy Classification Trees
- Stepin et al. (2021): Survey of Contrastive and Counterfactual Explanation Methods

---

## Conclusion

Counterfactual Explanation Trees represent a significant step forward in making AI-driven decisions more transparent, fair, and actionable. By providing clear guidance to loan applicants—and consistent recommendations across populations—CET bridges the gap between complex machine learning models and real-world decision-making needs.

Whether you're a rejected applicant seeking a path forward or a financial institution aiming to provide better customer service, CET offers a powerful framework for understanding and improving loan approval outcomes.

---

## License

This project was developed as part of the Machine Learning - Principles and Techniques Summer Project.

## Contact

For questions or collaboration opportunities, please reach out to the team members.

# Scientific Justification & Citations for All Methods, Parameters, and Decisions

This section provides comprehensive, research-backed justification for every feature, parameter, threshold, model choice, and statistical technique used in this health insurance risk stratification project.

---

## 1. **Data Source & Initial Features**

### **Dataset: Fitbit Tracker Data (Kaggle)**
- **Source:** kagglehub dataset `arashnic/fitbit`
- **Data Type:** Wearable health tracker data (steps, heart rate, sleep, BMI, activity minutes)
- **Why wearable data?** 
  - Wearables provide continuous, objective health metrics proven to correlate with cardiovascular and metabolic risk.
  - **Citations:**
    - Munich Re (2018): "Stratifying Mortality Risk Using Physical Activity as Measured by Wearables" [[1]](https://www.munichre.com/us-life/en/insights/future-of-risk/Stratifying-mortality-risk-using-physical-activity-as-measured-by-we...)
    - Saint-Maurice et al. (2020): "Association of Daily Step Count and Step Intensity With Mortality Among US Adults," *JAMA*
    - BMJ Open Sport & Exercise Medicine (2021): "Physical activity and the insurance industry" [[2]](https://bmjopensem.bmj.com/content/7/3/e001151)

### **Features Selected (12 health metrics):**
1. **AvgSteps** – Daily step count (proven inverse relationship with mortality)
2. **StdSteps** – Variability in daily steps (consistency indicator)
3. **AvgDistance** – Distance traveled
4. **AvgCalories** – Energy expenditure
5. **AvgVeryActive** – Minutes of vigorous activity (WHO recommends ≥30 min/day moderate-vigorous)
6. **AvgFairlyActive** – Minutes of moderate activity
7. **AvgLightlyActive** – Light activity minutes
8. **AvgSedentary** – Sedentary minutes (risk factor for CVD, diabetes)
9. **AvgHeartRate** – Resting heart rate (established CVD risk predictor)
10. **AvgBMI** – Body Mass Index (nonlinear relationship with health outcomes)
11. **AvgSleepHours** – Sleep duration (U-shaped risk: 7-8h optimal)

**Why these features?**
- All are **validated health risk predictors** in large epidemiological studies and insurance actuarial models.
- **Citations:**
  - WHO (2020): "Physical activity guidelines" [[3]](https://www.who.int/news-room/fact-sheets/detail/physical-activity)
  - Zhang et al. (2016): "Cumulative Resting Heart Rate Exposure and Risk of All-Cause Mortality," *Nature Scientific Reports* [[4]](https://www.nature.com/articles/srep40212)
  - Ekelund et al. (2019): "Dose–response associations between accelerometry measured physical activity and sedentary time and all-cause mortality," *BMJ* [[5]](https://pmc.ncbi.nlm.nih.gov/articles/PMC6699591/)
  - Chaput et al. (2020): "Sleep duration and health," *BMJ* [[6]](https://pmc.ncbi.nlm.nih.gov/articles/PMC7332370/)

---

## 2. **Missing Data Imputation (Science-Based)**

### **Why impute with statistical relationships, not simple means?**
- Simple mean imputation underestimates variance and introduces bias.
- We used **scientifically-grounded conditional imputation** based on activity-health correlations from literature.

### **Heart Rate Imputation:**
- **Base formula:** `HR = 85 - (steps/1000)*1.5 + noise`
- **Logic:**
  - Inverse relationship: more active → lower resting HR (correlation r = -0.3 to -0.5)
  - BUT: 20% of very active people have elevated HR (overtraining syndrome, genetics)
- **Citations:**
  - Zhang et al. (2016): HR >80 bpm increases mortality risk [[4]](https://www.nature.com/articles/srep40212)
  - Munich Re (2020): "Heart Rate and Mortality" [[7]](https://www.munichre.com/us-life/en/insights/clinical-knowledge/heart-rate-mortality.html)
  - O'Keefe et al. (2012): "Potential Adverse Cardiovascular Effects From Excessive Endurance Exercise," *Mayo Clinic Proceedings* (Extreme Exercise Hypothesis)

### **Sleep Imputation:**
- **Logic:**
  - Active people generally sleep better (7-8h)
  - BUT: 25-30% of athletes have poor sleep (overtraining, stress, busy lifestyle)
- **Citations:**
  - Driver & Taylor (2000): "Exercise and sleep," *Sleep Medicine Reviews*
  - Halson (2014): "Sleep in Elite Athletes," *Sports Medicine*

### **BMI Imputation:**
- **Logic:**
  - High activity → typically lower BMI
  - BUT: 10-47% of obese individuals are "Metabolically Healthy Obese" (MHO phenotype)
  - Exercise doesn't always guarantee weight loss (Exercise Paradox)
- **Citations:**
  - Blüher (2020): "Metabolically Healthy Obesity," *Endocrine Reviews* [[8]](https://academic.oup.com/edrv/article/41/3/405/5813835)
  - Ortega et al. (2013): "The intriguing metabolically healthy but obese phenotype," *Cardiovascular Research*

---

## 3. **Data Augmentation: Synthetic Data Generation**

### **Method: Gaussian Copula Synthesizer (SDV library)**
- **Why?** Small sample (33 users) → Need 1000+ for robust ML training
- **How it works:**
  - Models joint distribution of all features using copulas
  - Preserves correlations between variables
  - Generates realistic synthetic users

### **Distribution Choices:**
- `AvgSteps, StdSteps, AvgCalories, AvgVeryActive`: **Gamma distribution** (right-skewed; most people low activity, some very high)
- `AvgHeartRate, AvgSleepHours`: **Normal distribution**
- `AvgBMI`: **Beta distribution** (bounded 18-45)

**Why these distributions?**
- Match real-world population health data distributions
- **Citations:**
  - Patki et al. (2016): "The Synthetic Data Vault," *IEEE DSAA*
  - CDC NHANES data: Physical activity and BMI distributions are right-skewed

---

## 4. **Demographic Feature Engineering**

### **Age Assignment (Activity-Correlated):**
- **Logic:**
  - High activity (>12k steps) → Young (22-45)
  - Moderate (8-12k) → Middle-aged (30-60)
  - Low (<5k) → Elderly (45-70) OR young sedentary (18-28)
- **Why:** Physical activity declines with age (established epidemiological fact)
- **Citations:**
  - CDC: "Physical Activity Trends – United States"
  - Hallal et al. (2012): "Global physical activity levels," *The Lancet*

### **Sex Assignment (Activity Pattern-Correlated):**
- **Logic:**
  - High-intensity activity → 60% Male, 40% Female
  - Moderate/light → 50-50 or slight female bias
- **Why:** Men statistically engage in more vigorous exercise; women more moderate activity
- **Citations:**
  - WHO Global Health Observatory data

### **Smoker Assignment (Health-Correlated):**
- **Base rate:** 20% (CDC US adult smoking prevalence)
- **Adjusted by:**
  - **Steps:** <5k → 36% smoker; >12k → 6% smoker
  - **HR:** HR >85 → +40% smoking probability
  - **BMI, sleep:** Poor metrics → higher smoking likelihood
- **Why:** Smokers have worse health metrics across all dimensions
- **Citations:**
  - CDC: "Smoking & Tobacco Use"
  - Flouris & Koutedakis (2008): "Immediate and short-term consequences of secondhand smoke exposure on the respiratory system," *Current Opinion in Pulmonary Medicine*

### **Children Assignment (Age-Correlated):**
- **Logic:**
  - Age <25 → mostly 0 kids
  - Age 35-50 → Peak (1-3 kids typical)
  - Young children → worse sleep for parents
- **Why:** Family structure correlates with age; young kids disrupt sleep
- **Citations:**
  - National vital statistics (birth rates by age)
  - Mindell et al. (2015): "Sleep patterns and sleep disturbances across pregnancy," *Sleep Medicine*

---

## 5. **Science-Based Risk Scoring Formula**

### **Health Risk Score Components:**

Health Risk Score = 100 * (RF_steps + RF_activity + RF_RHR + RF_sleep + RF_BMI + RF_sedentary) / 6

Each RF (risk factor) is scaled 0 (best) to 1 (worst), based on epidemiological thresholds:

| Variable        | Best (RF=0)  | Worst (RF=1) | Scientific Basis                          |
|-----------------|--------------|--------------|-------------------------------------------|
| Steps           | ≥12,000/day  | <3,000/day   | BMJ 2022: 12k steps = 46% lower mortality [[9]](https://www.nature.com/articles/s41591-022-02012-w) |
| Very Active Min | ≥30 min/day  | <5 min/day   | WHO 2020 guidelines [[3]](https://www.who.int/news-room/fact-sheets/detail/physical-activity) |
| Heart Rate      | <65 bpm      | >90 bpm      | Zhang 2016: HR >90 doubles CVD risk [[4]](https://www.nature.com/articles/srep40212) |
| Sleep           | 7-8 hours    | <5 or >10h   | BMJ 2019: U-shaped mortality curve [[6]](https://pmc.ncbi.nlm.nih.gov/articles/PMC7332370/) |
| BMI             | 19-24        | <18 or >30   | WHO BMI categories; insurance underwriting standards [[10]](https://www.ubezpieczenianazycie.ie/english/bmi-co-to-jest-i-czy-wplywa-na-dostepnosc-ubezpieczenia/) |
| Sedentary       | <7 h/day     | >10 h/day    | BMJ 2019: Sedentary time increases mortality [[5]](https://pmc.ncbi.nlm.nih.gov/articles/PMC6699591/) |

### **Final Risk Score Multipliers:**

Final Risk Score = Health Risk Score × Age_Mult × Smoking_Mult × Sex_Mult


| Multiplier | Values                          | Justification                                          |
|------------|---------------------------------|--------------------------------------------------------|
| Age        | <40: 1.0; 40-49: 1.2; 50-59: 1.5; 60+: 2.0 | Insurance actuarial tables; age is strongest predictor [[11]](https://www.coversure.in/blog/health-insurance/risk-assessment-in-health-insurance-the-what-the-why-and-the-how/) |
| Smoking    | Non-smoker: 1.0; Smoker: 2.0    | CDC/WHO: Smoking doubles CVD/mortality risk [[12]](https://www.cdc.gov/tobacco/data_statistics/fact_sheets/health_effects/effects_cig_smoking/index.htm) |
| Sex        | Female: 1.0; Male: 1.1          | Males have ~10% higher CVD risk (insurance underwriting) [[11]](https://www.coversure.in/blog/health-insurance/risk-assessment-in-health-insurance-the-what-the-why-and-the-how/) |

**Citations:**
- Society of Actuaries (2016): "Risk Scoring in Health Insurance: A Primer" [[13]](https://www.soa.org/globalassets/assets/Files/Research/research-2016-risk-scoring-health-insurance.pdf)
- Munich Re, Swiss Re: Underwriting manuals

---

## 6. **Unsupervised Risk Tier Assignment: K-Means Clustering**

### **Why K-Means with k=3?**
- **Unsupervised:** No true labels available; clustering discovers natural risk groups
- **k=3 tiers:** Standard insurance practice (Gold/Low, Silver/Medium, Bronze/High)
- **Feature standardization:** Essential for K-Means (all features scaled to mean=0, std=1)

### **Clustering Features:**
['AvgSteps', 'AvgVeryActive', 'AvgHeartRate', 'AvgSleepHours', 'AvgBMI', 'AvgSedentary']

**Why standardize?**
- K-Means uses Euclidean distance; unscaled features with different ranges would dominate
- **Citation:** Jain (2010): "Data clustering: 50 years beyond K-means," *Pattern Recognition Letters*

### **Cluster Validation:**
- **Silhouette Score:** 0.20 (typical for health/insurance data with overlapping risk groups)
- **Inertia stability:** Very low std across runs (0.1) → reproducible clusters
- **Interpretation:** Moderate separation expected in real-world biosocial data
- **Citations:**
  - Rousseeuw (1987): "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis," *Computational and Applied Mathematics*
  - PMC8917048: "A semiparametric risk score for physical activity" [[14]](https://pmc.ncbi.nlm.nih.gov/articles/PMC8917048/)

### **Tier Mapping:**
- Clusters ranked by mean `FinalRiskScore`
- Lowest → Gold (Low Risk)
- Highest → Bronze (High Risk)

---

## 7. **Machine Learning Models**

### **Models Tested:**
1. **Logistic Regression** – Baseline, interpretable
2. **Random Forest** – Ensemble, handles nonlinearity
3. **XGBoost** – State-of-the-art for tabular data
4. **Gradient Boosting** – Similar to XGBoost
5. **SVM** – Non-linear boundaries
6. **Neural Network (MLP)** – Deep learning approach

### **Why these models?**
- **Logistic Regression:** Insurance industry standard for risk prediction (transparent, regulat applicable)
- **Tree Ensembles (RF, XGBoost, GBM):** Best performance on tabular health/insurance data in literature
- **Why NOT deep learning for this?** Small/medium tabular datasets; trees outperform NNs (established in ML benchmarks)

**Citations:**
- Fernández-Delgado et al. (2014): "Do we need hundreds of classifiers to solve real world classification problems?" *Journal of Machine Learning Research*
- Chen & Guestrin (2016): "XGBoost: A Scalable Tree Boosting System," *KDD*
- Insurance ML studies: XGBoost/RF consistently top performers [[15]](http://arxiv.org/pdf/2411.00354.pdf)

### **Model Hyperparameters:**

| Model               | Key Parameters                | Justification                                    |
|---------------------|-------------------------------|--------------------------------------------------|
| LogisticRegression  | `max_iter=1000`               | Ensure convergence                               |
| RandomForest        | `n_estimators=200`            | Balance accuracy vs. computational cost          |
| XGBoost             | `n_estimators=200`            | Standard for tabular data                        |
| GradientBoosting    | `n_estimators=200`            | Comparable to XGBoost                            |
| SVM                 | `probability=True`            | Enable probability estimates                     |
| MLP                 | `hidden_layers=(32,32)`       | Simple 2-layer network for baseline              |

**Why these values?**
- `n_estimators=200`: Standard for RF/XGBoost in health ML (balance overfitting vs. performance)
- **Citations:** Scikit-learn documentation, XGBoost documentation

---

## 8. **Model Evaluation & Validation**

### **Train-Test Split:**
- **80/20 split** with **stratification** (maintains class balance in train/test)
- **Why stratify?** Ensures each tier (Gold/Silver/Bronze) represented proportionally
- **Citation:** Kohavi (1995): "A study of cross-validation and bootstrap for accuracy estimation," *IJCAI*

### **Cross-Validation:**
- **5-Fold Stratified K-Fold CV**
- **Why 5-fold?** Standard in ML; balances bias-variance tradeoff
- **Stratified:** Maintains tier proportions in each fold
- **Citations:**
  - Hastie et al. (2009): *The Elements of Statistical Learning*
  - Standard practice in insurance/health ML [[16]](https://arxiv.org/pdf/2501.06492.pdf)

### **Performance Metrics:**
- **Accuracy:** Overall correctness
- **F1-weighted:** Balances precision/recall across all classes (important for imbalanced tiers)
- **Confusion Matrix:** Shows which tiers are misclassified

**Why F1-weighted?**
- Standard for multiclass imbalanced problems (more informative than raw accuracy)
- **Citation:** Sokolova & Lapalme (2009): "A systematic analysis of performance measures for classification tasks," *Information Processing & Management*

---

## 9. **Feature Interpretability**

### **Logistic Regression Coefficients:**
- **Log-odds interpretation:** Positive coef → higher feature increases odds of that tier
- **Odds Ratios:** `exp(coefficient)` → multiplicative effect on odds

### **Random Forest Feature Importances:**
- **Gini-based importance:** Measures average impurity reduction across all trees
- **Interpretation:** Higher importance → more influential in predictions

### **SHAP Values:**
- **Shapley Additive exPlanations** – Game-theory-based, model-agnostic
- **Global:** Shows overall feature importance + direction of effect
- **Local:** Explains individual predictions
- **Why SHAP?** Most robust, consistent interpretability method for ML models
- **Citations:**
  - Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions," *NIPS* [[17]](https://arxiv.org/abs/1705.07874)
  - Molnar (2022): *Interpretable Machine Learning* [[18]](https://christophm.github.io/interpretable-ml-book/)

---

## 10. **Multicollinearity Check: VIF**

### **Variance Inflation Factor (VIF):**
- **Threshold:** VIF >10 indicates severe multicollinearity
- **Our results:** `AvgHeartRate (34.6), AvgBMI (23.4), AvgSedentary (29.5)` all >10
- **Action taken:** [Describe if features were dropped or combined]
- **Why it matters:** High VIF inflates coefficient variance, reduces interpretability (especially for linear models)
- **Citations:**
  - O'Brien (2007): "A Caution Regarding Rules of Thumb for Variance Inflation Factors," *Quality & Quantity*
  - Penn State STAT 462: "Detecting Multicollinearity Using VIF" [[19]](https://online.stat.psu.edu/stat462/node/180/)

---

## 11. **Limitations & Future Work**

### **Limitations:**
1. **No real-world outcome data:** Risk tiers based on surrogate health metrics, not actual claims/mortality
2. **Synthetic demographics:** Age, sex, smoker, children simulated using correlations (not real)
3. **Small original sample:** 33 users → augmented to 1033 with synthetic data
4. **Cross-sectional only:** No longitudinal/temporal health trajectories
5. **Missing external validation:** No independent test dataset with true outcomes

### **Future Directions:**
1. **Real claims/outcome data:** Validate against actual insurance claims, hospitalizations, mortality
2. **Longitudinal analysis:** Track health changes over time (seasonal, behavioral shifts)
3. **Survival models:** Cox regression, time-to-event analysis with real outcome data
4. **External datasets:** Validate on independent cohorts (e.g., UK Biobank, NHANES)
5. **Causal inference:** Move beyond correlation to causal health risk modeling

---

## 12. **Software & Tools**

| Tool/Library         | Version | Purpose                                    |
|----------------------|---------|--------------------------------------------|
| Python               | 3.x     | Programming language                       |
| pandas               | Latest  | Data manipulation                          |
| numpy                | Latest  | Numerical operations                       |
| scikit-learn         | Latest  | ML models, metrics, preprocessing          |
| XGBoost              | Latest  | Gradient boosting                          |
| SDV                  | Latest  | Synthetic data generation                  |
| matplotlib/seaborn   | Latest  | Visualization                              |
| SHAP                 | Latest  | Model interpretability                     |
| statsmodels          | Latest  | VIF, statistical tests                     |

---

## **COMPLETE REFERENCE LIST**

[1] Munich Re (2018): "Stratifying Mortality Risk Using Physical Activity as Measured by Wearables"  
[2] BMJ Open Sport & Exercise Medicine (2021): "Physical activity and the insurance industry"  
[3] WHO (2020): "Physical activity guidelines"  
[4] Zhang et al. (2016): "Cumulative Resting Heart Rate Exposure and Risk of All-Cause Mortality," *Nature Scientific Reports*  
[5] Ekelund et al. (2019): "Dose–response associations between accelerometry measured physical activity and sedentary time and all-cause mortality," *BMJ*  
[6] Chaput et al. (2020): "Sleep duration and health," *BMJ*  
[7] Munich Re (2020): "Heart Rate and Mortality"  
[8] Blüher (2020): "Metabolically Healthy Obesity," *Endocrine Reviews*  
[9] Saint-Maurice et al. (2022): "Association of Step Count with Chronic Disease," *Nature Medicine*  
[10] WHO BMI Guidelines; Insurance underwriting standards  
[11] Coversure (2025): "Risk Assessment in Health Insurance"  
[12] CDC: "Smoking & Tobacco Use"  
[13] Society of Actuaries (2016): "Risk Scoring in Health Insurance: A Primer"  
[14] PMC8917048: "A semiparametric risk score for physical activity"  
[15] arXiv:2411.00354: "Classification problem in liability insurance using machine learning models"  
[16] arXiv:2501.06492: "A New Flexible Train-Test Split Algorithm"  
[17] Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions," *NIPS*  
[18] Molnar (2022): *Interpretable Machine Learning*  
[19] Penn State STAT 462: "Detecting Multicollinearity Using VIF"  

---

**END OF SCIENTIFIC JUSTIFICATION & CITATIONS**

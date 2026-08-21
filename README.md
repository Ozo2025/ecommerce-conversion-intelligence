# Ecommerce Conversion Intelligence

An end-to-end machine learning and LLM-powered application that predicts ecommerce purchase conversion from visitor session behavior and translates the prediction into actionable business intelligence.

The project combines traditional machine learning, MLflow experiment tracking, natural-language processing through the Nebius Token Factory API, input validation, automated testing, Streamlit, and Docker.

---

## Business Problem

Ecommerce teams collect large amounts of behavioral session data, but converting those signals into actionable sales intelligence can be difficult.

This project addresses that problem by creating an application that estimates whether an ecommerce visitor is likely to generate revenue based on session behavior.

Instead of requiring a business user to manually construct a structured model input, the application accepts a description of the visitor session in plain English.

For example:

> This is a returning visitor shopping in November. They viewed 25 product pages and spent 720 seconds on product-related pages. They visited 2 administrative pages for 45 seconds and 1 informational page for 20 seconds. Their bounce rate is 0.02, exit rate is 0.04, page value is 35.5, and special day score is 0. Operating system is 2, browser is 2, region is 1, traffic type is 2, and they are shopping on the weekend.

The application converts that description into structured features, validates the data, generates a conversion probability, and presents the result in business-friendly language.

---

## Project Architecture

```text
Natural-Language Session Description
                |
                v
        Nebius LLM Interface
                |
                v
        Structured Features
                |
                v
        Feature Validation
                |
                v
       Preprocessing Pipeline
                |
                v
     Random Forest Classifier
                |
                v
      Conversion Probability
                |
                v
        Nebius LLM Interface
                |
                v
 Business Summary & Recommendation
                |
                v
        Streamlit Interface
```

The LLM serves two roles in the application. First, it converts natural-language session descriptions into structured model features. After the machine learning model generates a prediction, the LLM converts the prediction and supplied session information into a concise business summary and recommendation.

---

## Dataset

The project uses the **Online Shoppers Purchasing Intention** dataset.

The dataset contains:

- 12,330 ecommerce sessions
- 17 predictive features
- 1 binary target: `Revenue`
- 10,422 non-converting sessions
- 1,908 converting sessions

This represents an approximate conversion rate of **15.47%**, creating a meaningful class-imbalance challenge.

### Features

The model uses:

- `Administrative`
- `Administrative_Duration`
- `Informational`
- `Informational_Duration`
- `ProductRelated`
- `ProductRelated_Duration`
- `BounceRates`
- `ExitRates`
- `PageValues`
- `SpecialDay`
- `Month`
- `OperatingSystems`
- `Browser`
- `Region`
- `TrafficType`
- `VisitorType`
- `Weekend`

Target:

```text
Revenue
```

The dataset itself is intentionally excluded from Git through `.gitignore`.

---

## Data Preprocessing

The preprocessing pipeline automatically identifies numerical and categorical features.

Numerical features are processed using imputation.

Categorical features are processed using imputation and one-hot encoding.

The original 17 model features are transformed into the numerical representation required by the machine learning models.

The preprocessing pipeline is included with the trained estimator so the same transformations are applied consistently during training and prediction.

---

## Machine Learning Experiments

Multiple model types and hyperparameter configurations were evaluated and tracked with MLflow.

Experiments included:

- Logistic Regression baseline
- Random Forest
- Tuned Random Forest
- Gradient Boosting
- Tuned Gradient Boosting

### Experiment Results

| Model | Configuration | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | Baseline | 0.8500 | 0.5107 | 0.7487 | 0.6072 | 0.8962 |
| Random Forest | 200 trees, depth 10 | 0.8662 | 0.5469 | 0.7932 | 0.6474 | 0.9240 |
| Random Forest | 500 trees, depth 15 | **0.8779** | 0.5839 | 0.7382 | **0.6520** | 0.9253 |
| Gradient Boosting | 100 estimators, depth 3, LR 0.10 | 0.9015 | **0.7235** | 0.5890 | 0.6494 | **0.9286** |
| Gradient Boosting | 200 estimators, depth 3, LR 0.05 | 0.9006 | 0.7175 | 0.5916 | 0.6485 | 0.9282 |

---

## Selected Model

The final application uses:

**Random Forest Classifier**

Configuration:

```text
n_estimators = 500
max_depth = 15
```

Performance:

```text
Accuracy:  0.8779
Precision: 0.5839
Recall:    0.7382
F1 Score:  0.6520
ROC-AUC:   0.9253
```

The Random Forest was selected because **F1 score was used as the primary model-selection metric**.

Because the dataset is imbalanced, accuracy alone could overstate model quality. F1 provides a more useful balance between precision and recall for identifying converting sessions.

Although Gradient Boosting produced a slightly higher ROC-AUC, the selected Random Forest achieved the strongest F1 score among the tested configurations.

---

## MLflow Experiment Tracking

MLflow is used to track:

- Model type
- Hyperparameters
- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Model artifacts
- Run status

The project includes:

```text
compare_experiments.py
```

This script retrieves MLflow experiment results and identifies the strongest run according to F1 score.

The application searches completed MLflow runs in descending F1-score order and loads the highest-performing model artifact that can be successfully loaded. This provides a fallback if a completed MLflow run has an unavailable or incomplete model artifact.

---

## LLM Integration

The application integrates an LLM through the **Nebius Token Factory OpenAI-compatible API**.

Model used during development:

```text
nvidia/Nemotron-3_5-Lightning
```

### Natural-Language Feature Extraction

The LLM converts natural-language descriptions into the exact structured features required by the machine learning model.

Example:

```text
"This is a returning visitor shopping in November..."
```

becomes structured data such as:

```json
{
  "ProductRelated": 25,
  "ProductRelated_Duration": 720,
  "Month": "Nov",
  "VisitorType": "Returning_Visitor",
  "Weekend": true
}
```

The extraction prompt explicitly instructs the LLM not to invent missing information.

### LLM-Generated Business Response

After the trained machine learning model generates a conversion prediction and probability, the LLM receives the model result and supplied structured session information.

It then generates:

- A concise business summary
- A practical ecommerce or sales recommendation

The response-generation prompt instructs the LLM to:

- State the model's conversion probability
- Describe the result as a model estimate rather than a guarantee
- Refer only to supplied session information
- Avoid claiming that an individual feature caused the prediction
- Provide one practical business action

The application also includes defensive JSON extraction logic so that valid structured responses can be recovered if the LLM includes additional text around its final JSON output.

---

## Missing-Input Protection

A prediction is generated only when all required model features are available.

For example, an incomplete request such as:

```text
This is a returning visitor who viewed 25 product pages in November.
```

does not trigger a prediction.

Instead, the application responds:

```text
I need more information before I can estimate conversion probability.
```

and identifies the missing features.

This prevents the system from silently inventing model inputs.

---

## Business Explanation Layer

After a successful prediction, the application converts the model output into business-facing information using the LLM response-generation layer.

The interface provides:

- Conversion probability
- Likely / unlikely conversion classification
- LLM-generated business summary
- LLM-generated recommended action
- Model-use disclaimer
- Expandable structured feature data

For a tested visitor session, the model produced approximately:

```text
Conversion Probability: 57.99%
Prediction: Likely to Convert
```

The LLM then generated a concise interpretation and recommended business action based on the model result and supplied session information.

The prediction is presented as decision support rather than a guarantee of customer behavior.

---

## Streamlit Interface

The project includes an interactive Streamlit application:

```text
streamlit_app.py
```

The interface allows a user to:

1. Describe a visitor session in natural language
2. Load an example session
3. Clear the current session
4. Analyze conversion likelihood
5. View conversion probability
6. View the classification result
7. Read an LLM-generated business interpretation
8. View an LLM-generated recommended action
9. Inspect the extracted model features
10. Receive clear feedback when required information is missing

Run locally with:

```bash
streamlit run streamlit_app.py
```

Then open the local address displayed by Streamlit.

---

## Automated Testing

The project contains automated tests for preprocessing, model behavior, LLM-powered natural-language parsing, and interface validation.

Test files:

```text
tests/test_preprocess.py
tests/test_model.py
tests/test_interface.py
```

The test suite includes:

- 4 preprocessing tests
- 2 model tests
- 3 interface and LLM integration tests

The interface tests verify valid feature handling, incomplete-input protection, and natural-language feature extraction using a mocked LLM response so the automated test suite does not require a live API call.

Run:

```bash
pytest tests/ -v
```

Verified result:

```text
9 passed
```

---

## Docker

The project can also run inside a Docker container.

Build the image:

```bash
docker build -t ecommerce-conversion-intelligence .
```

Run the container:

```bash
docker run --rm -p 8503:8501 --env-file .env ecommerce-conversion-intelligence
```

Then open:

```text
http://localhost:8503
```

The Docker build and containerized Streamlit application were successfully tested during development.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ozo2025/ecommerce-conversion-intelligence.git
cd ecommerce-conversion-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Copy:

```text
.env.example
```

to:

```text
.env
```

Configure:

```text
NEBIUS_API_KEY=your_api_key_here
NEBIUS_BASE_URL=https://api.tokenfactory.us-central1.nebius.com/v1/
NEBIUS_MODEL=nvidia/Nemotron-3_5-Lightning
```

Never commit the real `.env` file or API key to source control.

---

## Data Setup

The dataset is intentionally excluded from the repository.

Place the dataset at:

```text
data/online_shoppers_intention.csv
```

before training or running data-dependent scripts.

---

## Reproducing the Selected Model

The current `configs/config.yaml` is configured to reproduce the selected Random Forest model used by the application.

Selected configuration:

```yaml
model:
  type: random_forest
  n_estimators: 500
  max_depth: 15
```

The Random Forest was selected using F1 score as the primary evaluation metric. Among the tested configurations, it achieved the highest F1 score while maintaining strong ROC-AUC performance:

```text
Accuracy:  0.8779
Precision: 0.5839
Recall:    0.7382
F1 Score:  0.6520
ROC-AUC:   0.9253
```

To reproduce the selected model from a fresh environment:

1. Place `online_shoppers_intention.csv` in the `data/` directory.
2. Create and activate the Python virtual environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Train the model using the configuration in `configs/config.yaml`:

```bash
python -m src.train
```

5. Review the MLflow experiment results:

```bash
python compare_experiments.py
```

6. Run the automated test suite:

```bash
pytest tests/ -v
```

7. Copy `.env.example` to `.env` and add a valid Nebius API key.

8. Start the application:

```bash
streamlit run streamlit_app.py
```

The application searches completed runs from the local `ecommerce_conversion` MLflow experiment in descending F1-score order and loads the highest-performing model artifact that is available. Because MLflow's local database and artifacts are intentionally excluded from Git, a fresh clone must first run the training step before launching the application.

---

## Project Structure

```text
ecommerce-conversion-intelligence/
|
|-- configs/
|   `-- config.yaml
|
|-- data/
|   `-- .gitkeep
|
|-- models/
|   `-- .gitkeep
|
|-- src/
|   |-- __init__.py
|   |-- app.py
|   |-- evaluate.py
|   |-- llm_interface.py
|   |-- preprocess.py
|   `-- train.py
|
|-- tests/
|   |-- conftest.py
|   |-- test_interface.py
|   |-- test_model.py
|   `-- test_preprocess.py
|
|-- .env.example
|-- .gitignore
|-- compare_experiments.py
|-- Dockerfile
|-- README.md
|-- requirements.txt
`-- streamlit_app.py
```

---

## Key Technologies

- Python 3.12
- pandas
- NumPy
- scikit-learn
- MLflow
- OpenAI-compatible API
- Nebius Token Factory
- Streamlit
- pytest
- Docker
- Git / GitHub

---

## Project Highlights

This project demonstrates an end-to-end machine learning workflow including:

- Business problem definition
- Real-world structured dataset
- Data preprocessing
- Imbalanced classification
- Multiple model experiments
- Hyperparameter comparison
- MLflow experiment tracking
- Metric-driven model selection
- Model artifact loading
- LLM integration for both feature extraction and response generation
- Structured natural-language feature extraction
- Defensive LLM JSON parsing
- Input validation
- Graceful handling of incomplete requests
- Business-focused model interpretation
- Interactive Streamlit application
- Automated testing
- Docker containerization
- Secure API-key management
- Reproducible model configuration

---

## Reflection

This capstone brought together the full machine learning lifecycle in one application. The biggest lesson was that building a useful ML product requires more than achieving a strong model score. The preprocessing pipeline, experiment tracking, model selection, natural-language interface, validation logic, testing, containerization, and user experience all have to work together reliably.

One of the most challenging parts was integrating the LLM with the predictive model in a way that was both useful and dependable. The LLM needed to extract structured ecommerce features from conversational input without inventing missing values, and it also needed to explain the prediction clearly after the trained model produced a result. Handling incomplete inputs and inconsistent LLM output required additional validation and defensive parsing logic.

Another challenge was balancing model metrics on an imbalanced dataset. Accuracy alone was not sufficient because non-converting sessions represented the majority class. Comparing precision, recall, F1, and ROC-AUC across multiple experiments made the tradeoffs clearer and led to selecting the Random Forest configuration with the strongest F1 score.

With more time, I would expand the application with probability calibration, SHAP-based explanations, feature-importance visualizations, real-time ecommerce event ingestion, CRM integration, model-drift monitoring, and hosted cloud deployment. I would also evaluate whether different classification thresholds improve the business value of the conversion-prioritization workflow.

---

## Limitations and Future Improvements

The current system is a demonstration application built around historical ecommerce session data.

Potential future improvements include:

- Probability threshold optimization
- Additional model families
- Cross-validation
- Model calibration
- Feature importance visualization
- SHAP-based model explanations
- Real-time ecommerce event ingestion
- Customer and account-level history
- CRM integration
- Automated sales prioritization
- Production model monitoring
- Drift detection
- Hosted cloud deployment

The application should be treated as a decision-support system rather than a guarantee of individual customer behavior.

---

## Author

**Brad Oswald**

Capstone Project — End-to-End Machine Learning Application
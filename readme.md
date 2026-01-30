# Diabetes Health Indicator Prediction

## Project Overview 🔍
###  Brief description of the project

- This project is an end-to-end **Machine learning web application** that predicts a user’s health status related to  diabetes based on key medical and lifestyle indicators.
It combines `Machine Learning`, `Backend APIs`, `Frontend UI`, `Docker`, `CI/CD`, and AWS cloud deploymennt into a single production-ready system. 

### Purpose and goals
- Build **ML classification model** for diabetes prediction
- Set up reproducible environments using uv.
- Serve predictions using a scalable REST API (FastAPI)
- Provide an easy-to-use Streamlit web interface
- Automate testing and deploymennt using **CI/CD pipelines** 
- Containerize the application using Docker
- Deploy the application on **AWS**
- So that people can predict with the help of API whether there is diabetes or not.
### Key features
- Multi-class Diabetes Prediction (`Healthy`/`Pre-Diabetic`/`Diabetic`)
- Machine Learning model with proper preprocessing and  Model evaluation metrics (Weighted F1, Confusion Matrix)
- FastAPI backend for inference and health checks
- Streamlit frontend for user interaction
- Dockerized services
- CI/CD pipeline with GitHub Actions
- AWS deploymennt
### Dataset source
```bash
Dataset: CDC Diabetes Health Indicators(CDC)
```
## Quick Setup Instructions
- Prerequisites (Python version, dependencies)
- Installation steps
- Environment setup
- Basic configuration

### Prerequisites
- Python 3.10 or higher
- Git
- Docker (optional, for containerized deploymennt)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-health-indicator-prediction.git
   cd diabetes-health-indicator-prediction
   ```

2. Install UV package manager (if not already installed):
   ```bash
   curl -Ls https://astral.sh/uv/install.sh | sh
   ```

3. Install dependencies:
   ```bash
   uv sync --all-extras --dev
   ```

### Environment Setup
1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```

2. Verify installation:
   ```bash
   python -c "import fastapi, streamlit, sklearn; print('All dependencies installed successfully')"
   ```

### Basic Configuration
1. The application uses environment variables for configuration. Copy the example:
   ```bash
   cp .env.example .env  # If .env.example exists
   ```

2. Update the `.env` file with your settings (database URLs, API keys, etc.)

3. For local development, the default configuration should work out of the box.

### Run Locally(Without Docker)
- Start backend:
```bash
uvicorn run app/api.py 
```
- Start frontend:
```bash
streamlit run app/frontend.py
```


## In-Depth Documentation
### Detailed architecture explanation
This project follows a containerized, microservices-based architecture  for deploying a machine learning model in production.
- The Streamlit frontend collects user inputs and sends requests to the backend.
- The FastAPI backend exposes REST endpoints for prediction and health checks.
-The trained ML model and preprocessor are loaded at application startup and used only for inference.
- The backend is stateless, enabling easy scaling and reliable performance.
- Both frontend and backend run in separate Docker containers and communicate over HTTP.
- CI/CD with GitHub Actions ensures automated testing, linting, and Docker image builds.


### Data preprocessing details
- The raw data is first cleaned by removing duplicate,handling missing values & standardizing column names.
- StandardScaler use to fetch feature in same scale,So model learn easyly.
- The target column is identified and converted into a integer format for model training.
- Feature and target  variables are separated to avoid data leakage.
- Input feature divided into numerical & categorical columns.
- Numerical features are scaled using StandardScaler, while categorical features are encoded appropriately.
- The preprocessing pipeline is fitted only on training data and then saved as a .pkl file for reuse.
- Cleaned and transformed train and test datasets are saved for reproducibility.

### Model training information
- The cleaned and preprocessed dataset is used to train a multi-class classification model for predicting diabetes status.
- Features and target variables are separated, followed by a train–test split for validation.
- Model performance is evaluated using Weighted F1-score, making it suitable for handling class imbalance.
- Cross-validation is performed to ensure model stability and generalization & confusion matrix is generated to analyze class-wise prediction performance.
- The final trained model and preprocessing pipeline are saved as .pkl files for production inference.

### API endpoints documentation
| Method | Endpoint | Description |
|------|--------|------------|
| GET | /health | Health check |
| POST | /predict | Diabetes prediction |

#### Backend Fastapi 
- The backend expose RESTful endpoints using FastAPI for health monitoring and model inference.
- Health check for Verifies that the API service and model are running correctly.
- Accepts user health data and returns the predicted diabetes status.
- Request body in JSON payload containing numerical and categorical health features
- Response in Predicted class label (Healthy / Pre-Diabetic / Diabetic) & probabilities for each class.
#### Frontend (Streamlit)
- The frontend is built using Streamlit to provide a simple and interactive user interface.
- It allows users to enter health-related input features through a structured form.
- No prediction is triggered until the Submit button is clicked, ensuring controlled requests.
- On submission, the frontend sends a POST request to the FastAPI backend with the user input as JSON.
- The response from the backend is parsed and displayed as it in this format (Healthy / Pre-Diabetic / Diabetic) with Class-wise prediction probabilities.
- Database schema (if applicable)

## Project Structure
```bash
├── Assignment.md
├── Dockerfile.streamlit
├── Dockerfile.uvicorn
├── __pycache__
│   └── main.cpython-310.pyc
├── app
│   ├── __init__.py
│   ├── __pycache__
│   ├── api.py
│   └── frontend.py
├── confusion_matrix.png
├── data
│   ├── diabetes_cleaned.csv
│   └── processed
├── docker-compose.yml
├── docs
│   ├── Makefile
│   ├── build
│   ├── make.bat
│   ├── mlruns
│   ├── source
│   └── sphinx
├── images
│   ├── Screenshot 2026-01-12 194049.png
│   ├── Screenshot 2026-01-12 194101.png
│   ├── Screenshot 2026-01-12 194125.png
│   ├── Screenshot 2026-01-12 194150.png
│   ├── Screenshot 2026-01-12 194203.png
│   ├── Screenshot 2026-01-12 194222.png
│   ├── Screenshot 2026-01-12 194244.png
│   ├── Screenshot 2026-01-12 194351.png
│   ├── Screenshot 2026-01-12 194410.png
│   ├── Screenshot 2026-01-12 194446.png
│   ├── Screenshot 2026-01-12 194506.png
│   ├── Screenshot 2026-01-12 194519.png
│   ├── Screenshot 2026-01-12 194530.png
│   ├── Screenshot 2026-01-12 194557.png
│   ├── Screenshot 2026-01-12 194611.png
│   ├── Screenshot 2026-01-12 194644.png
│   ├── Screenshot 2026-01-12 194700.png
│   ├── Screenshot 2026-01-12 194715.png
│   ├── Screenshot 2026-01-12 194748.png
│   ├── Screenshot 2026-01-12 194937.png
│   ├── Screenshot 2026-01-12 195039.png
│   └── Screenshot 2026-01-12 195234.png
├── infra
│   ├── ecs
│   ├── eks
│   └── iam_policy.json
├── mlruns
│   ├── 0
│   └── 673199432592859409
├── models
│   ├── model.pkl
│   └── preprocessor.pkl
├── notebooks
│   ├── 01_eda.ipynb
│   └── 02_model_dev.ipynb
├── pyproject.toml
├── readme.md
├── requirement.txt
├── src
│   ├── __init__.py 
│   ├── __pycache__
│   ├── data_prep.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── tests
│   ├── __pycache__
│   ├── conftest.py
│   ├── test_data_prep.py
│   ├── test_inference.py
│   └── test_integration.py
└── uv.lock
```

### Root directory overview
 - The root directory contains all components required for data processing, model training, inference, backend API, frontend UI, testing, containerization, and CI/CD automation.
### Description of main folders 
- src/
    - Contains the complete machine learning pipeline, including data preprocessing, training, and inference.
- tests/
    - Includes unit and integration tests to validate data preprocessing, model inference, and API behavior.
- app/
    - It Holds the application -:
         - FastAPI backend for provide predictions
         - streamlit frontend for user interaction
  
- .github/workflows/
    -  It contains GitHub Actions CI pipelines for linting, testing, and Docker image builds.

### Key files explanation
- At the start of the project, download **UV**, then create a virtual environment and install the dependencies. 
- After that, download the dataset to observe and explore it in the notebooks.
- Read the dataset for cleaning and training, and identify the target column on which the model should be trained.

#### src/data_prep.py
- Implements data cleaning steps such as removing duplicates, handling missing values, and converting the target column to integer type.
- Before preprocessing, separates features and the target column, identifies numerical and categorical columns, applies **StandardScaler** to numerical columns, and performs encoding for categorical columns.
- **StandardScaler** is used to bring all features to the same scale so the model can learn more effectively.
- Saves the fitted preprocessing object as a `.pkl` file using `joblib`.
- The `main` function calls all defined functions, splits the data into train and test sets, fits the preprocessor on the training data (to prevent data leakage and ensure accurate predictions on unseen data), and saves the train and test data as CSV files.
- The `clean_health_data` function handles basic cleaning such as removing duplicates, handling NaN values, and stripping column names.
- The `build_preprocessor` function handles preprocessing for numerical and categorical columns.
#### src/train.py
- Loads the dataset and creates an MLflow experiment where all project runs are tracked under the experiment name **diabetes_classification**.
- Sets up logging in the format `(time, levelname, message)` to display logs in the terminal.
- The script includes a data-loading function that reads the input CSV file and logs the data-loading process. Inside the main training pipeline, the dataset is loaded, and a validation check is performed to ensure that the specified target column exists in the DataFrame. The features and target variable are then separated.
-A train–validation split is performed based on the test_size argument:
   - If `test_size > 0`, the dataset is split into training and validation sets.
   - If `test_size = 0`, the entire dataset is used for training.

- The script loads a previously saved preprocessing pipeline and applies it to transform the training (and validation) features before model training.
- A **RandomForestClassifier** is initialized, and MLflow is used to log all important hyperparameters. Optional cross-validation is performed to evaluate model stability using the weighted F1-score. The model is then trained on the transformed training data.
- If a validation set is available, the model is evaluated using accuracy, weighted F1-score, and a classification report. A confusion matrix is generated to visualize correct and incorrect predictions, saved as a .png file, and logged as an MLflow artifact.
- Finally, the trained model is saved as a .pkl file. Depending on the configuration, the model can be saved either alone or together with the preprocessor. All relevant artifacts and metrics are logged to MLflow for experiment tracking and reproducibility.
#### src/predict.py 
- Loads `model.pkl` and `preprocessor.pkl`.
- Aligns input data columns in the exact order and shape used during training.
- Implements a predict function that converts input data to a DataFrame if necessary, applies preprocessing, performs predictions, and returns prediction probabilities.
#### tests/test_data_prep.py
- Creates` test_clean_health_data`, which imports and tests the `clean_health_data` function from `src/data_prep.py`.
- Creates `test_preprocessor`, which imports and tests the `build_preprocessor` function to validate transformed columns.
#### tests/test_inference.py
- Creates the `test_predictor_load()` function to test loading of the model and preprocessor.
- Creates `test_predict_single_row(monkeypatch)`, which initializes a predictor object, uses pytest’s `monkeypatch` to override the model’s `predict()` method, and returns a fixed output.
#### tests/test_integration.py
- Converts the real predictor object into a fake predictor object using `MagicMock` and `@patch` in the `app/api.py` file.
#### app/api.py
- Implements the FastAPI backend and defines a `CLASS_MAP` variable for readable prediction outputs.
- The `app` variable sets the API title, description, and version visible in the browser.
- Loads the model and preprocessor using `model_predictor`.
- Collects user input using the `PredictRequest` class with integer and float fields.
- Ensures the prediction response format is consistent.
- Includes a health-check endpoint.
- In the final stage, converts input data into a dictionary, calls the prediction function, processes the data, and returns the final response.
#### app/frontend.py
- The `API_URL` variable stores the backend FastAPI endpoint.
- Sets page configuration such as page title and layout.
- Creates a form to collect user input, with a submit button; predictions are triggered only after submission.
- On submission, creates a payload and sends a POST request to the backend using `requests.post(API_URL, json=payload)`.
- Parses the JSON response and displays results such as **Diabetic**, **Healthy**, or *Pre-diabetic*, along with prediction probabilities.
#### Dockerfile.uvicorn
- Creates a base image and sets the working directory inside the container.
- Installs system dependencies and the UV environment.
- Copies all project files into the container and installs dependencies.
- Exposes **port 8000** and runs the application using:
    `CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]`
#### Dockerfile.streamlit
- Creates a base image and sets the working directory.
- Installs system dependencies and the UV environment.
- Copies project files and installs all dependencies.
- Exposes **port 8501** and runs the Streamlit app using:
    `CMD ["streamlit", "run", "app/frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]`
#### docker-compose.yml
- Defines Docker Compose version **3.8**.
- Creates services for **uvicorn** and **streamlit**.
- Specifies build configuration, image names, and port mappings (`8000:8000`).
- Defines a shared network to enable communication between the frontend and backend.
#### .github/workflows/ci.yml
- Defines a GitHub CI pipeline triggered on pushes to the `main` branch.
- Runs jobs on `ubuntu-latest` with the following steps:
   1. Checkout repository
   2. Set up Python version
   3. Install UV environment
   4. Install dependencies
   5. Run Ruff lint checks (unused imports, syntax, and style issues)
   6. Run Ruff format checks
   7. Run tests
   8. Build API Docker image
   9. Build Streamlit Docker image

## Docker Setup and Run Instructions
### Docker prerequisites
- Docker installed and running
- Docker Compose installed
- Basic knowledge of Docker commands
### Building Docker images
Build the backend and frontend images using Docker Compose:
```bash
 docker-compose build 
 ```
### Running containers locally
Start all services (FastAPI backend and Streamlit frontend):
```bash
docker-compose up
```
Stop the containers:
```bash
docker-compose down
```

- Environment variables
### Port mappings
- FastAPI backend: 8000:8000
- Streamlit frontend: 8501:8501

- Volume mounts

## AWS Deploymennt Instructions
- Prerequisites (AWS account, CLI setup)
### ECR setup
- Two ECR repositories are created:
   - One for the backend
   - One for the frontend
- Docker images are built locally and authenticated with ECR.
- Images are pushed to their respective repositories.
- ECR repositories and Docker image connection 
   - ![ECR repositories](images/Screenshot%202026-01-12%20194049.png)
### ECS setup
- Create Cluster
   - An ECS cluster is created using the AWS Management Console.
   - The cluster is used to run containerized backend and frontend services.
- ECS Cluster creation
   - ![Cluster image](images/Screenshot%202026-01-12%20194150.png)
   - ![Cluster image2](images/Screenshot%202026-01-12%20194203.png)

- Separate task definitions are created for backend and frontend containers.
- Task definitions include:
   - Container image from ECR
   - Port mappings
   - CPU and memory configuration
- Task definition configuration
      - ![TF image](images/Screenshot%202026-01-12%20194700.png)
      - ![Backend task](images/Screenshot%202026-01-12%20194715.png)
      - ![Frontend task](images/Screenshot%202026-01-12%20194748.png)

- Two ECS services are created:
   - Backend service
   - Frontend service
- Services ensure containers remain running and accessible.
- Each service runs its respective task definition.
- ECS services overview
   - ![Services imacge](images/Screenshot%202026-01-12%20194222.png)
   - ![Backend Service img](images/Screenshot%202026-01-12%20194410.png)
   - ![Frontend service img](images/Screenshot%202026-01-12%20194506.png)

### Networking configuration
  - VPC setup
    - The application is deployed using the AWS Management Console within the default VPC.
    - ![vpc setup](images/Screenshot%202026-01-16%20153645.png)
    - ![subnets](images/Screenshot%202026-01-16%20153703.png)
  - Security groups
    - Security Groups are configured to control inbound and outbound traffic.
    - ![security group](images/Screenshot%202026-01-16%20154413.png)
    - ![security group](images/Screenshot%202026-01-16%20154426.png)

### Application Running
- The backend and frontend applications are successfully running on ECS.
- The application is accessible through a public endpoint.
   - ![Running Backend api](images/Screenshot%202026-01-12%20194937.png)
   - ![Runing Frontend](images/Screenshot%202026-01-12%20195039.png)
   - ![predicted Result](images/Screenshot%202026-01-12%20195234.png)

## Usage
### How to use the API
  - This API  takes health-related input data from the user and predicts whether the user has diabetes or Non-Diabetic using a trained Machine Learning model.
  - The prediction is returned in JSON format.

### Example requests
- API Endpoint
   - Post/predict
- Request body (JSON)
   - {"age": 45,"bmi": 28.6,"blood_pressure": 80,"glucose": 150,"insulin": 130}
- Response (JSON)
   - {"prediction": "Diabetic","confidence": 0.87}
### Frontend interaction
- User enter health detail (age, BMI,blood_pressure, glucose, etc.) in the frontend form.
- On clicking submit then frontend send a POST request to the backend API.
- Then Backend pass the data to trained model then predict result send back to frontend.
- Frontend Shows the results with Class-wise prediction probabilities. 
   - Prediction: Diabetic/healthy/pre-diabetic
 
- Testing instructions

## Development
- Setting up development environment
- Running tests
- Code formatting and linting
- Contributing guidelines

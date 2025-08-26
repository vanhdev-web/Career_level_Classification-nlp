# Career Level Classification - NLP

This project provides a comprehensive solution for classifying job descriptions into different career levels using Natural Language Processing (NLP) techniques. It features a user-friendly web-based interface built with Streamlit, allowing users to input job details and receive real-time career level predictions. The backend leverages a robust machine learning pipeline for data preprocessing, feature engineering, model training, and evaluation.

**Table of Content**
1. [Dataset Overview](#dataset-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Tech Stack](#tech-stack)
6. [Project Structure](#project-structure)
7. [Contributing](#contributing)
8. [License](#license)

## Dataset Overview
**Dataset statistics**  

![](https://drive.google.com/uc?id=1sK-UoyWNzGvmEJnBfrYlzSC9NHBG2W5F) 

**Correlation Matrix**  

![](https://drive.google.com/uc?id=1LyOCh5WvDJLgrwZkXjdRfYvCWVypmynB)
![](https://drive.google.com/uc?export=view&id=1KcLg8O-wPxTUgzFRFPLnYbVF_7p_vO-9)

**Sample**  

![](https://drive.google.com/uc?id=1f-BQeYkyG-h5PG9HelaevJ7lCx-T7n4Z)

## Features

*   **Interactive Web User Interface**: A Streamlit application for intuitive input of job title, location, description, function, and industry.
*   **Real-time Career Level Prediction**: Instantly classify job descriptions into career levels based on the trained NLP model.
*   **Comprehensive NLP Model Training Pipeline**: Includes modules for data loading, cleaning, text preprocessing, and feature extraction.
*   **Hybrid Feature Engineering**: Supports both text-based features (TF-IDF) and categorical features (One-Hot Encoding).
*   **Imbalanced Data Handling**: Incorporates techniques like `RandomOverSampler` and `SMOTEN` to address class imbalance during model training.
*   **Model Persistence**: Trained models are saved using `pickle` for efficient deployment and faster prediction inference.
*   **Automated Model Evaluation**: Utilizes `LazyPredict` to quickly evaluate and compare the performance of multiple classification algorithms.
*   **Form Reset Functionality**: Convenient button to clear input fields in the web application.

## Installation

To get this project up and running on your local machine, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/vanhdev-web/Career_level_Classification-nlp.git
    cd Career_level_Classification-nlp
    ```

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

This project has two main components: the model training script and the interactive web application.

### 1. Model Training (Optional, if you want to retrain the model)

The `nlp.py` script is responsible for loading the data, preprocessing it, training the machine learning model, and saving the trained model as `career_model.pkl`. The pre-trained model is already included in the repository, so this step is only necessary if you wish to retrain or modify the model.

To run the training script:
```bash
python nlp.py
```
This will generate (or update) the `career_model.pkl` file.

### 2. Running the Web Application

The `app.py` file contains the Streamlit web application. After installing the dependencies, you can launch the application from your terminal:

```bash
streamlit run app.py
```

This command will open a new tab in your web browser displaying the "Career Level Classification" application. You can then fill in the job details and click the "Predict" button to see the predicted career level.

## Tech Stack

The project is built using the following technologies:

*   **Language**: Python
*   **Web Framework**: Streamlit
*   **Data Manipulation**: Pandas
*   **Machine Learning**: Scikit-learn
*   **Imbalanced Data Handling**: Imbalanced-learn
*   **Automated ML**: LazyPredict
*   **Model Persistence**: Pickle
*   **Regular Expressions**: `re` module

## Project Structure

```
.
├── app.py                      # Streamlit web application for prediction
├── career_model.pkl            # Pre-trained machine learning model
├── final_project.ods           # Raw dataset used for model training
├── jobs_levels_report.html     # (Potentially) an analysis report or evaluation result
├── nlp.py                      # Script for NLP model training, preprocessing, and persistence
└── requirements.txt            # List of Python dependencies
```

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

## License

This project is open-source and available under the [MIT License](LICENSE).
*(Note: A LICENSE file is not present in the provided structure, so this is a placeholder. Please add an actual LICENSE file to your repository if you intend to use the MIT License or any other.)*

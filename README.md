# Student Performance Prediction

## Project Overview
This project uses Machine Learning regression models to predict students' final marks based on their study hours, attendance percentage, and previous scores.

The project covers data inspection, cleaning, exploratory data analysis (EDA), model training, and performance evaluation.

## Dataset
- **Rows:** 1,000 generated student records (before cleaning).
- **Features:** StudyHours, AttendancePct, PreviousScore.
- **Target:** FinalMarks.
- **Note:** This is a synthetically generated dataset created for learning and practice, not real student data.

## Technologies and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Machine Learning Models
The following regression models were trained and compared:
- Linear Regression
- Decision Tree Regressor (Unlimited Depth)
- Decision Tree Regressor (Maximum Depth 3)
- Decision Tree Regressor (Maximum Depth 5)

## Model Evaluation

Models were compared using Mean Absolute Error (MAE) on the test dataset.

| Model | Test MAE |
|---|---:|
| Linear Regression | 4.92 |
| Decision Tree (Unlimited) | 6.80 |
| Decision Tree (Depth 3) | 6.94 |
| Decision Tree (Depth 5) | 5.91 |

Linear Regression achieved the lowest test MAE among the models on this train/test split.

## Project Files
- `Regression_student_performance.ipynb` — Jupyter Notebook containing the complete analysis and model training.
- `Regression_student_performance.csv` — Dataset used in the project.
- `Regression_student_performance.py` — Python script exported from the notebook.

## How to Run
1. Clone this repository.
2. Install the required libraries:
   `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Open the notebook in Jupyter Notebook or Google Colab.
4. Run the cells in order.

## Conclusion
This project demonstrates a complete beginner-level regression workflow, from data preparation and exploratory analysis to model training, evaluation, and comparison.

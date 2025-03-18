#!/usr/bin/env python

# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go
from textblob import TextBlob

# Load the dataset
data = pd.read_csv('student_assessment_data.csv')

# --- Data Exploration and Preprocessing ---
# Calculate descriptive statistics
average_score = data['quiz_score'].mean()
print(f'Average Quiz Score: {average_score}')

completion_rate = data['completion_status'].value_counts(normalize=True) * 100
print(f'Completion Rate:\n{completion_rate}')

average_participation = data['participation_score'].mean()
print(f'Average Participation Score: {average_participation}')

average_content_quality = data['content_quality_rating'].mean()
print(f'Average Content Quality Rating: {average_content_quality}')

average_engagement = data['engagement_score'].mean()
print(f'Average Engagement Score: {average_engagement}')

pass_rate = (data['quiz_score'] >= 60).mean()  # Assuming 60 is the passing grade
print(f'Pass Rate: {pass_rate:.2f}%')

# --- Data Visualization ---
# Visualizing quiz scores
plt.figure(figsize=(10, 6))
plt.bar(data['student_name'], data['quiz_score'], color='skyblue')
plt.title('Quiz Scores by Student')
plt.xlabel('Student Name')
plt.ylabel('Quiz Score')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Visualizing engagement scores
plt.figure(figsize=(10, 6))
plt.bar(data['student_name'], data['engagement_score'], color='lightgreen')
plt.title('Engagement Scores by Student')
plt.xlabel('Student Name')
plt.ylabel('Engagement Score')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Visualizing participation scores
plt.figure(figsize=(10, 6))
plt.bar(data['student_name'], data['participation_score'], color='orange')
plt.title('Participation Scores by Student')
plt.xlabel('Student Name')
plt.ylabel('Participation Score')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Visualizing content quality ratings
plt.figure(figsize=(10, 6))
plt.bar(data['student_name'], data['content_quality_rating'], color='purple')
plt.title('Content Quality Ratings by Student')
plt.xlabel('Student Name')
plt.ylabel('Content Quality Rating')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Visualizing the pass rate
plt.figure(figsize=(10, 6))
plt.bar(['Pass Rate'], [pass_rate], color='gold')
plt.title('Pass Rate of Students')
plt.ylabel('Percentage (%)')
plt.ylim(0, 100)
plt.grid(axis='y')
plt.show()

# --- Machine Learning ---
# Prepare data for the model
data['pass'] = (data['quiz_score'] >= 50).astype(int)  # 1 if passed, 0 otherwise

# Correct way to select features
features = data[['attempts', 'time_spent', 'engagement_score', 'content_quality_rating', 'participation_score']]
target = data['pass']
print(target.value_counts())

# Identify numerical features
numerical_features = features.select_dtypes(include=np.number).columns.tolist()

# Ajustement de n_quantiles
n_samples = len(data)
n_quantiles = min(10, n_samples)
# Create preprocessing pipelines for numerical features

# Créer des pipelines de prétraitement pour les caractéristiques numériques
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('quantile', QuantileTransformer(output_distribution='normal', n_quantiles=n_quantiles))
])
# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features)
    ]
)
# Apply transformations

#Define models
logistic_regression = LogisticRegression(random_state=42)
random_forest = RandomForestClassifier(random_state=42)
gradient_boosting = GradientBoostingClassifier(random_state=42)
mlp_classifier = MLPClassifier(
random_state=42, max_iter=300) # Neural network
svm_classifier = SVC(
probability=True,
random_state=42) # Support Vector Machine, probability needed for ROC

pipeline_logistic = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', logistic_regression)
])

pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', random_forest)
])

pipeline_gb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', gradient_boosting)
])

pipeline_mlp = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', mlp_classifier)
])

pipeline_svm = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', svm_classifier)
])

# Split data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target)  # Stratify to maintain class distribution

# Train and evaluate models
pipelines = {
    'Logistic Regression': pipeline_logistic,
    'Random Forest': pipeline_rf,
    'Gradient Boosting': pipeline_gb,
    'MLP': pipeline_mlp,
    'SVM': pipeline_svm
}


model_results = {}
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)  # Gestion de la précision

    # Calculate probabilities only if both classes are present
    if len(np.unique(y_test)) > 1:
        y_prob = pipeline.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
        auc_roc = roc_auc_score(y_test, y_prob)
    else:
        auc_roc = None  # Set AUC-ROC to None if only one class is present

    model_results[name] = {
        'accuracy': accuracy,
        'classification_report': report,
        'auc_roc': auc_roc,
        'pipeline': pipeline,
        'y_prob': y_prob if 'y_prob' in locals() else None  # Store probabilities if calculated
    }

# Display results
for name, results in model_results.items():
    print(f"Results for {name}:")
    print(f"Accuracy: {results['accuracy']}")
    print(f"AUC-ROC: {results['auc_roc'] if results['auc_roc'] is not None else 'Not applicable'}")
    print(f"Classification Report:\n{results['classification_report']}")
    print("\n")
#--- Model Selection and Threshold Optimization ---
#Select the best model (e.g., based on AUC-ROC)
best_model_name = max(model_results, key=lambda k: model_results[k]['auc_roc'])
best_model = model_results[best_model_name]['pipeline']
y_prob_best = model_results[best_model_name]['y_prob']

#Optimize threshold for the best model
fpr, tpr, thresholds = roc_curve(y_test, y_prob_best)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f'Optimal threshold for {best_model_name}: {optimal_threshold:.2f}')

#Evaluate the best model with the optimized threshold
y_pred_optimal = (y_prob_best >= optimal_threshold).astype(int)
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
report_optimal = classification_report(y_test, y_pred_optimal)
auc_roc_optimal = roc_auc_score(y_test, y_prob_best)

print(f'Results for {best_model_name} with optimal threshold:')
print(f'Accuracy: {accuracy_optimal:.2f}')
print(f'Classification Report:\n{report_optimal}')
print(f'AUC-ROC: {auc_roc_optimal:.2f}\n')

#Identify at-risk students based on the best model and optimized threshold
at_risk_students = X_test[y_prob_best < optimal_threshold]
print("At-risk students based on engagement metrics:")
print(at_risk_students)

#--- Qualitative Feedback and Sentiment Analysis ---
#Sample qualitative feedback data (replace with real data)
qualitative_data = {
    'student_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'student_name': [
        'Olivia', 'Lucas', 'Emma', 'Noah', 'Ava', 'James', 'Sophia', 'Ben', 'Liam', 'Isabella'
    ],
    'qualitative_feedback': [
        "Great effort, but struggled with the final section.",
        "Excellent understanding of the core concepts.",
        "Needs to improve time management during quizzes.",
        "Shows potential but needs more consistent effort.",
        "Outstanding performance and engagement.",
        "Struggling with basic concepts, needs extra help.",
        "Good grasp of the material, but needs to participate more.",
        "Very low engagement, needs significant improvement.",
        "Satisfactory performance, can aim higher.",
        "Needs to focus more on understanding the material."
    ]
}
df_qualitative = pd.DataFrame(qualitative_data)

# Perform sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Polarity: -1 (negative) to +1 (positive)
    sentiment_score = analysis.sentiment.polarity
    return sentiment_score

df_qualitative['sentiment_score'] = df_qualitative['qualitative_feedback'].apply(analyze_sentiment)

# Categorize sentiment
def categorize_sentiment(score):
    if score > 0.2:
        return 'Positive'
    elif score < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

df_qualitative['sentiment'] = df_qualitative['sentiment_score'].apply(categorize_sentiment)




# Merge qualitative data with the main DataFrame
df = pd.merge(data, df_qualitative, on=['student_id', 'student_name'], how='left')

# --- Personalized Feedback Generation ---
def generate_feedback(student_id, student_name, quiz_score, attempts, time_spent, engagement_score, content_quality_rating, attendance_rate, qualitative_feedback, sentiment_score):
    feedback = f"Feedback for {student_name} (Student ID: {student_id}):\n"

    # Check quiz score
    if quiz_score < 50:
        feedback += "Quiz Performance: Needs Improvement\n"
        feedback += "Recommendations: Review the material and consider attending additional study sessions.\n"
    elif quiz_score < 75:
        feedback += "Quiz Performance: Satisfactory\n"
        feedback += "Recommendations: Good effort! Focus on areas where you lost points.\n"
    else:
        feedback += "Quiz Performance: Excellent\n"
        feedback += "Recommendations: Keep up the great work! Continue to challenge yourself.\n"

    # Check engagement score
    if engagement_score < 5:
        feedback += "Engagement Level: Low\n"
        feedback += "Recommendations: Increase participation in class activities and discussions.\n"

    # Check attendance rate
    if attendance_rate < 80:
        feedback += "Attendance Rate: Below Average\n"
        feedback += "Recommendations: Aim to attend all classes to enhance your learning experience.\n"

    # Check content quality rating
    if content_quality_rating < 3:
        feedback += "Content Quality Rating: Needs Attention\n"
        feedback += "Recommendations: Provide feedback on course materials to improve quality.\n"

    # Incorporate qualitative feedback and sentiment
    feedback += f"Instructor's Feedback: {qualitative_feedback}\n"
    feedback += f"Sentiment Analysis: {sentiment_score:.2f} ({categorize_sentiment(sentiment_score)})\n"

    return feedback

# Generate feedback for each student
for index, row in df.iterrows():
    feedback = generate_feedback(
        row['student_id'], row['student_name'], row['quiz_score'], row['attempts'],
        row['time_spent'], row['engagement_score'], row['content_quality_rating'],
        row['attendance_rate'], row['qualitative_feedback'], row['sentiment_score']
    )
    print(feedback)

# --- Clustering ---
# Select features for clustering
features_cluster = df[['quiz_score', 'engagement_score', 'participation_score']]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_cluster)

# Determine the optimal number of clusters using the Elbow method
inertia = []
K = range(1, 6)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Explicitly set n_init
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(K)
plt.grid()
plt.show()

# Choose the optimal number of clusters (e.g., 3 based on the Elbow method)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)  # Explicitly set n_init
df['cluster'] = kmeans.fit_predict(features_scaled)

# Automated feedback based on clusters
def generate_feedback2(cluster):
    if cluster == 0:
        return "Excellent performance! Keep up the great work and continue to engage actively."
    elif cluster == 1:
        return "Good job! You are performing well, but consider increasing your engagement in class activities."
    elif cluster == 2:
        return "It seems you may need additional support. Please reach out for help and consider focusing on your participation."

# Apply feedback generation
df['feedback'] = df['cluster'].apply(generate_feedback2)

print(df[['student_id', 'student_name', 'cluster', 'feedback']])

# Display the clustered data with feedback
print(df[['student_name', 'quiz_score', 'engagement_score', 'participation_score', 'cluster', 'feedback']])

#--- Interactive Dashboard with Dash ---
#Create a copy of the DataFrame for Dash
df2 = df.copy()

#Create the Dash app
app = Dash(__name__)
server = app.server

# Define the app layout
app.layout = html.Div([
    html.H1(
        children='Tableau de bord personnalisé des étudiants',
        style={'textAlign': 'center'}
    ),
    dcc.Dropdown(
        options=[{'label': name, 'value': name} for name in df2.student_name.unique()],
        value='Alice',
        id='dropdown-selection'
    ),
    html.Div(
        id='metrics-content',
        style={
            'textAlign': 'center',
            'marginTop': '20px'
        }
    ),
    dcc.Graph(id='graph-content')
])

# Define the callback to update the dashboard
@app.callback(
    [Output('metrics-content', 'children'),
     Output('graph-content', 'figure')],
    [Input('dropdown-selection', 'value')]
)
def update_dashboard(value):
    dff = df2[df2.student_name == value]
    metrics = [
        html.H4(children='Métriques pour l\'étudiant sélectionné'),
        html.P(f'Nom : {dff["student_name"].values[0]}'),
        html.P(f'Score du quiz : {dff["quiz_score"].values[0]}'),
        html.P(f'Score d\'engagement : {dff["engagement_score"].values[0]}'),
        html.P(f'Score de participation : {dff["participation_score"].values[0]}'),
        html.P(f'Évaluation de la qualité du contenu : {dff["content_quality_rating"].values[0]}'),
        html.P(f'Taux de complétion : {completion_rate[dff["completion_status"].values[0]]:.2f}%'),
        html.P(f'Feedback : {dff["feedback"].values[0]}'),
        html.P(f'Sentiment : {dff["sentiment"].values[0]}')
    ]
    fig = go.Figure(data=[
        go.Bar(
            name='Quiz Score',
            x=dff['student_name'],
            y=dff['quiz_score']
        ),
        go.Bar(
            name='Engagement Score',
            x=dff['student_name'],
            y=dff['engagement_score']
        ),
        go.Bar(
            name='Participation Score',
            x=dff['student_name'],
            y=dff['participation_score']
        )
    ])
    fig.update_layout(barmode='group', title='Scores des étudiants')
    return metrics, fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[17]:


# Load the dataset
data = pd.read_csv('student_assessment_data.csv')


# In[18]:


# Calculer la moyenne des scores de quiz
average_score = data['quiz_score'].mean()
print(f'Average Quiz Score: {average_score}')

# Calculer le taux de complétion
completion_rate = data['completion_status'].value_counts(normalize=True) * 100
print(f'Completion Rate:\n{completion_rate}')

# Calculer la moyenne des scores de participation
average_participation = data['participation_score'].mean()
print(f'Average Participation Score: {average_participation}')

# Calculer la moyenne des évaluations de qualité du contenu
average_content_quality = data['content_quality_rating'].mean()
print(f'Average Content Quality Rating: {average_content_quality}')

# Calculer la moyenne des scores d'engagement
average_engagement = data['engagement_score'].mean()
print(f'Average Engagement Score: {average_engagement}')

# Calculer le pourcentage d'étudiants ayant réussi le quiz
pass_rate = (data['quiz_score'] >= 60).mean() * 100  # Supposant que 60 est la note de passage
print(f'Pass Rate: {pass_rate:.2f}%')

# Visualiser les scores de quiz
plt.figure(figsize=(10, 6))
plt.bar(data['student_name'], data['quiz_score'], color='skyblue')
plt.title('Quiz Scores by Student')
plt.xlabel('Student Name')
plt.ylabel('Quiz Score')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Visualiser les scores d'engagement
plt.figure(figsize=(10, 6))
plt.bar(data['student_name'], data['engagement_score'], color='lightgreen')
plt.title('Engagement Scores by Student')
plt.xlabel('Student Name')
plt.ylabel('Engagement Score')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Visualiser les scores de participation
plt.figure(figsize=(10, 6))
plt.bar(data['student_name'], data['participation_score'], color='orange')
plt.title('Participation Scores by Student')
plt.xlabel('Student Name')
plt.ylabel('Participation Score')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Visualiser les évaluations de qualité du contenu
plt.figure(figsize=(10, 6))
plt.bar(data['student_name'], data['content_quality_rating'], color='purple')
plt.title('Content Quality Ratings by Student')
plt.xlabel('Student Name')
plt.ylabel('Content Quality Rating')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Visualiser le taux de réussite
plt.figure(figsize=(10, 6))
plt.bar(['Pass Rate'], [pass_rate], color='gold')
plt.title('Pass Rate of Students')
plt.ylabel('Percentage (%)')
plt.ylim(0, 100)
plt.grid(axis='y')
plt.show()


# In[19]:


# Partie Machine Learning
# Préparation des données pour le modèle
data['pass'] = (data['quiz_score'] >= 50).astype(int)  # 1 si réussi, 0 sinon
features = data[['attempts', 'time_spent', 'engagement_score', 'content_quality_rating', 'participation_score']]
target = data['pass']
print(target.value_counts())
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialiser et entraîner le modèle de régression logistique
model = LogisticRegression()
model.fit(X_train, y_train)

# Faire des prédictions
y_pred = model.predict(X_test)

# Évaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model: {accuracy:.2f}')
print(classification_report(y_test, y_pred))


# In[20]:


# Get predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of passing
# Afficher les probabilités prédites
print(y_prob)
# Set a threshold to identify at-risk students (e.g., probability < 0.5)
threshold = 0.7
at_risk_students = X_test[y_prob < threshold]

# Display at-risk students
print("At-risk students based on engagement metrics:")
print(at_risk_students)


# In[21]:


# Sample data: Student engagement metrics
# data = {
#     'student_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'student_name': ['Olivia', 'Lucas', 'Emma', 'Noah', 'Ava', 'James', 'Sophia', 'Ben', 'Liam', 'Isabella'],
#     'quiz_score': [85, 75, 90, 65, 95, 40, 80, 30, 70, 30],
#     'attempts': [2, 1, 1, 3, 1, 2, 1, 3, 2, 1],
#     'completion_status': ['Completed'] * 10,
#     'time_spent': [60, 45, 50, 70, 30, 55, 35, 80, 45, 50],
#     'engagement_score': [7, 6, 8, 5, 9, 4, 8, 3, 6, 5],
#     'content_quality_rating': [4, 3, 5, 3, 4, 2, 4, 2, 3, 3],
#     'participation_score': [5, 4, 5, 4, 5, 3, 4, 2, 4, 3],
#     'average_time_spent': [30, 45, 50, 60, 30, 45, 35, 60, 40, 50],
#     'percentage_answered': [90, 100, 95, 80, 100, 70, 95, 60, 85, 75],
#     'feedback_score': [4, 5, 5, 3, 5, 2, 4, 2, 3, 3],
#     'attendance_rate': [95, 90, 92, 85, 92, 75, 88, 70, 82, 80],
#     'learning_style': ['Visual', 'Auditory', 'Kinesthetic', 'Visual', 'Auditory', 'Kinesthetic', 'Visual', 'Auditory', 'Kinesthetic', 'Visual'],
#     'motivation_level': [8, 7, 9, 6, 8, 5, 9, 4, 6, 1]
# }

# Create a DataFrame
df = pd.DataFrame(data)


# In[22]:


def generate_feedback(student_id, student_name, quiz_score, attempts, time_spent, engagement_score, content_quality_rating, 
                      attendance_rate):
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

    return feedback


# In[23]:


# Generate feedback for each student
for index, row in df.iterrows():
    feedback = generate_feedback(row['student_id'], row['student_name'],row['quiz_score'], row['attempts'],row['time_spent'],
                                 row['engagement_score'],row['content_quality_rating'], 
                                 row['attendance_rate'])
    print(feedback)
    
   


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select features for clustering
features = df[['quiz_score', 'engagement_score', 'participation_score']]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Determine the optimal number of clusters using the Elbow method
inertia = []
K = range(1, 6)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
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
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
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

# Display the clustered data with feedback
print(df[['student_name', 'quiz_score', 'engagement_score', 'participation_score', 'cluster', 'feedback']])
# In[28]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go

# Créer un DataFrame fictif
#data = {
 #   'student_name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
  #  'quiz_score': [80, 60, 70, 90, 85],
   # 'engagement_score': [90, 40, 50, 85, 75],
    #'participation_score': [85, 60, 55, 80, 70]
#}
df = pd.DataFrame(data)

# Calculer les statistiques
average_score = df['quiz_score'].mean()
completion_rate = df['completion_status'].value_counts(normalize=True) * 100
average_participation = df['participation_score'].mean()
average_content_quality = df['content_quality_rating'].mean()
average_engagement = df['engagement_score'].mean()
pass_rate = (df['quiz_score'] >= 60).mean() * 100  # Supposant que 60 est la note de passage

# Sélectionner les caractéristiques pour le clustering
features = df[['quiz_score', 'engagement_score', 'participation_score']]

# Standardiser les caractéristiques
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Déterminer le nombre optimal de clusters en utilisant la méthode du coude
inertia = []
K = range(1, 6)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

# Choisir le nombre optimal de clusters (par exemple, 3 basé sur la méthode du coude)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(features_scaled)

# Générer des feedbacks automatisés basés sur les clusters
def generate_feedback2(cluster):
    if cluster == 0:
        return "Excellent performance! Keep up the great work and continue to engage actively."
    elif cluster == 1:
        return "Good job! You are performing well, but consider increasing your engagement in class activities."
    elif cluster == 2:
        return "It seems you may need additional support. Please reach out for help and consider focusing on your participation."

# Appliquer la génération de feedback
df['feedback'] = df['cluster'].apply(generate_feedback2)

# Créer l'application Dash
app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1(children='Tableau de bord personnalisé des étudiants', style={'textAlign': 'center'}),
    dcc.Dropdown(df.student_name.unique(), 'Alice', id='dropdown-selection'),
    html.Div(id='metrics-content', style={'textAlign': 'center', 'marginTop': '20px'}),
    dcc.Graph(id='graph-content')
])

@app.callback(
    [Output('metrics-content', 'children'),
     Output('graph-content', 'figure')],
    [Input('dropdown-selection', 'value')]
)
def update_dashboard(value):
    dff = df[df.student_name == value]
    metrics = [
        html.H4(children='Métriques pour l\'étudiant sélectionné'),
        html.P(f'Nom : {dff["student_name"].values[0]}'),
        html.P(f'Score du quiz : {dff["quiz_score"].values[0]}'),
        html.P(f'Score d\'engagement : {dff["engagement_score"].values[0]}'),
        html.P(f'Score de participation : {dff["participation_score"].values[0]}'),
        html.P(f'Évaluation de la qualité du contenu : {dff["content_quality_rating"].values[0]}'),
        html.P(f'Taux de complétion : {completion_rate[dff["completion_status"].values[0]]:.2f}%'),
        html.P(f'Feedback : {dff["feedback"].values[0]}')
    ]
    fig = go.Figure(data=[
        go.Bar(name='Quiz Score', x=dff['student_name'], y=dff['quiz_score']),
        go.Bar(name='Engagement Score', x=dff['student_name'], y=dff['engagement_score']),
        go.Bar(name='Participation Score', x=dff['student_name'], y=dff['participation_score'])
    ])
    fig.update_layout(barmode='group', title='Scores des étudiants')
    return metrics, fig

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





# Instalar todas las librerías necesarias
# pip install numpy matplotlib pandas scikit-learn

# =============================================================================
# ANÁLISIS DE CLASIFICACIÓN - STUDENTS DATA Y HOSPITAL DATA
# Modelos utilizados: Regresión Logística y SVM (Support Vector Machine)
# =============================================================================

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from matplotlib.colors import ListedColormap

# =============================================================================
# 1. CARGA DE LOS DATOS
# =============================================================================
print("=== CARGANDO LOS DATASETS ===")
students_dataset = pd.read_csv('students_data.csv')
hospital_dataset = pd.read_csv('hospital_data.csv')

print("Students dataset cargado exitosamente")
print("Hospital dataset cargado exitosamente")

# =============================================================================
# 2. EXPLORACIÓN INICIAL
# =============================================================================
print("\n=== EXPLORACIÓN INICIAL - STUDENTS DATA ===")
print("Primeras 5 filas del dataset de estudiantes:")
print(students_dataset.head())
print("\nEstadísticas descriptivas del dataset de estudiantes:")
print(students_dataset.describe())

print("\n=== EXPLORACIÓN INICIAL - HOSPITAL DATA ===")
print("Primeras 5 filas del dataset hospitalario:")
print(hospital_dataset.head())
print("\nEstadísticas descriptivas del dataset hospitalario:")
print(hospital_dataset.describe())

# =============================================================================
# 3. ANÁLISIS DE LA VARIABLE OBJETIVO
# =============================================================================
print("\n=== ANÁLISIS DE VARIABLES OBJETIVO ===")
print("Distribución de la variable 'Result' en Students Data:")
print(students_dataset['Result'].value_counts())
print("\nDistribución de la variable 'Risk_Level' en Hospital Data:")
print(hospital_dataset['Risk_Level'].value_counts())

# =============================================================================
# 4. PREPROCESAMIENTO - STUDENTS DATA
# =============================================================================
print("\n=== PREPROCESAMIENTO - STUDENTS DATA ===")

# Crear una copia del dataset para preprocesamiento
students_processed = students_dataset.copy()

# Codificar variables categóricas para students
le_major = LabelEncoder()
le_year = LabelEncoder()
le_scholarship = LabelEncoder()
le_extracurricular = LabelEncoder()
le_result = LabelEncoder()

students_processed['Major'] = le_major.fit_transform(students_processed['Major'])
students_processed['Year'] = le_year.fit_transform(students_processed['Year'])
students_processed['Scholarship'] = le_scholarship.fit_transform(students_processed['Scholarship'])
students_processed['Extracurricular'] = le_extracurricular.fit_transform(students_processed['Extracurricular'])
students_processed['Result'] = le_result.fit_transform(students_processed['Result'])

# Para visualización usamos solo 2 características: GPA y Study_Hours
X_students_viz = students_processed[['GPA', 'Study_Hours']].values
y_students = students_processed['Result'].values

print("Variables categóricas codificadas exitosamente")
print("Características para visualización: GPA y Study_Hours")

# =============================================================================
# 4. PREPROCESAMIENTO - HOSPITAL DATA
# =============================================================================
print("\n=== PREPROCESAMIENTO - HOSPITAL DATA ===")

# Crear una copia del dataset para preprocesamiento
hospital_processed = hospital_dataset.copy()

# Codificar variables categóricas para hospital
le_gender = LabelEncoder()
le_diagnosis = LabelEncoder()
le_smoker = LabelEncoder()
le_exercise = LabelEncoder()
le_risk = LabelEncoder()

hospital_processed['Gender'] = le_gender.fit_transform(hospital_processed['Gender'])
hospital_processed['Diagnosis'] = le_diagnosis.fit_transform(hospital_processed['Diagnosis'])
hospital_processed['Smoker'] = le_smoker.fit_transform(hospital_processed['Smoker'])
hospital_processed['Exercise'] = le_exercise.fit_transform(hospital_processed['Exercise'])
hospital_processed['Risk_Level'] = le_risk.fit_transform(hospital_processed['Risk_Level'])

# Para visualización usamos solo 2 características: Age y Blood_Pressure
X_hospital_viz = hospital_processed[['Age', 'Blood_Pressure']].values
y_hospital = hospital_processed['Risk_Level'].values

print("Variables categóricas codificadas exitosamente")
print("Características para visualización: Age y Blood_Pressure")

# =============================================================================
# 5. DIVISIÓN DE LOS DATOS (80/20)
# =============================================================================
print("\n=== DIVISIÓN DE LOS DATOS ===")

# División para Students Data
X_train_students, X_test_students, y_train_students, y_test_students = train_test_split(
    X_students_viz, y_students, test_size=0.20, random_state=0)

print("Students Data - Conjunto de entrenamiento (X_train):")
print(X_train_students)
print("Students Data - Etiquetas de entrenamiento (y_train):")
print(y_train_students)
print("Students Data - Conjunto de prueba (X_test):")
print(X_test_students)
print("Students Data - Etiquetas de prueba (y_test):")
print(y_test_students)

# División para Hospital Data
X_train_hospital, X_test_hospital, y_train_hospital, y_test_hospital = train_test_split(
    X_hospital_viz, y_hospital, test_size=0.20, random_state=0)

print("\nHospital Data - Conjunto de entrenamiento (X_train):")
print(X_train_hospital)
print("Hospital Data - Etiquetas de entrenamiento (y_train):")
print(y_train_hospital)
print("Hospital Data - Conjunto de prueba (X_test):")
print(X_test_hospital)
print("Hospital Data - Etiquetas de prueba (y_test):")
print(y_test_hospital)

# =============================================================================
# ESCALADO DE CARACTERÍSTICAS
# =============================================================================
print("\n=== ESCALADO DE CARACTERÍSTICAS ===")

# Escalado para Students Data
sc_students = StandardScaler()
X_train_students = sc_students.fit_transform(X_train_students)
X_test_students = sc_students.transform(X_test_students)

print("Students Data - Conjunto de entrenamiento escalado (X_train):")
print(X_train_students)
print("Students Data - Conjunto de prueba escalado (X_test):")
print(X_test_students)

# Escalado para Hospital Data
sc_hospital = StandardScaler()
X_train_hospital = sc_hospital.fit_transform(X_train_hospital)
X_test_hospital = sc_hospital.transform(X_test_hospital)

print("\nHospital Data - Conjunto de entrenamiento escalado (X_train):")
print(X_train_hospital)
print("Hospital Data - Conjunto de prueba escalado (X_test):")
print(X_test_hospital)

# =============================================================================
# 6. ENTRENAMIENTO DE MODELOS - STUDENTS DATA
# =============================================================================
print("\n=== ENTRENAMIENTO DE MODELOS - STUDENTS DATA ===")

# Modelo 1: Regresión Logística
classifier_lr_students = LogisticRegression(random_state=0)
classifier_lr_students.fit(X_train_students, y_train_students)
print("Modelo de Regresión Logística entrenado para Students Data")

# Modelo 2: SVM
classifier_svm_students = SVC(kernel='linear', random_state=0)
classifier_svm_students.fit(X_train_students, y_train_students)
print("Modelo SVM entrenado para Students Data")

# Predicción de un nuevo resultado (Ejemplo: GPA = 3.0, Study_Hours = 25)
resultado_lr_students = classifier_lr_students.predict(sc_students.transform([[3.0, 25]]))
resultado_svm_students = classifier_svm_students.predict(sc_students.transform([[3.0, 25]]))
print(f"Predicción Regresión Logística para GPA=3.0 y Study_Hours=25: {resultado_lr_students}")
print(f"Predicción SVM para GPA=3.0 y Study_Hours=25: {resultado_svm_students}")

# =============================================================================
# 6. ENTRENAMIENTO DE MODELOS - HOSPITAL DATA
# =============================================================================
print("\n=== ENTRENAMIENTO DE MODELOS - HOSPITAL DATA ===")

# Modelo 1: Regresión Logística
classifier_lr_hospital = LogisticRegression(random_state=0)
classifier_lr_hospital.fit(X_train_hospital, y_train_hospital)
print("Modelo de Regresión Logística entrenado para Hospital Data")

# Modelo 2: SVM
classifier_svm_hospital = SVC(kernel='linear', random_state=0)
classifier_svm_hospital.fit(X_train_hospital, y_train_hospital)
print("Modelo SVM entrenado para Hospital Data")

# Predicción de un nuevo resultado (Ejemplo: Age = 45, Blood_Pressure = 120)
resultado_lr_hospital = classifier_lr_hospital.predict(sc_hospital.transform([[45, 120]]))
resultado_svm_hospital = classifier_svm_hospital.predict(sc_hospital.transform([[45, 120]]))
print(f"Predicción Regresión Logística para Age=45 y Blood_Pressure=120: {resultado_lr_hospital}")
print(f"Predicción SVM para Age=45 y Blood_Pressure=120: {resultado_svm_hospital}")

# =============================================================================
# 7. EVALUACIÓN DE MODELOS - STUDENTS DATA
# =============================================================================
print("\n=== EVALUACIÓN DE MODELOS - STUDENTS DATA ===")

# Evaluación Regresión Logística - Students
y_pred_lr_students = classifier_lr_students.predict(X_test_students)
print("\n--- REGRESIÓN LOGÍSTICA - STUDENTS ---")
print("Predicciones sobre el conjunto de prueba:")
print(np.concatenate((y_pred_lr_students.reshape(len(y_pred_lr_students), 1), 
                     y_test_students.reshape(len(y_test_students), 1)), 1))

print("Matriz de confusión:")
cm_lr_students = confusion_matrix(y_test_students, y_pred_lr_students)
print(cm_lr_students)
# La matriz de confusión muestra: 
# - Verdaderos Negativos (esquina superior izquierda): estudiantes que realmente reprobaron y fueron correctamente clasificados
# - Falsos Positivos (esquina superior derecha): estudiantes que reprobaron pero fueron clasificados como aprobados  
# - Falsos Negativos (esquina inferior izquierda): estudiantes que aprobaron pero fueron clasificados como reprobados
# - Verdaderos Positivos (esquina inferior derecha): estudiantes que realmente aprobaron y fueron correctamente clasificados

acc_lr_students = accuracy_score(y_test_students, y_pred_lr_students)
prec_lr_students = precision_score(y_test_students, y_pred_lr_students, average='macro')
rec_lr_students = recall_score(y_test_students, y_pred_lr_students, average='macro')
f1_lr_students = f1_score(y_test_students, y_pred_lr_students, average='macro')

print(f"Accuracy: {acc_lr_students:.2f}")
# Accuracy representa el porcentaje total de predicciones correctas del modelo
# Es decir, de todos los estudiantes evaluados, qué porcentaje fue clasificado correctamente

print(f"Precision: {prec_lr_students:.2f}")
# Precision indica qué tan confiables son las predicciones positivas del modelo
# De todos los estudiantes que el modelo predijo como "aprobados", qué porcentaje realmente aprobó

print(f"Recall: {rec_lr_students:.2f}")
# Recall mide qué tan bien el modelo encuentra todos los casos positivos reales
# De todos los estudiantes que realmente aprobaron, qué porcentaje fue correctamente identificado

print(f"F1-score: {f1_lr_students:.2f}")
# F1-score es el promedio armónico entre precisión y recall
# Proporciona una medida balanceada del rendimiento cuando hay clases desbalanceadas

# Evaluación SVM - Students
y_pred_svm_students = classifier_svm_students.predict(X_test_students)
print("\n--- SVM - STUDENTS ---")
print("Predicciones sobre el conjunto de prueba:")
print(np.concatenate((y_pred_svm_students.reshape(len(y_pred_svm_students), 1), 
                     y_test_students.reshape(len(y_test_students), 1)), 1))

print("Matriz de confusión:")
cm_svm_students = confusion_matrix(y_test_students, y_pred_svm_students)
print(cm_svm_students)

acc_svm_students = accuracy_score(y_test_students, y_pred_svm_students)
prec_svm_students = precision_score(y_test_students, y_pred_svm_students, average='macro')
rec_svm_students = recall_score(y_test_students, y_pred_svm_students, average='macro')
f1_svm_students = f1_score(y_test_students, y_pred_svm_students, average='macro')

print(f"Accuracy: {acc_svm_students:.2f}")
print(f"Precision: {prec_svm_students:.2f}")
print(f"Recall: {rec_svm_students:.2f}")
print(f"F1-score: {f1_svm_students:.2f}")

# =============================================================================
# 7. EVALUACIÓN DE MODELOS - HOSPITAL DATA
# =============================================================================
print("\n=== EVALUACIÓN DE MODELOS - HOSPITAL DATA ===")

# Evaluación Regresión Logística - Hospital
y_pred_lr_hospital = classifier_lr_hospital.predict(X_test_hospital)
print("\n--- REGRESIÓN LOGÍSTICA - HOSPITAL ---")
print("Predicciones sobre el conjunto de prueba:")
print(np.concatenate((y_pred_lr_hospital.reshape(len(y_pred_lr_hospital), 1), 
                     y_test_hospital.reshape(len(y_test_hospital), 1)), 1))

print("Matriz de confusión:")
cm_lr_hospital = confusion_matrix(y_test_hospital, y_pred_lr_hospital)
print(cm_lr_hospital)
# En contexto médico, la matriz de confusión muestra:
# - Verdaderos Negativos: pacientes de bajo riesgo correctamente identificados
# - Falsos Positivos: pacientes de bajo riesgo clasificados como alto riesgo (sobretratamiento)
# - Falsos Negativos: pacientes de alto riesgo clasificados como bajo riesgo (muy peligroso)
# - Verdaderos Positivos: pacientes de alto riesgo correctamente identificados

acc_lr_hospital = accuracy_score(y_test_hospital, y_pred_lr_hospital)
prec_lr_hospital = precision_score(y_test_hospital, y_pred_lr_hospital, average='macro')
rec_lr_hospital = recall_score(y_test_hospital, y_pred_lr_hospital, average='macro')
f1_lr_hospital = f1_score(y_test_hospital, y_pred_lr_hospital, average='macro')

print(f"Accuracy: {acc_lr_hospital:.2f}")
print(f"Precision: {prec_lr_hospital:.2f}")
print(f"Recall: {rec_lr_hospital:.2f}")
print(f"F1-score: {f1_lr_hospital:.2f}")

# Evaluación SVM - Hospital
y_pred_svm_hospital = classifier_svm_hospital.predict(X_test_hospital)
print("\n--- SVM - HOSPITAL ---")
print("Predicciones sobre el conjunto de prueba:")
print(np.concatenate((y_pred_svm_hospital.reshape(len(y_pred_svm_hospital), 1), 
                     y_test_hospital.reshape(len(y_test_hospital), 1)), 1))

print("Matriz de confusión:")
cm_svm_hospital = confusion_matrix(y_test_hospital, y_pred_svm_hospital)
print(cm_svm_hospital)

acc_svm_hospital = accuracy_score(y_test_hospital, y_pred_svm_hospital)
prec_svm_hospital = precision_score(y_test_hospital, y_pred_svm_hospital, average='macro')
rec_svm_hospital = recall_score(y_test_hospital, y_pred_svm_hospital, average='macro')
f1_svm_hospital = f1_score(y_test_hospital, y_pred_svm_hospital, average='macro')

print(f"Accuracy: {acc_svm_hospital:.2f}")
print(f"Precision: {prec_svm_hospital:.2f}")
print(f"Recall: {rec_svm_hospital:.2f}")
print(f"F1-score: {f1_svm_hospital:.2f}")

# =============================================================================
# 8. VISUALIZACIONES - STUDENTS DATA (REGRESIÓN LOGÍSTICA)
# =============================================================================

# Visualización de los resultados en el conjunto de entrenamiento - Students LR
X_set, y_set = sc_students.inverse_transform(X_train_students), y_train_students
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.5, stop = X_set[:, 0].max() + 0.5, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 2, stop = X_set[:, 1].max() + 2, step = 0.25))
plt.contourf(X1, X2, classifier_lr_students.predict(sc_students.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Regresión Logística - Students (Conjunto de Entrenamiento)')
plt.xlabel('GPA')
plt.ylabel('Study Hours')
plt.legend()
plt.show()

# Visualización de los resultados en el conjunto de prueba - Students LR
X_set, y_set = sc_students.inverse_transform(X_test_students), y_test_students
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.5, stop = X_set[:, 0].max() + 0.5, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 2, stop = X_set[:, 1].max() + 2, step = 0.25))
plt.contourf(X1, X2, classifier_lr_students.predict(sc_students.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Regresión Logística - Students (Conjunto de Prueba)')
plt.xlabel('GPA')
plt.ylabel('Study Hours')
plt.legend()
plt.show()

# =============================================================================
# VISUALIZACIONES - STUDENTS DATA (SVM)
# =============================================================================

# Visualización de los resultados en el conjunto de entrenamiento - Students SVM
X_set, y_set = sc_students.inverse_transform(X_train_students), y_train_students
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.5, stop = X_set[:, 0].max() + 0.5, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 2, stop = X_set[:, 1].max() + 2, step = 0.25))
plt.contourf(X1, X2, classifier_svm_students.predict(sc_students.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM - Students (Conjunto de Entrenamiento)')
plt.xlabel('GPA')
plt.ylabel('Study Hours')
plt.legend()
plt.show()

# Visualización de los resultados en el conjunto de prueba - Students SVM
X_set, y_set = sc_students.inverse_transform(X_test_students), y_test_students
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.5, stop = X_set[:, 0].max() + 0.5, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 2, stop = X_set[:, 1].max() + 2, step = 0.25))
plt.contourf(X1, X2, classifier_svm_students.predict(sc_students.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM - Students (Conjunto de Prueba)')
plt.xlabel('GPA')
plt.ylabel('Study Hours')
plt.legend()
plt.show()

# =============================================================================
# VISUALIZACIONES - HOSPITAL DATA (REGRESIÓN LOGÍSTICA)
# =============================================================================

# Visualización de los resultados en el conjunto de entrenamiento - Hospital LR
X_set, y_set = sc_hospital.inverse_transform(X_train_hospital), y_train_hospital
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 0.5),
                     np.arange(start = X_set[:, 1].min() - 10, stop = X_set[:, 1].max() + 10, step = 1))
plt.contourf(X1, X2, classifier_lr_hospital.predict(sc_hospital.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Regresión Logística - Hospital (Conjunto de Entrenamiento)')
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.legend()
plt.show()

# Visualización de los resultados en el conjunto de prueba - Hospital LR
X_set, y_set = sc_hospital.inverse_transform(X_test_hospital), y_test_hospital
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 0.5),
                     np.arange(start = X_set[:, 1].min() - 10, stop = X_set[:, 1].max() + 10, step = 1))
plt.contourf(X1, X2, classifier_lr_hospital.predict(sc_hospital.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Regresión Logística - Hospital (Conjunto de Prueba)')
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.legend()
plt.show()

# =============================================================================
# VISUALIZACIONES - HOSPITAL DATA (SVM)
# =============================================================================

# Visualización de los resultados en el conjunto de entrenamiento - Hospital SVM
X_set, y_set = sc_hospital.inverse_transform(X_train_hospital), y_train_hospital
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 0.5),
                     np.arange(start = X_set[:, 1].min() - 10, stop = X_set[:, 1].max() + 10, step = 1))
plt.contourf(X1, X2, classifier_svm_hospital.predict(sc_hospital.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM - Hospital (Conjunto de Entrenamiento)')
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.legend()
plt.show()

# Visualización de los resultados en el conjunto de prueba - Hospital SVM
X_set, y_set = sc_hospital.inverse_transform(X_test_hospital), y_test_hospital
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 5, stop = X_set[:, 0].max() + 5, step = 0.5),
                     np.arange(start = X_set[:, 1].min() - 10, stop = X_set[:, 1].max() + 10, step = 1))
plt.contourf(X1, X2, classifier_svm_hospital.predict(sc_hospital.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM - Hospital (Conjunto de Prueba)')
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.legend()
plt.show()

# =============================================================================
# 8. COMPARACIÓN Y ANÁLISIS FINAL
# =============================================================================

print("\n" + "="*80)
print("ANÁLISIS COMPARATIVO Y RECOMENDACIONES FINALES")
print("="*80)

print("\n--- COMPARACIÓN PARA STUDENTS DATA ---")
print(f"Regresión Logística - Accuracy: {acc_lr_students:.2f}, Precision: {prec_lr_students:.2f}, Recall: {rec_lr_students:.2f}, F1: {f1_lr_students:.2f}")
print(f"SVM                 - Accuracy: {acc_svm_students:.2f}, Precision: {prec_svm_students:.2f}, Recall: {rec_svm_students:.2f}, F1: {f1_svm_students:.2f}")

# ANÁLISIS DETALLADO PARA STUDENTS DATA:
# En el contexto educativo, es importante equilibrar la capacidad de identificar estudiantes
# en riesgo de reprobar (alta sensibilidad/recall) con la precisión para evitar 
# intervenciones innecesarias (alta precisión).
# 
# - Si la Regresión Logística tiene mayor recall, es mejor para identificar todos los 
#   estudiantes que podrían reprobar, permitiendo intervenciones tempranas.
# - Si SVM tiene mayor precisión, es mejor para minimizar falsas alarmas.
# - El F1-score nos da el mejor balance entre ambos criterios.
# 
# RECOMENDACIÓN STUDENTS: Usar el modelo con mayor F1-score, ya que proporciona 
# el mejor equilibrio entre identificar estudiantes en riesgo y minimizar falsas alarmas.

print("\n--- COMPARACIÓN PARA HOSPITAL DATA ---")
print(f"Regresión Logística - Accuracy: {acc_lr_hospital:.2f}, Precision: {prec_lr_hospital:.2f}, Recall: {rec_lr_hospital:.2f}, F1: {f1_lr_hospital:.2f}")
print(f"SVM                 - Accuracy: {acc_svm_hospital:.2f}, Precision: {prec_svm_hospital:.2f}, Recall: {rec_svm_hospital:.2f}, F1: {f1_svm_hospital:.2f}")

# ANÁLISIS DETALLADO PARA HOSPITAL DATA:
# En el contexto médico, los falsos negativos (pacientes de alto riesgo clasificados 
# como bajo riesgo) son especialmente peligrosos, ya que podrían no recibir el 
# tratamiento necesario.
# 
# - El RECALL es crítico: necesitamos identificar la mayor cantidad posible de 
#   pacientes de alto riesgo real.
# - La PRECISIÓN también es importante para evitar tratamientos innecesarios, 
#   pero secundaria al recall en contextos de salud.
# 
# RECOMENDACIÓN HOSPITAL: Priorizar el modelo con mayor recall para maximizar 
# la detección de pacientes de alto riesgo, incluso si esto resulta en algunos 
# falsos positivos. En medicina, es preferible pecar de cauteloso.

print("\n--- INTERPRETACIÓN DE MÉTRICAS ---")
print("• ACCURACY: Porcentaje total de predicciones correctas")
print("• PRECISION: De las predicciones positivas, cuántas fueron correctas")
print("• RECALL: De los casos reales positivos, cuántos fueron detectados") 
print("• F1-SCORE: Balance harmónico entre precisión y recall")

print("\n--- RECOMENDACIONES FINALES ---")
if f1_lr_students > f1_svm_students:
    print("STUDENTS DATA: Se recomienda REGRESIÓN LOGÍSTICA")
    print("- Mejor balance entre identificar estudiantes en riesgo y minimizar falsas alarmas")
    print("- Modelo más interpretable para entender factores que afectan el rendimiento")
else:
    print("STUDENTS DATA: Se recomienda SVM")
    print("- Mejor capacidad de separación entre estudiantes que aprueban y reprueban")
    print("- Mayor robustez en la clasificación")

if rec_lr_hospital > rec_svm_hospital:
    print("\nHOSPITAL DATA: Se recomienda REGRESIÓN LOGÍSTICA")
    print("- Mayor capacidad para detectar pacientes de alto riesgo")
    print("- Crítico en contexto médico para no perder casos peligrosos")
else:
    print("\nHOSPITAL DATA: Se recomienda SVM")
    print("- Mayor capacidad para detectar pacientes de alto riesgo")
    print("- Mejor rendimiento en la identificación de casos críticos")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)

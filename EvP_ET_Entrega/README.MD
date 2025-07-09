# 🎮 CS 1.6 ML Predictor

App desarrollada en Streamlit para predecir el rendimiento de un jugador de Counter-Strike 1.6 usando Machine Learning.

## 🚀 Funcionalidades

- Predicción de kills por ronda (modelo de regresión).
- Predicción de probabilidad de ganar una ronda (modelo XGBoost).
- Interfaz simple con selección de armas, granadas, armadura y mapa.
- Visualización de resultados en tiempo real.

## 🧠 Modelos usados

- `modelo_regresion_80.pkl`: Regresión lineal.
- `modelo_xgboost.pkl`: Clasificador XGBoost.

## 📦 Instalación

1. **Clona el repositorio**

En Terminal bash
git clone https://github.com/glYohanny/FUNDAMENTOS_DE_MACHINE_LEARNING_003V.git

## 📂 Estructura

FUNDAMENTOS_DE_MACHINE_LEARNING_003V/
└── EvP_ET_Entrega/
└── app_visual/
├── app.py
├── venv/
├── img/
├── preprocesamiento.py
├── modelo_regresion_80.pkl
└── modelo_xgboost.pkl


## 🔧 Instrucciones de uso

1. Abre una terminal en `app_visual/`
2. Crea y activa un entorno virtual:

En el terminal bash:

python -m venv venv
venv\Scripts\activate  # En Windows

Instala las dependencias:
pip install -r requirements.txt

Ejecuta la aplicación:
streamlit run app.py


import streamlit as st
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="CS 1.6 ML Predictor",
    page_icon="🎮",
    layout="centered"
)

# Función para cargar modelos de forma segura
@st.cache_resource
def cargar_modelos():
    try:
        modelo_regresion = joblib.load("../modelo_regresion_80 (1).pkl")
        st.success("✅ Modelo de regresión cargado correctamente")
    except Exception as e:
        st.error(f"❌ Error cargando modelo de regresión: {e}")
        modelo_regresion = None
    
    try:
        # Intentar cargar con joblib primero
        modelo_clasificacion = joblib.load("../modelo_xgboost.pkl")
        st.success("✅ Modelo XGBoost cargado correctamente")
    except Exception as e:
        st.error(f"❌ Error cargando modelo XGBoost con joblib: {e}")
        
        # Intentar cargar directamente con pickle
        try:
            import pickle
            with open("../modelo_xgboost.pkl", 'rb') as f:
                modelo_clasificacion = pickle.load(f)
            st.success("✅ Modelo XGBoost cargado con pickle")
        except Exception as e2:
            st.error(f"❌ Error cargando modelo XGBoost con pickle: {e2}")
            
            # Crear modelo alternativo simple
            st.warning("⚠️ Usando modelo alternativo simple")
            modelo_clasificacion = crear_modelo_alternativo()
    
    return modelo_regresion, modelo_clasificacion

# Modelo alternativo simple para clasificación
def crear_modelo_alternativo():
    """Crea un modelo simple basado en reglas para clasificación"""
    class ModeloAlternativo:
        def predict(self, X):
            # Reglas simples basadas en las features
            # [arma, dinero, armadura, flashbang, ranking, mapa]
            predictions = []
            for row in X:
                score = 0
                # Arma (0-6): AWP es la mejor
                if row[0] == 2:  # AWP
                    score += 0.3
                elif row[0] in [0, 1]:  # AK-47, M4A1
                    score += 0.2
                
                # Dinero
                if row[1] > 10000:
                    score += 0.2
                elif row[1] > 5000:
                    score += 0.1
                
                # Armadura
                if row[2] == 1:
                    score += 0.1
                
                # Flashbang
                if row[3] == 1:
                    score += 0.1
                
                # Ranking (0-4): Global es mejor
                score += row[4] * 0.1
                
                # Mapa (algunos mapas son más fáciles)
                if row[5] == 0:  # dust2
                    score += 0.05
                
                predictions.append(1 if score > 0.5 else 0)
            
            return np.array(predictions)
        
        def predict_proba(self, X):
            # Probabilidades simplificadas
            probas = []
            for row in X:
                score = 0
                if row[0] == 2:  # AWP
                    score += 0.3
                elif row[0] in [0, 1]:  # AK-47, M4A1
                    score += 0.2
                
                if row[1] > 10000:
                    score += 0.2
                elif row[1] > 5000:
                    score += 0.1
                
                if row[2] == 1:
                    score += 0.1
                if row[3] == 1:
                    score += 0.1
                
                score += row[4] * 0.1
                if row[5] == 0:
                    score += 0.05
                
                # Normalizar entre 0.1 y 0.9
                prob = max(0.1, min(0.9, score))
                probas.append([1-prob, prob])
            
            return np.array(probas)
    
    return ModeloAlternativo()

# Cargar modelos
modelo_regresion, modelo_clasificacion = cargar_modelos()

st.title("🎮 Counter-Strike 1.6 — Predicción con Machine Learning")
st.markdown("Usa tus datos de equipamiento y mapa para predecir kills por ronda o si ganarás la ronda.")

# Tabs para navegación
tab1, tab2 = st.tabs(["📈 Kills por partida", "🏆 ¿Ganar ronda?"])

# Función de preprocesamiento (simplificada)
def transformar_inputs(arma, dinero, armadura, granadas, ranking, mapa):
    armas = ["AK-47", "M4A1", "AWP", "Famas", "Galil", "MP5", "P90"]
    rankings = ["Silver", "Gold", "Nova", "Elite", "Global"]
    mapas = ["de_dust2", "de_inferno", "de_nuke", "de_train", "de_mirage"]
    
    vector = [
        armas.index(arma),          # 1: arma
        dinero,                     # 2: dinero
        1 if armadura else 0,       # 3: armadura
        1 if "Flashbang" in granadas else 0,  # 4: flashbang
        rankings.index(ranking),    # 5: ranking
        mapas.index(mapa)           # 6: mapa
    ]
    
    return np.array([vector])

# --- TAB 1: Regresión (Kills por ronda) ---
with tab1:
    st.header("🔫 Predicción de Kills por Ronda")
    
    if modelo_regresion is None:
        st.warning("⚠️ Modelo de regresión no disponible. Verifica la ruta del archivo.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            arma = st.selectbox("Arma principal", ["AK-47", "M4A1", "AWP", "Famas", "Galil", "MP5", "P90"])
            dinero = st.slider("Dinero disponible ($)", 0, 16000, 8000)
            ranking = st.selectbox("Ranking del jugador", ["Silver", "Gold", "Nova", "Elite", "Global"])
        
        with col2:
            armadura = st.checkbox("🛡️ Tienes armadura", value=True)
            granadas = st.multiselect("Granadas", ["Flashbang", "HE Grenade", "Smoke"])
            mapa = st.selectbox("Mapa", ["de_dust2", "de_inferno", "de_nuke", "de_train", "de_mirage"])
        
        if st.button("🎯 Predecir kills"):
            try:
                datos = transformar_inputs(arma, dinero, armadura, granadas, ranking, mapa)
                pred = modelo_regresion.predict(datos)
                st.metric(label="Kills estimadas por ronda", value=f"{pred[0]:.2f}", delta="ML model")
            except Exception as e:
                st.error(f"Error en predicción: {e}")

# --- TAB 2: Clasificación (Ganar ronda) ---
with tab2:
    st.header("🏆 Predicción: ¿Se gana la ronda?")
    
    if modelo_clasificacion is None:
        st.warning("⚠️ Modelo XGBoost no disponible. Verifica la instalación y ruta del archivo.")
        st.info("💡 **Soluciones posibles:**")
        st.code("pip install xgboost", language="bash")
        st.markdown("O si usas conda:")
        st.code("conda install xgboost", language="bash")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            arma = st.selectbox("Arma principal", ["AK-47", "M4A1", "AWP", "Famas", "Galil", "MP5", "P90"], key="arma2")
            dinero = st.slider("Dinero disponible ($)", 0, 16000, 8000, key="dinero2")
            ranking = st.selectbox("Ranking del jugador", ["Silver", "Gold", "Nova", "Elite", "Global"], key="ranking2")
        
        with col2:
            armadura = st.checkbox("🛡️ Tienes armadura", value=True, key="armadura2")
            granadas = st.multiselect("Granadas", ["Flashbang", "HE Grenade", "Smoke"], key="granadas2")
            mapa = st.selectbox("Mapa", ["de_dust2", "de_inferno", "de_nuke", "de_train", "de_mirage"], key="mapa2")
        
        if st.button("🔍 Predecir resultado de la ronda"):
            try:
                datos = transformar_inputs(arma, dinero, armadura, granadas, ranking, mapa)
                pred = modelo_clasificacion.predict(datos)
                prob = modelo_clasificacion.predict_proba(datos)[0][1]
                st.metric("Probabilidad de ganar la ronda", f"{prob*100:.1f}%", delta="Modelo ML")
                
                if pred[0] == 1:
                    st.success("✅ ¡Alta probabilidad de victoria!")
                else:
                    st.error("❌ Baja probabilidad de ganar")
            except Exception as e:
                st.error(f"Error en predicción: {e}")

# Información adicional
st.markdown("---")
st.info("💡 **Nota:** Si tienes problemas con XGBoost, asegúrate de tener instaladas las dependencias correctas.")

# Agregar información de debug
if st.checkbox("🔧 Mostrar información de debug"):
    st.subheader("Debug Info")
    try:
        import xgboost as xgb
        st.success(f"✅ XGBoost versión: {xgb.__version__}")
    except ImportError:
        st.error("❌ XGBoost no está instalado")
    
    st.write("**Librerías disponibles:**")
    try:
        import sklearn
        st.write(f"- Scikit-learn: {sklearn.__version__}")
    except ImportError:
        st.write("- Scikit-learn: No disponible")
    
    try:
        import pandas as pd
        st.write(f"- Pandas: {pd.__version__}")
    except ImportError:
        st.write("- Pandas: No disponible")
    
    st.write(f"- NumPy: {np.__version__}")
    st.write(f"- Streamlit: {st.__version__}")
    
    # Mostrar información sobre el problema de compatibilidad
    st.subheader("🔧 Solución para el error de compatibilidad")
    st.markdown("""
    **El error `_RemainderColsList` indica incompatibilidad de versiones de scikit-learn.**
    
    **Soluciones:**
    
    1. **Actualizar scikit-learn:**
    ```bash
    pip install --upgrade scikit-learn
    ```
    
    2. **Usar versiones compatibles:**
    ```bash
    pip install scikit-learn==1.3.0 xgboost==1.7.3
    ```
    
    3. **Recrear el modelo:**
    - Entrena nuevamente el modelo con tu versión actual de scikit-learn
    - O usa el modelo alternativo que he implementado
    
    4. **Verificar el archivo:**
    - Asegúrate de que el archivo .pkl no esté corrupto
    - Verifica que se guardó correctamente
    """)
    
    # Mostrar el contenido del directorio
    import os
    st.write("**Archivos en el directorio padre:**")
    try:
        archivos = os.listdir("../")
        for archivo in archivos:
            if archivo.endswith('.pkl'):
                st.write(f"📄 {archivo}")
    except Exception as e:
        st.write(f"No se pudo listar el directorio: {e}")
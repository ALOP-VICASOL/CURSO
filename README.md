# 🤖 Agente RAG con Groq y Streamlit

Este proyecto es un agente de Inteligencia Artificial basado en la arquitectura **RAG (Retrieval-Augmented Generation)**. Permite subir documentos a un directorio local y realizar preguntas sobre ellos utilizando los modelos de lenguaje ultra rápidos de **Groq**.

## 🚀 Características

- **Inferencia Ultra-rápida:** Gracias a la LPU de Groq.
- **Embeddings Locales:** Uso de `sentence-transformers` (HuggingFace) para procesar vectores sin costo adicional.
- **Interfaz Intuitiva:** Construido totalmente con Streamlit.
- **Persistencia en Memoria:** Indexación dinámica de documentos en el directorio seleccionado.

---

## 🛠️ Requisitos Previos

Antes de empezar, asegúrate de tener instalado:

1.  **Python 3.10+**
2.  **Groq API Key:** Obtén una de forma gratuita en Groq Console.
3.  **Directorio de Datos:** Una carpeta con archivos PDF o TXT que desees consultar.

---

## 📦 Instalación y Configuración

Sigue estos pasos para poner en marcha el agente:

### 1. Clonar el repositorio o crear la carpeta del proyecto
```bash
mkdir rag-groq-app
cd rag-groq-app
```

### 2. Crear un entorno virtual
```bash
python -m venv venv
# Activar en Windows:
.\venv\Scripts\activate
# Activar en Linux/Mac:
source venv/bin/activate
```

### 3. Instalar las dependencias
Crea un archivo `requirements.txt` con el siguiente contenido y ejecuta la instalación:
```text
streamlit
langchain
langchain-groq
langchain-community
langchain-huggingface
faiss-cpu
pypdf
python-dotenv
sentence-transformers
```

Instalación:
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
Crea un archivo llamado `.env` en la raíz del proyecto:
```env
GROQ_API_KEY=tu_api_key_aqui
```

---

## 🖥️ Ejecución de la Aplicación

Para iniciar el agente, simplemente ejecuta el comando de Streamlit:

```bash
streamlit run app.py
```

---

## 📖 Cómo usar el Agente

1.  **Carga de datos:** Coloca tus documentos (.pdf o .txt) en la carpeta `./data` (o la que especifiques en la interfaz).
2.  **Indexación:** En la barra lateral, haz clic en el botón **"Procesar Archivos"**. Esto convertirá tus documentos en vectores numéricos legibles para la IA.
3.  **Consulta:** Escribe tu pregunta en el chat. El agente buscará en tus archivos locales y te responderá utilizando el contexto encontrado a través de Groq.

---

## ⚙️ Tecnologías Utilizadas

- **Groq Cloud:** Modelos Llama 3 o Mixtral.
- **LangChain:** Framework de orquestación para LLMs.
- **FAISS:** Base de datos vectorial local.
- **HuggingFace:** Generación de embeddings.

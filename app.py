import streamlit as st
import os
import ssl
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import snapshot_download
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Configuración para omitir la verificación de certificados SSL
# Esto resuelve el error [SSL: CERTIFICATE_VERIFY_FAILED]
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

# Cargar variables de entorno (API Key de Groq)
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not groq_api_key or not pinecone_api_key:
    st.warning("⚠️ Faltan API Keys en el archivo .env (GROQ_API_KEY o PINECONE_API_KEY).")
    st.stop()

st.set_page_config(page_title="Groq RAG Agent", layout="wide", page_icon="🤖")
st.title("🤖 Agente RAG con Groq")
st.markdown("---")

# Configuración en la barra lateral
with st.sidebar:
    st.header("⚙️ Configuración")
    model = st.selectbox("Modelo de Groq", ["llama3-8b-8192", "mixtral-8x7b-32768"])
    data_path = "./data" # Ruta fija a la carpeta de datos
    process_btn = st.button("🚀 Procesar Archivos")
    
    if st.button("🗑️ Limpiar Historial"):
        st.session_state.messages = []
        st.rerun()
        
    st.info("Soporta archivos .pdf y .txt en el directorio especificado.")

# Cargar el modelo de embeddings de forma aislada. 
# Esto ayuda a gestionar mejor los hilos y conexiones en Python 3.13.
@st.cache_resource
def get_embeddings_model(): # type: ignore
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        # Descarga/verifica explícitamente los archivos del modelo en la caché local.
        # Esto ayuda a prevenir problemas de red durante la inicialización de SentenceTransformer.
        st.info(f"Verificando/descargando modelo de embeddings '{model_name}'...")
        snapshot_download(
            repo_id=model_name,
            allow_patterns=[
                "config.json", "sentence_bert_config.json", "modules.json",
                "tokenizer.json", "tokenizer_config.json", "vocab.txt",
                "special_tokens_map.json", "pytorch_model.bin", "model.safetensors"
            ]
        )
        st.success(f"Modelo de embeddings '{model_name}' disponible localmente.")
    except Exception as e:
        st.error(f"Error al asegurar el modelo de embeddings '{model_name}': {e}")
        st.warning("Por favor, asegúrate de tener una conexión a internet estable o considera descargar el modelo manualmente.")
        st.stop() # Detener la aplicación si la descarga del modelo falla

    # Inicializar HuggingFaceEmbeddings, que ahora debería cargar desde la caché local.
    return HuggingFaceEmbeddings(model_name=model_name)

# Inicializar Embeddings y Vector Store (Caché para evitar recargas innecesarias)
@st.cache_resource
def get_vector_store(directory): # type: ignore
    if not os.path.exists(directory):
        st.error(f"El directorio de datos '{directory}' no existe.")
        return None
    
    index_name = "quickstart"
    embeddings = get_embeddings_model()

    st.info(f"Procesando documentos y sincronizando con Pinecone (índice: {index_name})...")
    # Cargadores para PDF y Texto
    pdf_loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
    
    documents = []
    try:
        documents.extend(pdf_loader.load())
        documents.extend(txt_loader.load())
    except Exception as e:
        st.error(f"Error al cargar documentos: {e}")
        return None

    if not documents:
        st.warning(f"No se encontraron documentos .pdf o .txt en '{directory}'.")
        return None

    # División de texto en fragmentos para mejorar la búsqueda
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # Sincronización con Pinecone
    try:
        vector_store = PineconeVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            index_name=index_name,
            pinecone_api_key=pinecone_api_key
        )
        st.success(f"¡Sincronización con Pinecone completada con éxito!")
        return vector_store
    except Exception as e:
        st.error(f"Error al conectar o subir a Pinecone: {e}")
        return None


# Lógica de procesamiento de documentos
if process_btn:
    with st.spinner("Analizando y vectorizando documentos..."):
        vs = get_vector_store(data_path)
        if vs:
            st.session_state.vector_store = vs
            # El mensaje de éxito ya se muestra dentro de get_vector_store
        else:
            st.error(f"No se pudo inicializar el vector store para {data_path}")

# Interfaz de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "vector_store" in st.session_state:
    if prompt := st.chat_input("Hazme una pregunta sobre tus archivos:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            llm = ChatGroq(groq_api_key=groq_api_key, model_name=model)
            
            # Definición del prompt para el asistente
            system_prompt = (
                "Eres un asistente para tareas de respuesta a preguntas. "
                "Usa los siguientes fragmentos de contexto recuperado para responder a la pregunta. "
                "Si no sabes la respuesta, di que no lo sabes. "
                "Usa tres frases como máximo y mantén la respuesta concisa."
                "\n\n"
                "{context}"
            )
            
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )

            # Función auxiliar para formatear documentos recuperados
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # Configuración del recuperador
            retriever = st.session_state.vector_store.as_retriever()

            # Construcción de la cadena usando LCEL (evita por completo langchain.chains)
            # Esta cadena recupera contexto, mantiene la pregunta original y genera la respuesta
            rag_chain = (
                {"context": retriever | format_docs, "input": RunnablePassthrough(), "docs": retriever}
                | RunnablePassthrough.assign(answer=prompt_template | llm | StrOutputParser())
            )

            # Ejecutar la cadena
            response = rag_chain.invoke(prompt)
            
            answer = response["answer"]
            st.markdown(answer)
            
            # Mostramos las fuentes usando la clave 'docs' que definimos en la cadena
            if response.get("docs"):
                with st.expander("📚 Fuentes consultadas"):
                    for doc in response["docs"]:
                        st.write(f"- {os.path.basename(doc.metadata['source'])}")
            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Haz clic en 'Procesar Archivos' para cargar los documentos de la carpeta /data y comenzar.")
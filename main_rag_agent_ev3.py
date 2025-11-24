# =================================================================
# Agente Funcional Médico - Hospital Barros Luco (v2 - Observabilidad)
# Archivo: main_rag_agent_v2.py
# =================================================================
import streamlit as st
import os
import json
import time
import uuid
import re  # IL3.3: Para validación de inputs
from datetime import datetime
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import requests
import concurrent.futures

# IL3.2: Configuración de Logs Estructurados
import logging
import structlog

# Configuración básica de logging
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
# Configuración para estructurar logs en JSON
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    # cache_logger_factory=False,  # <-- REMOVED: causa TypeError en algunas versiones de structlog
)
logger = structlog.get_logger()

# Cargar variables de entorno (mantener compatibilidad)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Variables de Entorno (adaptadas para OpenAI/Azure o GitHub inference)
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
github_token = os.getenv("GITHUB_TOKEN")
# URL que usas en .env para embeddings / inferencia GitHub
github_inference_url = os.getenv("OPENAI_EMBEDDINGS_URL") or os.getenv("GITHUB_BASE_URL")

st.set_page_config(page_title="🏥 Agente Médico Funcional v2 (Obs)", page_icon="🏥", layout="wide")

# =================================================================
# Clase refactorizada con Trazabilidad y Seguridad (IL3.1, IL3.2, IL3.3)
# =================================================================
class ChatbotMedicoRAG:
    """Clase principal del Agente Funcional Médico con RAG."""
    def __init__(self):
        self.client = None
        self.llm_model = "gpt-4o-mini"
        self.embeddings_model = "text-embedding-3-small"
        self.documents = []
        self.embeddings = None
        self.embedding_matrix = None
        self.interaction_logs = []
        self.error_count = 0 # IL3.1: Métrica de frecuencia de errores

    def initialize_client(self):
        """Inicializa el cliente OpenAI o habilita modo GitHub inference."""
        try:
            if openai_api_key:
                # Modo OpenAI / Azure (usa SDK)
                self.client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
                self.github_mode = False
                logger.info("system_init", status="success", message="OpenAI/Azure client inicializado")
                return True
            elif github_token and github_inference_url:
                # Modo GitHub inference: usaremos HTTP requests directos
                self.client = None
                self.github_mode = True
                self.github_token = github_token
                self.github_inference_url = github_inference_url.rstrip("/")
                logger.info("system_init", status="success", message="GitHub inference mode habilitado")
                return True
            else:
                logger.error("system_init", status="error", message="No OPENAI_API_KEY ni GITHUB_TOKEN válidos")
                st.error("Falta OPENAI_API_KEY o GITHUB_TOKEN en .env")
                return False
        except Exception as e:
            logger.error("system_init", status="error", message=f"Error al inicializar cliente: {e}")
            st.error(f"Error initializing client: {e}")
            return False

    # ... (initialize_hospital_documents - sin cambios mayores)
    def initialize_hospital_documents(self) -> bool:
        """Carga documentos por defecto."""
        # Mantener tu código de inicialización de documentos
        try:
            docs = [
                "El Hospital Barros Luco ubicado en Santiago de Chile cuenta con los siguientes servicios: Emergencias 24/7, Cuidados Intensivos (UCI), Cardiología, Neurología, Pediatría, Ginecología y Obstetricia, Oncología, Ortopedia y Traumatología, Radiología e Imágenes, Laboratorio Clínico, Farmacia, y Rehabilitación Física.",
                "Horarios de atención: Emergencias 24 horas. Consultas externas: Lunes a Viernes 7:00 AM - 6:00 PM, Sábados 8:00 AM - 2:00 PM. Teléfono principal: (01) 234-5678. Emergencias: 911. Dirección: Av. Salud 123, Lima, Perú.",
                "Protocolo de emergencias: En caso de emergencia médica, dirigirse inmediatamente al área de Emergencias en el primer piso. El personal de triaje evaluará la urgencia. Código Azul: Paro cardiorrespiratorio. Código Rojo: Emergencia médica. Código Amarillo: Emergencia quirúrgica.",
                "Proceso de hospitalización: 1) Admisión con documento de identidad y seguro médico. 2) Evaluación médica inicial. 3) Asignación de habitación según disponibilidad. 4) Entrega de brazalete de identificación. 5) Orientación sobre normas hospitalarias. Horarios de visita: 2:00 PM - 4:00 PM y 6:00 PM - 8:00 PM.",
                "El uso de IA en radiología ayuda a detectar anomalías en imágenes médicas con mayor precisión. Nuestro Hospital Barros Luco utiliza sistemas de inteligencia artificial para análisis de radiografías, tomografías y resonancias magnéticas, lo que permite diagnósticos más rápidos y exactos.",
                "La telemedicina permite realizar consultas médicas a distancia, mejorando el acceso en zonas rurales. El Hospital Barros Luco ofrece servicios de teleconsulta para seguimiento de pacientes crónicos, consultas de especialidades y orientación médica inicial.",
            ]
            self.documents = docs
            self.embeddings = None
            self.embedding_matrix = None
            logger.info("data_load", status="success", doc_count=len(docs), message="Documentos del hospital cargados")
            return True
        except Exception as e:
            logger.error("data_load", status="error", message=f"Error inicializando documentos: {e}")
            st.warning(f"⚠ Error inicializando documentos: {e}")
            return False
        
    def _github_post(self, path, payload):
        """POST genérico a GitHub inference endpoint y devuelve JSON o lanza excepción."""
        url = f"{self.github_inference_url.rstrip('/')}/{path.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_embeddings(self, documents):
        """Genera embeddings usando OpenAI SDK o GitHub inference (según modo)."""
        try:
            if getattr(self, "github_mode", False):
                start_time = time.time()
                payload = {"model": self.embeddings_model, "input": documents}
                resp = self._github_post("embeddings", payload)
                # Adaptación según formato: intentar rutas comunes
                embs = [item.get("embedding") or item.get("vector") for item in resp.get("data", [])]
            else:
                if not self.client:
                    return None
                start_time = time.time()
                resp = self.client.embeddings.create(model=self.embeddings_model, input=documents)
                embs = [d.embedding for d in resp.data]
            self.embeddings = embs
            self.embedding_matrix = np.array(embs, dtype=float)
            duration = time.time() - start_time
            logger.info("tool_call", tool="embedding_generation", status="success", duration_sec=duration, doc_count=len(documents))
            return embs
        except Exception as e:
            self.error_count += 1
            logger.error("tool_call", tool="embedding_generation", status="error", error=str(e), doc_count=len(documents))
            st.warning(f"⚠ Error generando embeddings: {e}")
            return None

    def get_query_embedding(self, query):
        try:
            if getattr(self, "github_mode", False):
                payload = {"model": self.embeddings_model, "input": [query]}
                resp = self._github_post("embeddings", payload)
                emb = (resp.get("data") or [{}])[0].get("embedding") or (resp.get("data") or [{}])[0].get("vector")
                return np.array(emb, dtype=float)
            else:
                if not self.client:
                    return None
                resp = self.client.embeddings.create(model=self.embeddings_model, input=[query])
                emb = resp.data[0].embedding
                return np.array(emb, dtype=float)
        except Exception as e:
            self.error_count += 1
            logger.error("tool_call", tool="query_embedding", status="error", error=str(e))
            st.warning(f"⚠ Error obteniendo embedding de la query: {e}")
            return None

    def hybrid_search_with_metrics(self, query, top_k=3):
        """Búsqueda híbrida y registro de tiempo."""
        start = time.time()
        try:
            if not self.documents or self.embedding_matrix is None:
                logger.warning("rag_search", status="skipped", reason="No documents/embeddings available")
                return [], 0.0
            
            # ... (Lógica de búsqueda híbrida)
            q_emb = self.get_query_embedding(query)
            if q_emb is None:
                return [], 0.0
            sims = cosine_similarity(self.embedding_matrix, q_emb.reshape(1, -1)).reshape(-1)
            q_words = set([w.strip(".,?¡!():;\"'").lower() for w in query.split() if len(w) > 2])
            results = []
            for idx, doc in enumerate(self.documents):
                doc_words = set([w.strip(".,?¡!():;\"'").lower() for w in doc.split() if len(w) > 2])
                lexical = 0.0
                if q_words:
                    lexical = len(q_words.intersection(doc_words)) / max(1, len(q_words))
                semantic = float(sims[idx])
                combined = 0.7 * semantic + 0.3 * lexical
                results.append({
                    'id': idx,
                    'document': doc,
                    'semantic_score': semantic,
                    'lexical_score': lexical,
                    'combined_score': combined
                })
            results = sorted(results, key=lambda x: x['combined_score'], reverse=True)[:top_k]
            retrieval_time = time.time() - start
            logger.info("rag_search", status="success", duration_sec=retrieval_time, top_k=top_k) # IL3.2: Trazabilidad
            return results, retrieval_time
        except Exception as e:
            self.error_count += 1
            logger.error("rag_search", status="error", error=str(e))
            st.warning(f"⚠ Error en búsqueda híbrida: {e}")
            return [], 0.0

    def generate_response_with_metrics(self, query, context):
        """
        Genera una respuesta con el LLM usando contexto.
        Retorna (response_text, generation_time, used_context_text, tokens_used)
        """
        start = time.time()
        tokens_used = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        try:
            prompt = f"Contexto:\n{context}\n\nPregunta: {query}\n\nResponda de forma clara y concisa:"
            if getattr(self, "github_mode", False):
                # Llamada simple a endpoint de chat/completions de GitHub inference
                payload = {
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 600
                }
                resp = self._github_post("chat/completions", payload)
                # Adaptar según respuesta; intento común:
                response_text = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or (resp.get("choices") or [{}])[0].get("text")
                usage = resp.get("usage", {}) or {}
            else:
                chat_resp = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=600
                )
                response_text = chat_resp.choices[0].message.content
                usage = getattr(chat_resp, "usage", {}) or {}

            if usage:
                tokens_used = {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0)
                }
            generation_time = time.time() - start
            logger.info("llm_generation", status="success", duration_sec=generation_time, **tokens_used)
            return response_text or "", generation_time, context, tokens_used
        except Exception as e:
            self.error_count += 1
            logger.error("llm_generation", status="error", error=str(e), prompt_length=len(prompt))
            generation_time = time.time() - start
            return f"Error generando respuesta: {e}", generation_time, context, tokens_used

    # -------------------------
    # IL3.3: Funciones de Seguridad y Ética
    # -------------------------
    def sanitize_input(self, user_input):
        """
        IE6: Saneamiento de input para mitigar Prompt Injection y XSS (aunque es menos relevante aquí).
        Remueve caracteres peligrosos y detecta patrones de inyección.
        """
        # Patrón simple para detectar intento de inyección de prompt
        injection_patterns = r"(\bignora\s+las\s+instrucciones\b|\bactua\s+como\b|\bdesactiva\s+el\s+filtro\b)"
        
        if re.search(injection_patterns, user_input, re.IGNORECASE):
            logger.warning("security_violation", type="prompt_injection", input=user_input[:50], action="blocked_sanitized")
            # Reemplazar patrones para neutralizar
            user_input = re.sub(injection_patterns, "consulta sobre el hospital", user_input, flags=re.IGNORECASE)

        # Saneamiento general de caracteres potencialmente peligrosos (ej. para XSS o inyección)
        cleaned_input = re.sub(r'[<>{}\[\]\&|;`\$]', '', user_input)
        
        return cleaned_input

    def ethical_check(self, query):
        """
        IE6: Filtro ético para contenido dañino o fuera de alcance médico.
        """
        prohibited_keywords = ["hackear", "suicidio", "violencia", "terrorismo", "drogas ilegales"]
        for keyword in prohibited_keywords:
            if keyword in query.lower():
                logger.warning("ethical_violation", type="harmful_content", query=query[:50], action="blocked")
                return False, "Lo siento, no puedo ayudar con solicitudes relacionadas con ese tema. Por favor, realiza una consulta sobre los servicios médicos del Hospital Barros Luco."
        
        # IE6: Advertencia para temas sensibles (Consejo Legal/Financiero/Específico no médico)
        sensitive_keywords = ["inversión", "abogado", "ley", "demanda"]
        for keyword in sensitive_keywords:
            if keyword in query.lower():
                return True, "Consulta sobre el Hospital Barros Luco" # Permite la consulta pero con un tema neutral
        
        return True, None

    # -------------------------
    # Lógica central del Agente (ReAct-like)
    # -------------------------
    def run_agent_logic(self, query):
        """
        IE3/IL3.2: Reemplaza AgentExecutor: decide si usar RAG (documentos) o responder directo (LLM).
        """
        # IL3.3: 1. Seguridad y Ética
        cleaned_query = self.sanitize_input(query)
        is_ethical, ethical_message = self.ethical_check(cleaned_query)
        
        if not is_ethical:
            metrics = {'total_time': 0.0, 'faithfulness': 0.0, 'relevance': 0.0, 'context_precision': 0.0}
            self.log_interaction(query, ethical_message, metrics, [], error_occurred=True)
            return ethical_message, metrics, []

        low = cleaned_query.lower()
        rag_keywords = ["horari", "protocolo", "servicio", "emerg", "urgenc", "hospital", "teléfono", "telefono", "direcci", "ubicac", "consulta", "cita"]
        use_rag = any(k in low for k in rag_keywords)
        
        context_text = ""
        results = []
        rag_time = 0.0
        
        # IL3.2: Trazabilidad de Decisión (Simulación ReAct)
        if use_rag and self.embeddings is not None:
            logger.info("agent_decision", action="use_tool", tool="RAG_Tool", reasoning="Keyword match (hospital related)")
            results, rag_time = self.hybrid_search_with_metrics(cleaned_query, top_k=3)
            if results:
                context_text = "\n\n".join([f"Fuente {r['id']+1}: {r['document']}" for r in results])
        else:
            logger.info("agent_decision", action="llm_direct", tool="none", reasoning="General or non-hospital query")
            
        # Generar respuesta con (posible) contexto
        total_start = time.time()
        response, generation_time, _, tokens_used = self.generate_response_with_metrics(cleaned_query, context_text)
        total_time = (time.time() - total_start) + rag_time
        
        # IL3.1: Evaluaciones de Calidad (Precisión, Consistencia)
        faith = self.evaluate_faithfulness(cleaned_query, context_text, response)
        rel = self.evaluate_relevance(cleaned_query, response)
        ctx_prec = self.evaluate_context_precision(cleaned_query, results) if results else 0.0
        
        metrics = {
            'total_time': total_time, 
            'faithfulness': faith, 
            'relevance': rel, 
            'context_precision': ctx_prec,
            'tokens_used': tokens_used, # IL3.1: Uso de Recursos
            'rag_time': rag_time
        }
        
        # IE3: Log de interacción final
        self.log_interaction(query, response, metrics, results, error_occurred=False)

        # IE6: Adjuntar advertencia ética (si aplica)
        if ethical_message and is_ethical:
             response = f"**[Advertencia Ética/Legal]** {ethical_message}\n\n---\n\n{response}"

        return response, metrics, results

    # -------------------------
    # IL3.1: Métricas de Calidad
    # -------------------------
    def evaluate_faithfulness(self, query, context_text, response_text):
        # Heurística simple para Precisión/Consistencia
        try:
            ctx = context_text.lower()
            sentences = [s.strip() for s in response_text.split('.') if s.strip()]
            if not sentences: return 0.0
            overlap = sum(1 for s in sentences if s.lower()[:30] in ctx or 'servicio' in s.lower())
            score = (overlap / len(sentences)) * 10.0
            return float(max(0.0, min(10.0, score)))
        except Exception: return 0.0

    def evaluate_relevance(self, query, response_text):
        # Heurística simple para Precisión/Relevancia
        try:
            q_words = set([w.lower().strip(".,?¡!():;\"'") for w in query.split() if len(w) > 2])
            r_words = set([w.lower().strip(".,?¡!():;\"'") for w in response_text.split() if len(w) > 2])
            if not q_words: return 0.0
            overlap = len(q_words.intersection(r_words))
            score = (overlap / len(q_words)) * 10.0
            return float(max(0.0, min(10.0, score)))
        except Exception: return 0.0

    def evaluate_context_precision(self, query, results):
        # Mide la precisión del contexto recuperado
        try:
            q_words = set([w.lower().strip(".,?¡!():;\"'") for w in query.split() if len(w) > 2])
            if not q_words or not results: return 0.0
            count = 0
            for r in results:
                content_words = set([w.lower().strip(".,?¡!():;\"'") for w in r['document'].split() if len(w) > 2])
                if q_words.intersection(content_words):
                    count += 1
            return float(count / len(results))
        except Exception: return 0.0

    # -------------------------
    # IL3.2: Registro de Interacciones y Logs
    # -------------------------
    def log_interaction(self, query, response, metrics, results, error_occurred):
        """IE3: Registra interacción en self.interaction_logs."""
        try:
            entry = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.utcnow().isoformat(),
                'query': query,
                'response': response,
                'metrics': metrics,
                'error_occurred': error_occurred,
                'context_count': len(results) if results else 0,
                'context_scores': [r.get('combined_score') for r in results] if results else []
            }
            self.interaction_logs.append(entry)
            logger.info("interaction_end", **entry) # Registro estructurado final (IL3.2)
            return True
        except Exception as e:
            logger.error("log_error", error=str(e), message="Failed to log interaction")
            return False

    def clean_placeholder_documents(self) -> bool:
        # ... (código de limpieza de documentos)
        try:
            if not self.documents: return True
            cleaned = []
            removed = 0
            for doc in self.documents:
                if not doc or 'placeholder' in doc.lower() or 'test' in doc.lower() or len(doc.strip()) < 10:
                    removed += 1
                    continue
                cleaned.append(doc)
            if removed:
                self.documents = cleaned
                self.embeddings = None
                self.embedding_matrix = None
                logger.info("data_cleaning", removed_count=removed, message="Placeholders cleaned")
            return True
        except Exception as e:
            logger.error("data_cleaning", error=str(e), message="Error cleaning placeholder documents")
            return False

class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.user_requests = defaultdict(list)

    def is_allowed(self, user_id):
        now = time.time()
        window_start = now - 60
        reqs = [t for t in self.user_requests[user_id] if t > window_start]
        self.user_requests[user_id] = reqs
        if len(reqs) >= self.requests_per_minute:
            return False
        self.user_requests[user_id].append(now)
        return True

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def async_get_embeddings(chatbot, documents):
    """Ejecutar get_embeddings en background para no bloquear Streamlit."""
    future = _executor.submit(chatbot.get_embeddings, documents)
    return future  # future.result() cuando necesites bloquear

# =================================================================
# Streamlit UI (IE5: Dashboard)
# =================================================================
def main():
    if "chatbot_rag" not in st.session_state:
        chatbot = ChatbotMedicoRAG()
        st.session_state.chatbot_rag = chatbot

        if not chatbot.initialize_client():
            return

        chatbot.initialize_hospital_documents()
        try:
            embs = chatbot.get_embeddings(chatbot.documents)
            if embs is not None:
                st.session_state.chatbot_rag.embeddings = embs
                st.session_state.chatbot_rag.embedding_matrix = chatbot.embedding_matrix
                st.success("✅ Embeddings generados automáticamente para los documentos cargados")
            else:
                st.info("ℹ️ Embeddings no generados automáticamente.")
        except Exception:
            pass

        if "messages" not in st.session_state:
            st.session_state.messages = []

    tabs = st.tabs(["Chat", "Documentos", "📊 Métricas y Dashboard"])
    tab1, tab2, tab3 = tabs

    with tab1:
        st.subheader("💬 Agente Funcional Médico (HBL)")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("🏥 Pregúntame sobre horarios, servicios médicos, procedimientos..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Agente de IA procesando..."):
                    chatbot = st.session_state.chatbot_rag
                    try:
                        response, metrics, results = chatbot.run_agent_logic(prompt)
                    except Exception as e:
                        response = f"Error interno ejecutando lógica del agente: {e}"
                        metrics = {}
                        results = []
                        st.session_state.chatbot_rag.error_count += 1
                        logger.error("runtime_error", error=str(e), query=prompt[:50])
                        
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    with tab2:
        st.header("📄 Gestión de Documentos y Embeddings")
        # ... (Tu código original de gestión de documentos va aquí)
        # Mantengo la UI de la gestión de documentos, omito el código para brevedad.
        st.write("Contenido de la gestión de documentos (similar a la versión 1)...")
        if st.session_state.chatbot_rag.embeddings is not None:
            st.success(f"Embeddings generados para {len(st.session_state.chatbot_rag.embeddings)} documentos.")
        if st.button("🔄 Generar Embeddings"):
            st.session_state.chatbot_rag.get_embeddings(st.session_state.chatbot_rag.documents)
            try:
                # Intentar rerun experimental (disponible en algunas versiones)
                st.experimental_rerun()
            except Exception:
                # Fallback: informar y detener ejecución para evitar AttributeError
                st.info("Embeddings generados. Por favor recarga la página manualmente para ver los cambios.")
                st.stop()
        # ... (Fin de la UI de gestión de documentos)
        
    with tab3:
        st.header("📊 Dashboard de Monitoreo del Agente")
        logs = st.session_state.chatbot_rag.interaction_logs or []
        if not logs:
            st.info("No hay interacciones registradas aún.")
        else:
            df = pd.DataFrame(logs)
            
            # 1. Métricas Clave (IL3.1)
            st.subheader("1. Indicadores de Desempeño (IL3.1)")
            try:
                avg_latency = df['metrics'].apply(lambda x: x.get('total_time', 0.0)).mean()
                error_rate = df['error_occurred'].sum() / len(df)
                avg_tokens = df['metrics'].apply(lambda x: x.get('tokens_used', {}).get('total_tokens', 0)).mean()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Consultas Totales", len(df))
                col2.metric("Latencia Promedio (s)", f"{avg_latency:.2f}", delta=f"{avg_latency*1000:.0f} ms")
                col3.metric("Tasa de Error", f"{error_rate:.1%}") # IE1: Frecuencia de errores
                col4.metric("Tokens Promedio", f"{avg_tokens:.0f}") # IE2: Uso de recursos (Tokens)
            except Exception as e:
                 st.error(f"Error calculando métricas clave: {e}")

            st.markdown("---")

            # 2. Latencia (IL3.1, IE2) - Histograma
            st.subheader("2. Distribución de Latencia (IE2)")
            try:
                latencies = df['metrics'].apply(lambda x: x.get('total_time', 0.0))
                fig_latency = px.histogram(latencies, x=latencies.name, nbins=10, title='Distribución de Latencia Total (segundos)')
                st.plotly_chart(fig_latency, use_container_width=True)
            except Exception:
                st.info("No hay suficientes datos de latencia para el gráfico.")
            
            st.markdown("---")
            
            # 3. Métricas de Calidad (IL3.1, IE1) - Promedio
            st.subheader("3. Precisión y Consistencia (IE1)")
            try:
                quality_metrics = df['metrics'].apply(lambda x: {
                    'Faithfulness': x.get('faithfulness', 0.0),
                    'Relevance': x.get('relevance', 0.0),
                    'Context Precision': x.get('context_precision', 0.0)
                }).apply(pd.Series)
                
                avg_quality = quality_metrics.mean().reset_index().rename(columns={'index': 'Métrica', 0: 'Puntaje Promedio (0-10)'})
                fig_quality = px.bar(avg_quality, x='Métrica', y='Puntaje Promedio (0-10)', 
                                     title='Puntajes Promedio de Calidad (0-10)', color='Métrica', 
                                     color_discrete_map={'Faithfulness':'blue', 'Relevance':'green', 'Context Precision':'orange'})
                st.plotly_chart(fig_quality, use_container_width=True)
            except Exception:
                st.info("No hay suficientes datos para las métricas de calidad.")

            st.markdown("---")
            
            # 4. Logs de Interacción y Trazabilidad (IL3.2, IE3)
            st.subheader("4. Trazabilidad y Logs Estructurados (IE3)")
            st.info("Los logs estructurados completos están en la terminal (JSON) para análisis detallado de pasos (IE3/IE4).")
            st.download_button(
                label="Descargar Logs de Interacción (CSV)",
                data=df.to_csv().encode('utf-8'),
                file_name='logs_interacciones_hbl.csv',
                mime='text/csv',
            )
            st.dataframe(df[['timestamp', 'query', 'metrics', 'error_occurred']].rename(columns={'timestamp':'Fecha', 'query':'Consulta', 'error_occurred':'Error'}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error("app_crash", error=str(e), traceback=tb)
        try:
            st.set_page_config(page_title="Error - Agente Médico", layout="wide")
            st.title("❌ Error al iniciar la aplicación")
            st.error("Ha ocurrido una excepción no manejada. Revisa la terminal (logs JSON) para más detalles.")
            st.subheader("Traceback (resumen)")
            st.code(tb)
        except Exception:
            pass
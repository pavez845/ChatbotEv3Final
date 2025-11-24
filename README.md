# ChatbotEV3
Evaluacion 3 Catalina Aguilar y Fernando Pavez

🏥 Agente Funcional Médico - Hospital Barros Luco (V2: Observabilidad)
Este proyecto implementa un Agente Funcional Inteligente para el Hospital Barros Luco, refactorizado para la Evaluación Parcial N°3 (IL3.x), enfocándose en la Observabilidad, la Trazabilidad de Decisiones (ReAct) y la Seguridad de sistemas de IA en producción.

El sistema utiliza un agente RAG para proveer información precisa del hospital, y ahora incluye un Dashboard de Monitoreo para medir rendimiento, latencia, y precisión.

Módulo,Logro,Implementación en V2
IL3.1,Métricas de Observabilidad,"Implementadas métricas de Latencia (total_time), Precisión (Faithfulness, Relevance) y Uso de Recursos (tokens_used)."
IL3.2,Análisis de Registros y Trazabilidad,"Uso de structlog para logs estructurados (JSON) en terminal, registrando las decisiones del agente (RAG vs. LLM) y el tiempo de cada herramienta."
IL3.3,Seguridad y Ética,Implementada Validación/Sanitización de Inputs (sanitize_input) y Filtros Éticos (ethical_check) para prevención de Prompt Injection y contenido inapropiado.
IL3.4,Escalabilidad y Sostenibilidad,"Las métricas generadas (Latency, Tokens, Error Rate) proveen la base de datos para la propuesta de optimización de desempeño y rediseño."

🛠️ 2. Configuración y Prerrequisitos
Prerrequisitos
Python 3.10 o superior (Recomendado: 3.12).

Acceso a Internet.

Clave de API de Inferencia (OpenAI o Azure AI).

Pasos de Instalación
Clonar el Repositorio (Si no lo has hecho ya):

git clone https://github.com/pavez845/ChatbotEV3
cd Chatbot_Ev3

Crear y Activar el Entorno Virtual:

python -m venv entorno
.\entorno\Scripts\Activate.ps1   # Windows (PowerShell)
# source entorno/bin/activate    # Linux/macOS

Instalar las Dependencias:
pip install -r requirements.txt


Ejecuta la aplicación Streamlit:

streamlit run main_rag_agent_v2.py
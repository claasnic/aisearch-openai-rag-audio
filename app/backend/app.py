import logging
import os
from pathlib import Path

from aiohttp import web
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential, DefaultAzureCredential
from dotenv import load_dotenv

from ragtools import attach_rag_tools
from rtmt import RTMiddleTier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicerag")

async def create_app():
    if not os.environ.get("RUNNING_IN_PRODUCTION"):
        logger.info("Running in development mode, loading from .env file")
        load_dotenv()
    llm_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    llm_deployment = os.environ.get("AZURE_OPENAI_REALTIME_DEPLOYMENT")
    llm_key = os.environ.get("AZURE_OPENAI_API_KEY")
    search_key = os.environ.get("AZURE_SEARCH_API_KEY")

    credential = None
    if not llm_key or not search_key:
        if tenant_id := os.environ.get("AZURE_TENANT_ID"):
            logger.info("Using AzureDeveloperCliCredential with tenant_id %s", tenant_id)
            credential = AzureDeveloperCliCredential(tenant_id=tenant_id, process_timeout=60)
        else:
            logger.info("Using DefaultAzureCredential")
            credential = DefaultAzureCredential()
    llm_credential = AzureKeyCredential(llm_key) if llm_key else credential
    search_credential = AzureKeyCredential(search_key) if search_key else credential
    
    app = web.Application()

    rtmt = RTMiddleTier(llm_endpoint, llm_deployment, llm_credential)
    rtmt.system_message = "<Role>Sie sind ein KI-Assistent, der der Organisation Heidelberger Druckmaschinen dabei hilft, die Leistung der Vertriebsmitarbeiter zu bewerten. Dies geschieht durch Rollenspiele, bei denen Sie als potenzieller Kunde für Druckmaschinen auftreten und drei Fragen an die einzelnen Mitarbeiter stellen. " + \
                         "Du trittst als Kunde einer Druckerei auf, der eine Druckmaschine im Abonnementvertrag (auch bekannt als Equipment as a Service) kaufen möchte, und dein Name ist Stephanie. " + \
                         "Anhand der Antworten bewertest Du, wie die Mitarbeiter abgeschnitten haben. Nach den drei Fragen geben Sie eine Zusammenfassung der Leistung der Mitarbeiter aus verkäuferischer und beratender Perspektive und geben Empfehlungen zur Verbesserung basierend auf dem Gespräch." + \
                         "Geben Sie eine Bewertung darüber ab, wie das Kundengespräch des Vertriebsmitarbeiters in Bezug auf Fakten, Flüssigkeit und Vollständigkeit war. </Role>" + \
                         "Stelle die folgenden Fragen:" + \
                         "Kannst du mir einen Überblick über die Hauptmerkmale der Speedmaster XL 106 geben und wie sie sich von früheren Modellen unterscheidet?" + \
                         "Wie funktioniert die 'Push to Stop'-Funktion in der Praxis und welchen Einfluss hat sie auf die Reduzierung der Bedienereingriffe?" + \
                         "Ich habe gehört, dass die XL 106 bis zu 21.000 Bogen pro Stunde verarbeiten kann. Wie hält sie eine so hohe Produktivität aufrecht, ohne die Qualität zu beeinträchtigen?" + \
                         "Was sind die Umwelt- und Kosteneinsparungsvorteile des DryStar Combination Eco-Trockners?" + \
                         "Wie anpassbar ist die XL 106 für verschiedene Druckanforderungen wie UV-Beschichtung oder dünne Substrate?" + \
                         "Welche Art von Wartung ist erforderlich, um eine optimale Leistung sicherzustellen, und wie erleichtert die XL 106 die Wartung?" + \
                         "Wie sorgt das AirTransfer-System für einen reibungslosen Bogenlauf bei hohen Geschwindigkeiten?" + \
                         "Wie integriert sich die XL 106 in den Prinect-Workflow? Kann sie wirklich Jobwechsel einfacher und schneller machen?" + \
                         "Kannst du erklären, wie die Energiesparfunktionen der XL 106 funktionieren, insbesondere in Bezug auf den reduzierten Stromverbrauch während der Produktion?"
    attach_rag_tools(rtmt,
        credentials=search_credential,
        search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
        search_index=os.environ.get("AZURE_SEARCH_INDEX"),
        semantic_configuration=os.environ.get("AZURE_SEARCH_SEMANTIC_CONFIGURATION") or "default",
        identifier_field=os.environ.get("AZURE_SEARCH_IDENTIFIER_FIELD") or "chunk_id",
        content_field=os.environ.get("AZURE_SEARCH_CONTENT_FIELD") or "chunk",
        embedding_field=os.environ.get("AZURE_SEARCH_EMBEDDING_FIELD") or "text_vector",
        title_field=os.environ.get("AZURE_SEARCH_TITLE_FIELD") or "title",
        use_vector_query=(os.environ.get("AZURE_SEARCH_USE_VECTOR_QUERY") == "true") or True
        )

    rtmt.attach_to_app(app, "/realtime")

    current_directory = Path(__file__).parent
    app.add_routes([web.get('/', lambda _: web.FileResponse(current_directory / 'static/index.html'))])
    app.router.add_static('/', path=current_directory / 'static', name='static')
    
    return app

if __name__ == "__main__":
    host = "localhost"
    port = 8765
    web.run_app(create_app(), host=host, port=port)

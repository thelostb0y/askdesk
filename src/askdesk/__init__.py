"""AskDesk — internal AI platform: RAG-grounded multi-agent Q&A as a service.

Public surface for SDK consumers:

    from askdesk.sdk import Client
    answer = Client("http://localhost:8000", api_key="...").ask("How do refunds work?")
"""

__version__ = "0.1.0"

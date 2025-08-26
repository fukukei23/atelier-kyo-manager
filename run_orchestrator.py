from app.utils.ai_research_orchestrator import ResearchOrchestrator as R
o = R(headless=False)
print(o.run("GUCCI"))

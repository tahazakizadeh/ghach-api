from pydantic import BaseModel
from typing import Dict

class AnalysisResponse(BaseModel):
    overall_score: float
    brix: float
    quality: str
    components: Dict[str, float]
    explanation: str
    tips: str

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from analyzer import WatermelonAnalyzer
from models import AnalysisResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="قاچ API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = WatermelonAnalyzer()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

async def get_nemotron_explanation(data):
    prompt = f"""
    تو یک متخصص کشاورزی و تشخیص کیفیت هندوانه هستی.
    تحلیل تصویر: امتیاز کلی {data['overall_score']}/10، Brix: {data['brix']}
    اجزا: {data['components']}
    به زبان فارسی ساده، گرم و کاربردی توضیح بده چرا این هندوانه این امتیاز رو گرفته و چیکار کنه کاربر.
    حداکثر ۳-۴ جمله.
    """
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://ghach.app",
                "X-Title": "Ghach App",
            },
            json={
                "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            },
            timeout=30.0
        )
        result = response.json()
        return result["choices"][0]["message"]["content"]

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_watermelon(file: UploadFile = File(...)):
    contents = await file.read()
    result = analyzer.analyze(contents)
    
    explanation = await get_nemotron_explanation(result)
    
    return AnalysisResponse(
        overall_score=result["overall_score"],
        brix=result["brix"],
        quality=result["quality"],
        components=result["components"],
        explanation=explanation,
        tips="برای بهترین طعم، هندوانه رو در دمای اتاق نگه دارید."
    )

@app.get("/")
def root():
    return {"message": "قاچ API آماده است 🚀"}

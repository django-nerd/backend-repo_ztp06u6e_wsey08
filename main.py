import os
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import db, create_document, get_documents

app = FastAPI(title="SmartNotes AI – Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AIRequest(BaseModel):
    flow: str
    user_note: Optional[str] = None
    user_syllabus: Optional[str] = None
    note1: Optional[str] = None
    note2: Optional[str] = None
    voice_text: Optional[str] = None
    pdf_text: Optional[str] = None
    query: Optional[str] = None


class SaveNoteRequest(BaseModel):
    original_note: str
    processed_note: str
    tags: List[str] = []


@app.get("/")
async def read_root():
    return {"message": "SmartNotes AI Backend Running"}


@app.get("/test")
async def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = os.getenv("DATABASE_NAME") or "❌ Not Set"
            try:
                response["collections"] = db.list_collection_names()
                response["database"] = "✅ Connected & Working"
                response["connection_status"] = "Connected"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


# Map flows to prompts
FLOW_PROMPTS = {
    "summarize": "Summarize this note clearly and simply:\n{{user_note}}",
    "rewrite": "Rewrite and improve this text. Make it clean, clear, and high quality:\n{{user_note}}",
    "bullet_points": "Turn this into clean bullet points:\n{{user_note}}",
    "short_version": "Create a short and crisp version of this:\n{{user_note}}",
    "eli5": "Explain the following note like I am 5 years old:\n{{user_note}}",
    "flashcards": "Create flashcards (term : definition) from this text:\n{{user_note}}",
    "mcqs": "Create 5 MCQs with correct answers from this text:\n{{user_note}}",
    "short_questions": "Create 5 short answer questions with answers from this text:\n{{user_note}}",
    "chapter_summary": "Split this note into chapters/sections and summarize each one:\n{{user_note}}",
    "mindmap": "Convert this note into a hierarchical mindmap:\n\nTopic\n\nSubtopic\n\nPoints\nText: {{user_note}}",
    "smart_tags": "Generate 3–6 topic tags for this note:\n{{user_note}}",
    "study_plan": "Create a 7-day and 30-day study plan for the following syllabus/topic:\n{{user_syllabus}}",
    "compare_notes": "Compare these two notes and list similarities, differences, and key insights:\nNote 1: {{note1}}\nNote 2: {{note2}}",
    "voice_cleanup": "Clean and format this speech transcript. Remove filler words.\nTranscript: {{voice_text}}",
    "pdf_extract_summary": "Extract main ideas and important points from this PDF text:\n{{pdf_text}}",
    "memory_recall": "From the saved notes, find the notes most related to: {{query}}\nSummarize the findings in simple language.",
}


def simple_rule_engine(prompt: str) -> str:
    """
    Lightweight stand-in for an AI model using heuristic formatting so the app is fully functional
    without external API keys. For production, replace with actual LLM calls.
    """
    # Very naive formatting: create bullets, summaries, etc.
    lines = [l.strip() for l in prompt.splitlines() if l.strip()]
    body = "\n".join(lines[1:]) if len(lines) > 1 else (lines[0] if lines else "")
    header = lines[0] if lines else ""

    if "bullet points" in header.lower():
        bullets = [f"- {s.strip()}" for s in body.split('.') if s.strip()]
        return "\n".join(bullets) or body
    if "short and crisp" in header.lower():
        return (body[:220] + ("…" if len(body) > 220 else "")).strip()
    if "explain the following note like i am 5" in header.lower():
        return "Imagine you are 5: " + body.replace(" therefore", " so").replace(" thus", " so")
    if "flashcards" in header.lower():
        parts = [p.strip() for p in body.split('.') if p.strip()]
        return "\n".join([f"Term {i+1}: {p}\nDefinition: {p}" for i, p in enumerate(parts[:8])]) or body
    if "mcqs" in header.lower():
        qs = [p.strip() for p in body.split('.') if p.strip()][:5]
        out = []
        for i, q in enumerate(qs):
            out.append(f"Q{i+1}. {q}?")
            out.append("A) Option 1")
            out.append("B) Option 2")
            out.append("C) Option 3")
            out.append("D) Option 4")
            out.append("Answer: A")
        return "\n".join(out) or body
    if "short answer questions" in header.lower():
        qs = [p.strip() for p in body.split('.') if p.strip()][:5]
        return "\n".join([f"Q{i+1}. {q}?\nAns: ..." for i, q in enumerate(qs)]) or body
    if "chapters/sections" in header.lower():
        parts = [p.strip() for p in body.split('.') if p.strip()]
        return "\n\n".join([f"Chapter {i+1}: {p}\nSummary: {p}" for i, p in enumerate(parts[:6])]) or body
    if "mindmap" in header.lower():
        return "Topic\n  └─ Subtopic 1\n      └─ Point A\n  └─ Subtopic 2\n      └─ Point B\n\n" + body[:200]
    if "generate 3–6 topic tags" in header.lower():
        # naive tags: take top words
        words = [w.strip(',.:;!"').lower() for w in body.split() if len(w) > 4]
        unique = []
        for w in words:
            if w not in unique:
                unique.append(w)
            if len(unique) >= 6:
                break
        return ", ".join(unique[:6]) or "general, study"
    if "study plan" in header.lower():
        return ("7-Day Plan:\n- Day 1: Read basics\n- Day 2: Key terms\n- Day 3: Practice\n- Day 4: Review\n- Day 5: Quiz\n- Day 6: Revise\n- Day 7: Mock test\n\n" \
                "30-Day Plan:\n- Weeks 1-3: Deep dive and exercises\n- Week 4: Consolidation, mocks, revision")
    if "compare these two notes" in header.lower():
        return "Similarities: ...\nDifferences: ...\nKey insights: ..."
    if "clean and format this speech transcript" in header.lower():
        return body.replace(" uh ", " ").replace(" um ", " ").replace(" kinda ", " ")
    if "extract main ideas" in header.lower():
        bullets = [f"• {s.strip()}" for s in body.split('.') if s.strip()]
        return "\n".join(bullets) or body
    if "summarize this note" in header.lower():
        return "Summary: " + (body[:400] + ("…" if len(body) > 400 else ""))
    if "rewrite and improve" in header.lower():
        return body.replace("very", "extremely").replace("good", "excellent").strip()

    return body


@app.post("/api/ai")
async def ai_tools(req: AIRequest):
    if req.flow not in FLOW_PROMPTS:
        raise HTTPException(status_code=400, detail="Unknown flow")

    prompt = FLOW_PROMPTS[req.flow]
    # Interpolate variables
    prompt = prompt.replace("{{user_note}}", req.user_note or "") \
                   .replace("{{user_syllabus}}", req.user_syllabus or "") \
                   .replace("{{note1}}", req.note1 or "") \
                   .replace("{{note2}}", req.note2 or "") \
                   .replace("{{voice_text}}", req.voice_text or "") \
                   .replace("{{pdf_text}}", req.pdf_text or "") \
                   .replace("{{query}}", req.query or "")

    output = simple_rule_engine(prompt)
    return {"output": output}


@app.post("/api/save")
async def save_note(req: SaveNoteRequest):
    doc = {
        "original_note": req.original_note,
        "processed_note": req.processed_note,
        "tags": req.tags,
        "timestamp": datetime.utcnow(),
    }
    try:
        _id = create_document("saved_notes", doc)
        return {"id": _id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def history():
    try:
        notes = get_documents("saved_notes", {}, limit=100)
        # Normalize ObjectId for JSON
        for n in notes:
            n["id"] = str(n.get("_id"))
            if "_id" in n:
                del n["_id"]
        # Sort newest first
        notes.sort(key=lambda x: x.get("timestamp") or x.get("created_at") or datetime.utcnow(), reverse=True)
        return {"notes": notes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/memory-recall")
async def memory_recall(query: str = Form(...)):
    try:
        notes = get_documents("saved_notes", {}, limit=100)
        # very naive match by substring count
        def score(n):
            text = (n.get("original_note", "") + " " + n.get("processed_note", "")).lower()
            return text.count(query.lower())
        top = sorted(notes, key=score, reverse=True)[:5]
        summary_lines = [f"- { (n.get('processed_note') or n.get('original_note') or '')[:80]}..." for n in top]
        return {"summary": "\n".join(summary_lines), "matches": [str(n.get("_id")) for n in top]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Basic text extract: read bytes and attempt utf-8 decode fallback
    content = await file.read()
    try:
        text = content.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    return {"text": text[:100000]}


@app.post("/api/upload/text")
async def upload_text(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    return {"text": text[:100000]}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# app.py    
import re
import threading
from typing import Dict, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diary_assistant import TextDiaryAssistant

app = FastAPI(title="Diary Bot API", version="1.0.1")

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: Dict[int, TextDiaryAssistant] = {}
_lock = threading.Lock()

def get_session(user_id: int) -> TextDiaryAssistant:
    with _lock:
        if user_id not in _sessions:
            _sessions[user_id] = TextDiaryAssistant()
        return _sessions[user_id]

# 요청/응답 모델
class DialogReq(BaseModel):
    userId: int
    dialog: str

def ok(data: dict):
    return {"statusCode": "200", "message": "요청 성공", "data": data, "isSuccess": True}

def err(msg: str, code: str = "500"):
    return {"statusCode": code, "message": msg, "data": None, "isSuccess": False}

@app.get("/health")
def health():
    return ok({"ok": True})

@app.post("/chat")
def chat(req: DialogReq):
    try:
        sess = get_session(req.userId)
        reply = sess.get_ai_response(req.dialog)
        return ok({"reply": reply, "isSufficient": bool(getattr(sess, "pending_confirm", None))})
    except Exception as e:
        return err(f"대화 처리 중 오류: {e}")

@app.get("/diary")
def diary(userId: int = Query(..., description="세션 사용자 ID")):
    try:
        sess = get_session(userId)
        if not getattr(sess, "chat_history", None):
            return err("아직 대화가 없습니다. /chat 먼저 호출하세요.", code="400")
        text = sess.generate_diary_text()

        # 기본값
        title = "오늘의 이야기"
        emotion = "HAPPY"

        # 제목 파싱
        m = re.search(r"제목\s*:\s*(.+)", text)
        if m:
            title = m.group(1).strip()

        # 감정 파싱
        m = re.search(r"오늘의\s*기분\s*:\s*(.+)", text)
        if m:
            emo_raw = m.group(1).strip().lower()
            if any(k in emo_raw for k in ["행복", "기쁨", "좋음"]): emotion = "HAPPY"
            elif any(k in emo_raw for k in ["슬픔", "우울"]): emotion = "SAD"
            elif any(k in emo_raw for k in ["신남", "설렘"]): emotion = "EXCITED"
            elif any(k in emo_raw for k in ["화", "분노", "짜증"]): emotion = "ANGRY"
            elif any(k in emo_raw for k in ["긴장", "걱정"]): emotion = "NERVOUS"
            elif any(k in emo_raw for k in ["편안", "평온"]): emotion = "CALM"
            elif any(k in emo_raw for k in ["피곤", "졸림"]): emotion = "TIRED"
            elif any(k in emo_raw for k in ["뿌듯", "자랑"]): emotion = "PROUD"

        # 본문 파싱
        parts = re.split(r"\[일기\s*내용\]\s*", text, maxsplit=1)
        content = parts[1].strip() if len(parts) == 2 else text

        return ok({"title": title, "content": content, "emotion": emotion})
    except Exception as e:
        return err(f"일기 생성 중 오류: {e}")

@app.get("/comment")
def comment(
    content: Optional[str] = Query(
        None,
        description="일기 본문. 없으면 세션의 마지막 일기 본문 사용"
    )
):
    try:
        # userId 세션 의존 없이 0번 세션 사용
        sess = get_session(0)
        if content is not None:
            setattr(sess, "diary_content", content)
        out = sess.generate_comment()
        return ok({"comment": out})
    except Exception as e:
        return err(f"코멘트 생성 중 오류: {e}")

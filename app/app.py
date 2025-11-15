# app.py
import re
import threading
from typing import Dict, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from diary_assistant import TextDiaryAssistant

app = FastAPI(title="Diary Bot API", version="1.2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 운영에선 프론트 도메인만 허용 권장
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

# ===== Request Models =====
class DialogReq(BaseModel):
    userId: int
    dialog: str

class DiaryReq(BaseModel):
    userId: int

class CommentReq(BaseModel):
    userId: Optional[int] = 0
    content: Optional[str] = None

class DiaryEnglishReq(BaseModel):
    userId: int
    diaryText: Optional[str] = None  # 주면 그 본문을 번역, 없으면 세션 최신 일기 본문 사용

# ===== Helpers =====
def ok(data: dict):
    return {"statusCode": "200", "message": "요청 성공", "data": data, "isSuccess": True}

def err(msg: str, code: str = "500"):
    return {"statusCode": code, "message": msg, "data": None, "isSuccess": False}

# ===== Endpoints =====
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

# (변경) GET → POST
@app.post("/diary")
def diary(req: DiaryReq):
    try:
        sess = get_session(req.userId)
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

# (변경) GET → POST
@app.post("/comment")
def comment(req: CommentReq):
    try:
        sess = get_session(req.userId or 0)
        if req.content is not None:
            setattr(sess, "diary_content", req.content)
        out = sess.generate_comment()
        return ok({"comment": out})
    except Exception as e:
        return err(f"코멘트 생성 중 오류: {e}")

# (추가) 일기 본문 영어만 text/plain으로 반환
@app.post("/diary/english", response_class=PlainTextResponse)
def diary_english(req: DiaryEnglishReq):
    try:
        sess = get_session(req.userId)

        # req.diaryText가 오면 그 본문을 번역, 없으면 세션의 최신 일기를 사용
        if req.diaryText and req.diaryText.strip():
            prompt_en = (sess.diary_body_english(req.diaryText) or "").strip()
        else:
            # 대화가 하나도 없으면 생성 불가
            if not getattr(sess, "chat_history", None):
                return PlainTextResponse("No conversation yet. Call /chat first.", status_code=400)
            # 최신 일기 확보 후 영어 본문 생성
            _ = sess.generate_diary_text()
            prompt_en = (sess.diary_body_english() or "").strip()

        if not prompt_en:
            return PlainTextResponse("Empty diary body.", status_code=400)

        return PlainTextResponse(prompt_en, status_code=200)
    except Exception as e:
        return PlainTextResponse(f"Failed to build English diary body: {e}", status_code=500)


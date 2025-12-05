# app.py
import threading
from typing import Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from diary_assistant import TextDiaryAssistant

# ======================
# FastAPI 기본 설정
# ======================
API_TITLE = "Diary Bot API"

app = FastAPI(title=API_TITLE)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# 세션 관리
# ======================
_sessions: Dict[int, TextDiaryAssistant] = {}
_lock = threading.Lock()


def get_session(user_id: int) -> TextDiaryAssistant:
    """
    user_id 별로 TextDiaryAssistant 인스턴스를 하나씩 유지.
    멀티스레드 환경을 고려해 Lock으로 보호.
    """
    with _lock:
        if user_id not in _sessions:
            _sessions[user_id] = TextDiaryAssistant()
        return _sessions[user_id]


# ======================
# Request Models
# ======================
class ResetReq(BaseModel):
    userId: int

    
class DialogReq(BaseModel):
    userId: int
    dialog: str


class DiaryReq(BaseModel):
    userId: int


class CommentReq(BaseModel):
    userId: int
    content: Optional[str] = None


class DiaryEnglishReq(BaseModel):
    userId: int
    # 주어지면 해당 일기 본문을 번역, 없으면 세션 최신 일기 본문 사용
    diaryText: Optional[str] = None


# ======================
# 공통 응답 Helper
# ======================
def ok(data: dict) -> dict:
    return {
        "statusCode": "200",
        "message": "요청 성공",
        "data": data,
        "isSuccess": True,
    }


def err(msg: str, code: str = "500") -> dict:
    return {
        "statusCode": code,
        "message": msg,
        "data": None,
        "isSuccess": False,
    }


# ======================
# Endpoints
# ======================
@app.get("/health")
def health():
    return ok({"ok": True})

@app.post("/session/reset")
def session_reset(req: ResetReq):
    try:
        sess = get_session(req.userId)
        sess.reset()
        return ok({"reset": True})
    except Exception as e:
        return err(f"세션 리셋 중 오류: {e}")


@app.post("/chat")
def chat(req: DialogReq):
    try:
        sess = get_session(req.userId)
        reply = sess.get_ai_response(req.dialog)
        return ok({
            "reply": reply,
            "isSufficient": bool(getattr(sess, "pending_confirm", None)),
        })
    except Exception as e:
        return err(f"대화 처리 중 오류: {e}")


@app.post("/diary")
def diary(req: DiaryReq):
    """
    - 현재 세션의 대화를 기반으로 일기를 생성
    - title: 일기 제목
    - content: 일기 본문
    - emotion: 일기 내용을 기반으로 한 5가지 감정 중 하나
    """
    try:
        sess = get_session(req.userId)

        if not getattr(sess, "chat_history", None):
            return err("아직 대화가 없습니다. /chat 먼저 호출하세요.", code="400")

        # 1) 전체 일기 텍스트 생성 (제목 + 오늘의 기분 + [일기 내용])
        full_text = sess.generate_diary_text()

        # 2) 제목 / 본문 분리
        parts = sess.extract_diary(full_text)
        title = (parts.get("title") or "오늘의 이야기").strip()
        content = (parts.get("body") or "").strip()

        # 3) "본문(content)"만 기반으로 감정 추론
        emotion = sess._infer_mood_from_text(content)

        return ok({"title": title, "content": content, "emotion": emotion})
    except Exception as e:
        return err(f"일기 생성 중 오류: {e}")


@app.post("/diary/english", response_class=PlainTextResponse)
def diary_english(req: DiaryEnglishReq):
    """
    - diaryText가 오면 그 본문을 영어로 번역하여 text/plain으로 반환
    - 없으면 현재 세션의 최신 일기를 생성(or 재사용)한 뒤,
      그 일기 본문을 영어로 변환하여 반환
    """
    try:
        sess = get_session(req.userId)

        # 외부에서 일기 본문을 직접 넘겨준 경우
        if req.diaryText and req.diaryText.strip():
            prompt_en = (sess.diary_body_english(req.diaryText) or "").strip()
        else:
            # 대화가 하나도 없으면 생성 불가
            if not getattr(sess, "chat_history", None):
                return PlainTextResponse(
                    "No conversation yet. Call /chat first.",
                    status_code=400,
                )

            # 최신 일기 확보 후 영어 본문 생성
            _ = sess.generate_diary_text()
            prompt_en = (sess.diary_body_english() or "").strip()

        if not prompt_en:
            return PlainTextResponse("Empty diary body.", status_code=400)

        return PlainTextResponse(prompt_en, status_code=200)
    except Exception as e:
        return PlainTextResponse(
            f"Failed to build English diary body: {e}",
            status_code=500,
        )


@app.post("/comment")
def comment(req: CommentReq):
    """
    - content가 오면 해당 내용을 바탕으로 코멘트 생성
    - content가 없으면 세션에 저장된 마지막 일기(self.diary_content)를 사용
    """
    try:
        sess = get_session(req.userId)
        if req.content is not None:
            setattr(sess, "diary_content", req.content)
        out = sess.generate_comment()
        return ok({"comment": out})
    except Exception as e:
        return err(f"코멘트 생성 중 오류: {e}")

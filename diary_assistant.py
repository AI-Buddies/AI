# diary_assistant.py
# -*- coding: utf-8 -*-
import os, re, json, random
from typing import List, Optional, Dict, Tuple
from google import genai
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.5-flash-lite"

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
GENAI_CLIENT = genai.Client(api_key=API_KEY)

SYSTEM_INSTRUCTION = (
    "당신은 초등학생의 눈높이에 맞춰 대화하는 친절한 AI 친구 '다이어리 봇'입니다.\n"
    "규칙:\n"
    "1) 매 답변은 '공감 1문장' + '질문 1문장'으로 말하시오.\n"
    "2) 아이가 말하지 않은 사실·감정을 단정하거나 전제로 물어보지 마시오.\n"
    "3) 질문은 항상 개방형(무엇/어디/언제/누가). '왜' 질문 금지.\n"
    "4) 개인정보 요구 금지.\n"
    "5) 질문은 자연스러운 구어체 반말 한 문장, '~이야기해 줄 수 있니?/말해줄 수 있니?/알려줄 수 있니?' 같은 문형 금지.\n"
    "6) 이모지 사용 금지.\n"
)

# LLM이 선택해야 하는 감정 라벨(5개 고정)
ALLOWED_MOODS = ["happy", "amazed", "sad", "angry", "anxiety"]


class TextDiaryAssistant:
    # =======================
    # 슬롯 정의
    # =======================
    slot_order = ["who", "when", "where", "what"]
    slot_kor = {
        "who": "누가(함께한 사람)",
        "when": "언제(시점)",
        "where": "어디서(장소)",
        "what": "무엇을(활동)",
    }

    # =======================
    # 초기화
    # =======================
    def __init__(self):
        self.client = GENAI_CLIENT
        self.chat_history: List[Tuple[str, str]] = []
        self.diary_content: str = ""
        self.slots: Dict[str, Optional[str]] = {k: None for k in self.slot_order}
        self.pending_confirm: bool = False
        self.user_turns: int = 0

        # =======================
        # 대화 히스토리 / 슬롯 관련
        # =======================
    def _history_text(self) -> str:
        if not self.chat_history:
            return "없음"
        lines: List[str] = []
        for u, a in self.chat_history:
            lines.append(f"아이: {u}")
            lines.append(f"AI: {a}")
        return "\n".join(lines)

    def _merge_slots(self, new_info: dict):
        for k, v in (new_info or {}).items():
            if k in self.slots and v and self.slots[k] is None:
                self.slots[k] = str(v).strip()

    def _extract_slots_from_utterance(self, user_text: str) -> dict:
        ask = (
            "아래 아이의 한 문장에서 기본 요소(누가, 언제, 어디서, 무엇을)를 추출하시오.\n"
            'JSON으로만 출력하고, 모르면 null로 설정하시오. 키는 ["who","when","where","what"].\n'
            '예) {"who":"친구들","when":"오늘 오후","where":null,"what":"축구"}\n\n'
            f'아이의 말: """{user_text}"""'
        )
        try:
            res = self.client.models.generate_content(model=MODEL_NAME, contents=ask)
            raw = (res.text or "").strip()
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e != -1 and e > s:
                try:
                    obj = json.loads(raw[s:e+1])
                    return {k: (obj.get(k) or None) for k in self.slot_order}
                except json.JSONDecodeError:
                    logger.warning("Slot JSON decode failed: %s", raw)
        except Exception as e:
            logger.error("Slot extraction failed: %s", e)
        return {}

    def _all_ready(self) -> bool:
        return all(self.slots[k] is not None for k in self.slot_order)

    def _missing_all(self):
        return [k for k in self.slot_order if self.slots[k] is None]

    def _choose_focus_slot(self, missing: list) -> Optional[str]:
        order = ["who", "what", "when", "where"]
        for k in order:
            if k in missing:
                return k
        return None
    
    def reset(self):
        """새로운 대화를 시작할 때 세션 상태 초기화"""
        self.chat_history.clear()
        self.diary_content = ""
        self.slots = {k: None for k in self.slot_order}
        self.pending_confirm = False
        self.user_turns = 0

    # =======================
    # 질문 백업 / 일기 제안
    # =======================
    def _fallback_question(self, slot: Optional[str], context: str) -> str:
        bank = {
            "who": ["누구랑 같이 있었어?", "함께한 사람은 누구였어?", "친구 중에 누가 있었어?"],
            "when": ["몇 시쯤이었어?", "오늘 중에 언제였어?", "수업 전이었어, 아니면 끝나고였어?"],
            "where": ["정확히 어디에서 있었어?", "그곳은 어떤 곳이었어?"],
            "what": ["무엇을 했는지 한 가지만 더 말해줄래?", "그때 뭐가 제일 재미있었어?", "구체적으로 어떤 활동이었어?"],
        }
        choices = bank.get(slot) or ["어떤 부분이 제일 기억에 남았어?"]
        return random.choice(choices)

    def _diary_offer(self) -> str:
        return random.choice([
            "이 정도면 충분해! 지금 내용으로 일기 써볼까?",
            "좋아, 이야기 많이 모였어. 바로 일기로 옮겨볼까?",
            "이야기 고마워! 이걸로 일기 써볼까?",
        ])

    # =======================
    # 메인 대화 응답
    # =======================
    def get_ai_response(self, user_text: str) -> str:
        user_text = (user_text or "").strip()
        if not user_text:
            return "무슨 일이 있었는지 한 문장으로만 말해줄래?"

        self.user_turns += 1
        self._merge_slots(self._extract_slots_from_utterance(user_text))

        ready = self._all_ready()
        missing = self._missing_all()
        missing_kor = [self.slot_kor[k] for k in missing]
        min_turns_met = self.user_turns >= 6
        focus_slot = self._choose_focus_slot(missing)
        last_ai = self.chat_history[-1][1] if self.chat_history else ""

        # 상태 모드 결정
        if ready and min_turns_met:
            mode = "OFFER_DIARY"
        elif not ready:
            mode = "ASK_SLOT"
        else:
            # 슬롯은 다 찼지만 min_turns는 아직 안 됨
            mode = "DEEPEN_DETAIL"

        if mode == "OFFER_DIARY":
            guidance = (
                "지금은 일기를 쓸 준비가 모두 끝난 상태야.\n"
                "공감 1문장 후, 자연스럽게 '이제 이 이야기로 일기를 써보자'는 제안을 한 문장으로 해.\n"
                "누가/언제/어디서/무엇을 같은 기본 정보는 다시 묻지 마.\n"
            )
        elif mode == "ASK_SLOT":
            guidance = (
                "아직 채워지지 않은 정보가 있어.\n"
                "공감 1문장 후, 부족한 정보 중 하나(focus_slot)에 대해서만 자연스럽게 한 문장으로 물어봐.\n"
                "이미 채워진 정보는 다시 묻지 마.\n"
            )
        else:  # DEEPEN_DETAIL
            guidance = (
                "일기를 쓰기에 필요한 기본 정보는 이미 충분해.\n"
                "공감 1문장 후, 아이의 느낌이나 기억에 남는 장면 한 가지 등 '부가적인 내용'을 편하게 더 말하게 유도해.\n"
                "누가/언제/어디서/무엇을 같은 기본 정보는 다시 묻지 마.\n"
            )

        # 공통 스타일 규칙 추가
        guidance += (
            "질문은 자연스러운 구어체 반말 한 문장, 고정 문형 금지.\n"
            "이전 턴 질문과 같은 서두/끝어미 반복 금지.\n"
            "이전 AI 질문 문장을 그대로 반복하지 마시오.\n"
            "같은 의미의 질문을 하더라도, 표현과 어휘를 반드시 다르게 바꾸시오.\n"
            "이모지 사용 금지.\n"
        )

        prompt = (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"mode: {mode}\n"
            f"지금까지의 대화:\n{self._history_text()}\n\n"
            f"마지막 아이의 말: {user_text}\n\n"
            f"수집된 정보(누구/언제/어디/무엇): {json.dumps(self.slots, ensure_ascii=False)}\n"
            f"ready_to_write: {str(ready).lower()}\n"
            f"min_turns_met(사용자 입력 ≥ 6): {str(min_turns_met).lower()}\n"
            f"부족한 항목: {', '.join(missing_kor) if missing_kor else '없음'}\n"
            f"focus_slot: {focus_slot or '없음'}\n"
            f"이전 AI 질문: {last_ai or '없음'}\n\n"
            "응답 형식: 공감 1문장 + 질문/제안 1문장 (두 문장, 한국어, 반말).\n"
            f"{guidance}"
        )

        try:
            res = self.client.models.generate_content(model=MODEL_NAME, contents=prompt)
            ai = (res.text or "").strip()
            self.pending_confirm = (mode == "OFFER_DIARY")
            self.chat_history.append((user_text, ai))
            return ai
        except Exception:
            # 1차 실패 → 단순 백업 프롬프트로 재시도
            retry_prompt = (
                f"{SYSTEM_INSTRUCTION}\n\n"
                f"지금까지의 대화(순서 유지):\n{self._history_text()}\n\n"
                f"마지막 아이의 말: {user_text}\n\n"
                f"수집된 정보(누구/언제/어디/무엇): {json.dumps(self.slots, ensure_ascii=False)}\n"
                f"ready_to_write: {str(ready).lower()} / min_turns_met: {str(min_turns_met).lower()}\n"
                f"focus_slot: {focus_slot or '없음'}\n\n"
                "두 문장만 출력: 1. 공감 1문장 2. 질문/제안 1문장.\n"
                "개방형만, '왜' 금지, 개인정보 금지, 자연스러운 구어체 반말, 이모지 사용 금지, 고정 문형 금지."
            )
            try:
                res2 = self.client.models.generate_content(model=MODEL_NAME, contents=retry_prompt)
                ai = (res2.text or "").strip()
                self.pending_confirm = bool(ready and min_turns_met)
            except Exception:
                # 2차 실패 → 완전 fallback
                if ready and min_turns_met:
                    ai = f"이야기 잘 들었어! {self._diary_offer()}"
                    self.pending_confirm = True
                else:
                    target = focus_slot or (self._missing_all()[0] if self._missing_all() else None)
                    question = self._fallback_question(target, user_text)
                    ai = f"이해했어! {question}"
                    self.pending_confirm = False

            self.chat_history.append((user_text, ai))
            return ai

    # =========================
    # 제목/본문 클린업 유틸
    # =========================
    def _clean_title(self, raw: str) -> str:
        """
        LLM이 출력한 제목에서 '제목: ' 같은 라벨이나 대괄호를 제거하고,
        한 줄짜리 순수 제목만 남긴다.
        """
        t = (raw or "").strip()
        if not t:
            return "오늘의 이야기"

        # "제목: ", "Title: " 같은 접두어 제거
        t = re.sub(r"^(제목|title)\s*[:：\-\]]\s*", "", t, flags=re.IGNORECASE)

        # 대괄호로 둘러싸여 있으면 제거: [신나는 축구 데이!] -> 신나는 축구 데이!
        m = re.match(r"^\[(.+)\]$", t)
        if m:
            t = m.group(1).strip()

        # 첫 줄만 사용
        t = t.splitlines()[0].strip()

        return t or "오늘의 이야기"

    def _clean_body(self, raw: str) -> str:
        """
        일기 본문에서 제목/라벨/오늘의 기분 같은 부분 제거.
        실제 일기 내용만 남긴다.
        """
        if not raw:
            return ""

        lines = []
        for line in raw.splitlines():
            l = line.strip()

            # 완전 빈 줄은 그대로 둠(문단 구분용)
            if not l:
                lines.append("")
                continue

            # 제목/일기 내용/오늘의 기분 라벨 라인은 제거
            if re.match(r"^(제목|title)\s*[:：]", l, re.IGNORECASE):
                continue
            if re.match(r"^\[?일기\s*내용\]?\s*$", l):
                continue
            if re.match(r"^오늘의\s*기분\s*[:：]", l):
                continue

            lines.append(line)

        body = "\n".join(lines).strip()
        # 연속 빈 줄 3줄 이상 → 2줄로 축소
        body = re.sub(r"\n{3,}", "\n\n", body)
        return body

    # =========================
    # 일기 추출 / 생성
    # =========================
    def extract_diary(self, diary_text: Optional[str] = None) -> Dict[str, str]:
        """
        일기 텍스트에서 제목과 [일기 내용] 구간을 분리.
        - diary_text가 없으면 self.diary_content를 사용.
        - 지원 포맷:
          1) "제목: ~~~" + "[일기 내용]" 블록
          2) 첫 줄 = 제목, 그 아래 줄부터 = 본문
        """
        text = (diary_text or self.diary_content or "").strip()
        if not text:
            return {"title": "오늘의 이야기", "body": ""}

        title = "오늘의 이야기"
        body_raw = text

        # 1) "제목: ..." 형식 먼저 시도
        m = re.search(r"제목\s*:\s*(.+)", text)
        if m:
            title = self._clean_title(m.group(1))
            # '제목: ...' 라인이 있으면 그 라인을 본문에서 제거
            body_raw = re.sub(r"^.*제목\s*:.+\n?", "", text, count=1, flags=re.MULTILINE)
        else:
            # 2) "제목:"이 없으면 첫 번째 non-empty 줄을 제목으로 사용
            lines = text.splitlines()
            first_idx = None
            for idx, line in enumerate(lines):
                if line.strip():
                    first_idx = idx
                    break

            if first_idx is not None:
                title_candidate = lines[first_idx]
                title = self._clean_title(title_candidate)
                # 제목 줄 이후를 본문 후보로 사용
                rest_lines = lines[first_idx + 1 :]
                body_raw = "\n".join(rest_lines).lstrip("\n")

        # 3) [일기 내용] 블록이 있으면 그 뒤만 사용
        parts = re.split(r"\[일기\s*내용\]\s*", body_raw, maxsplit=1)
        if len(parts) == 2:
            body_raw = parts[1]

        # 4) 본문 정리(라벨 제거 등)
        body = self._clean_body(body_raw)

        return {"title": title, "body": body}

    def generate_diary_text(self) -> str:
        dialogue = self._history_text()
        slot_summary = "\n".join(
            f"- {self.slot_kor[k]}: {self.slots.get(k) or '미상'}" for k in self.slot_order
        )
        prompt = (
            "다음 대화와 정리된 정보를 바탕으로 어린이의 1인칭(나) 시점에서 일기 한 편을 작성하시오. "
            "반말, 간결한 문장, 실제 행동과 감정이 드러나게 작성하시오.\n\n"
            f"<정리된 정보>\n{slot_summary}\n</정리된 정보>\n\n"
            f"<대화 내용>\n{dialogue}\n</대화 내용>\n\n"
            "<일기 형식>\n"
            "1) 첫 줄에는 일기 제목만 한 줄로 쓰시오. 예: 신나는 축구 데이!\n"
            "2) 둘째 줄은 비워 두고, 셋째 줄부터는 일기 내용을 여러 문장으로 쓰시오.\n"
            "3) '제목:', '[일기 내용]', '오늘의 기분:' 같은 말머리나 설명 문장은 쓰지 마시오.\n"
            "4) 위 형식에 맞는 순수 일기 텍스트만 출력하고, 다른 설명은 절대 쓰지 마시오.\n"
        )
        try:
            res = self.client.models.generate_content(model=MODEL_NAME, contents=prompt)
            text = (res.text or "").replace("*", "").strip()

            self.diary_content = text
            return self.diary_content
        except Exception as e:
            inferred = self._infer_mood_from_text(dialogue)
            fallback = (
                "오늘 있었던 일\n"
                "\n"
                "오늘의 일기 내용."
            )
            self.diary_content = fallback + f"\n\n(오류 메모: {e})"
            return self.diary_content

    # =========================
    # 번역 (영어 본문)
    # =========================
    def translate_to_english(self, text_ko: str) -> str:
        """
        한국어 일기 본문을 자연스러운 영어 '한 문장'으로 번역.
        - 핵심 정보(누가/어디서/무엇을/언제, 감정)를 최대한 포함
        - 이모지/불필요한 설명/줄바꿈 제거
        - 최종적으로 영어 한 문장만 반환
        """
        if not text_ko or not text_ko.strip():
            return ""

        ask = (
            "Read the following Korean diary passage and express it as exactly ONE concise, natural English sentence. "
            "Include key details about who, where, what, when, and the child's feelings if possible. "
            "Avoid emojis, line breaks, bullet points, or extra commentary. "
            "Output only one English sentence.\n\n"
            f'Korean:\n"""{text_ko.strip()}"""'
        )
        try:
            res = self.client.models.generate_content(model=MODEL_NAME, contents=ask)
            out = (res.text or "").strip()

            # 1) 줄바꿈/여러 공백을 공백 하나로 정리
            out = re.sub(r"\s+", " ", out)

            # 2) 혹시 여러 문장이 오면, 첫 번째 문장만 잘라내기
            #    - 마침표/느낌표/물음표 중 제일 먼저 나오는 문장부호까지를 한 문장으로 간주
            m = re.search(r"([^.?!]*[.?!])", out)
            if m:
                return m.group(1).strip()

            # 문장부호가 없으면 전체를 한 문장으로 취급
            return out
        except Exception:
            # 실패 시 원문을 그대로 반환 (영어 한 문장이 아니어도, 최소 동작 보장)
            return text_ko.strip()

    def diary_body_english(self, diary_text: Optional[str] = None) -> str:
        """
        (1) diary_text가 들어오면 그것을 '이미 잘려 있는 일기 본문'으로 간주해서 그대로 사용
        (2) diary_text가 없으면 self.diary_content에서 [일기 내용]만 추출
        (3) 한국어이면 영어 한 문장으로 번역해서 반환,
            이미 영어/비한글이면 그대로 반환.
        """
        # 1) 클라이언트가 diaryText를 넘긴 경우 → 그대로 본문으로 사용
        if diary_text is not None and diary_text.strip():
            body = diary_text.strip()
        else:
            # 2) 세션에 저장된 전체 일기 텍스트에서 본문 추출
            parts = self.extract_diary()
            body = (parts.get("body") or "").strip()

        if not body:
            return ""

        # 3) 한글 여부 체크: 한글 코드 범위 있으면 번역 대상
        if not re.search(r"[\uac00-\ud7a3]", body):
            # 이미 영어(또는 비한글)라고 보고 그대로 반환
            return body

        # 4) 여기서 translate_to_english가 이미 '한 문장'으로 강제함
        return (self.translate_to_english(body) or "").strip()

    # ===================
    # 감정 추론 관련
    # ===================
    def _infer_mood_from_text(self, text: str) -> str:
        ask = (
            "다음 내용을 읽고 아이의 주요 감정을 반드시 아래 5가지 중 하나만으로 고르시오.\n"
            "내용은 대화이거나 완성된 일기일 수 있습니다.\n"
            "허용값: happy, amazed, sad, angry, anxiety\n"
            "다른 단어, 동의어, 영어 단어, 설명 문장은 쓰지 마시오.\n"
            '출력 형식은 JSON 한 줄만 허용합니다. 예: {"mood":"happy"}\n\n'
            f"{text}"
        )
        try:
            res = self.client.models.generate_content(model=MODEL_NAME, contents=ask)
            raw = (res.text or "").strip()
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e != -1 and e > s:
                obj = json.loads(raw[s:e+1])
                mood = str(obj.get("mood") or "").strip()
                if mood in ALLOWED_MOODS:
                    return mood
        except Exception:
            pass
        # 기본값
        return "happy"

    def infer_mood_from_diary(self, diary_text: Optional[str] = None) -> str:
        """
        일기 전체 텍스트(또는 대화 히스토리)에서
        [일기 내용] 본문을 우선으로 감정 추론.
        """
        # 1) 우선순위: 인자로 받은 diary_text > self.diary_content
        text = (diary_text or self.diary_content or "").strip()

        # 2) 둘 다 없으면 대화 히스토리로 대체
        if not text:
            text = self._history_text()

        # 3) extract_diary를 사용해 [일기 내용] 본문 추출
        parts = self.extract_diary(text)
        body = (parts.get("body") or "").strip()

        # 4) 본문이 비어 있으면 전체 text 기준으로 감정 추론
        target = body or text

        return self._infer_mood_from_text(target)

    # =======================
    # 일기 코멘트 생성
    # =======================
    def generate_comment(self) -> str:
        diary = self.diary_content or "(아직 일기가 없어요)"
        prompt = (
            "다음 일기를 읽고 아이를 칭찬하고 공감하는 따뜻한 코멘트를 2~3줄로 작성해줘.\n"
            "아이 발화에서 드러난 감정 범위 안에서만 표현해줘.\n"
            f'일기: """{diary}"""'
        )
        try:
            res = self.client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return (res.text or "").replace("*", "").strip()
        except Exception as e:
            return f"코멘트 생성 중 오류가 발생했어: {e}"

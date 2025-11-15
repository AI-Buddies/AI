# diary_assistant.py
# -*- coding: utf-8 -*-
import os, re, json, random
from typing import List, Optional, Dict
from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash-lite"

SYSTEM_INSTRUCTION = (
    "너는 초등학생의 눈높이에 맞춰 대화하는 친절한 AI 친구 '다이어리 봇'이야.\n"
    "규칙:\n"
    "1) 매 답변은 '공감 1문장' + '질문 1문장'으로 말해.\n"
    "2) 아이가 말하지 않은 사실·감정을 단정하거나 전제로 물어보지 마.\n"
    "3) 질문은 항상 개방형(무엇/어디/언제/누가). '왜' 질문 금지.\n"
    "4) 개인정보 요구 금지.\n"
    "5) 질문은 자연스러운 구어체 반말 한 문장, '~이야기해 줄 수 있니?/말해줄 수 있니?/알려줄 수 있니?' 같은 문형 금지.\n"
    "6) 이모지는 0~1개만 사용하고, 꼭 필요할 때만 넣어.\n"
)

class TextDiaryAssistant:
    slot_order = ["who", "when", "where", "what"]
    slot_kor = {
        "who": "누가(함께한 사람)",
        "when": "언제(시점)",
        "where": "어디서(장소)",
        "what": "무엇을(활동)",
    }

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        self.client = genai.Client(api_key=api_key)
        self.chat_history = []
        self.diary_content = ""
        self.slots = {k: None for k in self.slot_order}
        self.pending_confirm = False
        self.user_turns = 0

    def _history_text(self) -> str:
        if not self.chat_history:
            return "없음"
        lines = []
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
            "아래 아이의 한 문장에서 기본 요소(누가, 언제, 어디서, 무엇을)를 추출해.\n"
            'JSON으로만 출력하고, 모르면 null로 둬. 키는 ["who","when","where","what"].\n'
            '예) {"who":"친구들","when":"오늘 오후","where":null,"what":"축구"}\n\n'
            f'아이의 말: """{user_text}"""'
        )
        try:
            res = self.client.models.generate_content(model=MODEL_NAME, contents=ask)
            raw = (res.text or "").strip()
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e != -1 and e > s:
                obj = json.loads(raw[s:e+1])
                return {k: (obj.get(k) or None) for k in self.slot_order}
        except Exception:
            pass
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

    def _fallback_question(self, slot: Optional[str], context: str) -> str:
        bank = {
            "who": ["누구랑 같이 있었어?","함께한 사람은 누구였어?","친구 중에 누가 있었어?"],
            "when": ["몇 시쯤이었어?","오늘 중에 언제였어?","수업 전이었어, 아니면 끝나고였어?"],
            "where": ["정확히 어디에서 있었어?","그곳은 어떤 곳이었어?"],
            "what": ["무엇을 했는지 한 가지만 더 말해줄래?","그때 뭐가 제일 재미있었어?","구체적으로 어떤 활동이었어?"],
        }
        choices = bank.get(slot) or ["어떤 부분이 제일 기억에 남았어?"]
        return random.choice(choices)

    def _diary_offer(self) -> str:
        return random.choice([
            "이 정도면 충분해! 지금 내용으로 일기 써볼까?",
            "좋아, 이야기 많이 모였어. 바로 일기로 옮겨볼까?",
            "이야기 고마워! 이걸로 일기 써볼까?",
        ])

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
        guidance = "\n".join([
            "ready_to_write가 true이고 min_turns_met이 true일 때만, 자연스럽게 일기 쓰자고 제안해.",
            "그 외에는 부족한 focus_slot 하나만 자연스럽게 물어봐.",
            "질문은 자연스러운 구어체 반말 한 문장, 고정 문형 금지.",
            "이전 턴 질문과 같은 서두/끝어미 반복 금지.",
            "이모지는 0~1개만 사용(선택).",
        ])
        prompt = (
            f"{SYSTEM_INSTRUCTION}\n\n"
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
            self.pending_confirm = bool(ready and min_turns_met)
            self.chat_history.append((user_text, ai))
            return ai
        except Exception:
            retry_prompt = (
                f"{SYSTEM_INSTRUCTION}\n\n"
                f"지금까지의 대화(순서 유지):\n{self._history_text()}\n\n"
                f"마지막 아이의 말: {user_text}\n\n"
                f"수집된 정보(누구/언제/어디/무엇): {json.dumps(self.slots, ensure_ascii=False)}\n"
                f"ready_to_write: {str(ready).lower()} / min_turns_met: {str(min_turns_met).lower()}\n"
                f"focus_slot: {focus_slot or '없음'}\n\n"
                "두 문장만 출력: 1. 공감 1문장 2. 질문/제안 1문장.\n"
                "개방형만, '왜' 금지, 개인정보 금지, 자연스러운 구어체 반말, 이모지 0~1개, 고정 문형 금지."
            )
            try:
                res2 = self.client.models.generate_content(model=MODEL_NAME, contents=retry_prompt)
                ai = (res2.text or "").strip()
                self.pending_confirm = bool(ready and min_turns_met)
            except Exception:
                if ready and min_turns_met:
                    ai = f"이야기 잘 들었어! {self._diary_offer()}"
                    self.pending_confirm = True
                else:
                    target = self._missing_all()[0] if self._missing_all() else None
                    question = self._fallback_question(target, user_text)
                    ai = f"이해했어! {question}"
                    self.pending_confirm = False
            self.chat_history.append((user_text, ai))
            return ai

    # ---------- 번역 유틸 추가 (이미지 프롬프트용) ----------
    def extract_diary(self, diary_text: Optional[str] = None) -> Dict[str, str]:
        """
        일기 텍스트에서 제목과 [일기 내용] 구간을 분리.
        diary_text가 없으면 self.diary_content를 사용.
        """
        text = (diary_text or self.diary_content or "").strip()
        # 제목
        title = "오늘의 이야기"
        m = re.search(r"제목\s*:\s*(.+)", text)
        if m:
            title = m.group(1).strip()
        # 본문
        parts = re.split(r"\[일기\s*내용\]\s*", text, maxsplit=1)
        body = parts[1].strip() if len(parts) == 2 else text
        return {"title": title, "body": body}

    def translate_to_english(self, text_ko: str) -> str:
        """
        한국어 일기 본문을 자연스러운 영어 문장으로 번역
        이모지/이상한 기호 제거
        이미지 생성기 입력을 고려해 간결하고 묘사 중심
        """
        if not text_ko or not text_ko.strip():
            return ""
        ask = (
            "Translate the following Korean diary passage into concise, natural English for an image generation prompt. "
            "Keep concrete details (who/where/what/when) and actions. Avoid emojis, honorifics, or extra commentary. "
            "Return only the English sentences.\n\n"
            f'Korean:\n"""{text_ko.strip()}"""'
        )
        try:
            res = self.client.models.generate_content(model=MODEL_NAME, contents=ask)
            return (res.text or "").strip()
        except Exception:
            # 실패 시 원문을 그대로 반환
            return text_ko.strip()

    def diary_body_english(self, diary_text: Optional[str] = None) -> str:
        # 일기에서 [일기 내용]만 뽑아 영어로 번역하여 반환
        parts = self.extract_diary(diary_text)
        body = (parts.get("body") or "").strip()
        if not body:
            return ""

        # 한글이 없으면 이미 영어(또는 비한글)로 판단 → 그대로 반환
        if not re.search(r"[\uac00-\ud7a3]", body):
            return body

        return (self.translate_to_english(body) or "").strip()

    # ---------- 일기/코멘트 ----------
    def generate_diary_text(self) -> str:
        dialogue = self._history_text()
        slot_summary = "\n".join(
            f"- {self.slot_kor[k]}: {self.slots.get(k) or '미상'}" for k in self.slot_order
        )
        prompt = (
            "다음 대화와 정리된 정보를 바탕으로 어린이의 1인칭(나) 시점에서 일기 한 편을 작성해줘."
            "반말, 간결한 문장, 실제 행동과 감정이 드러나게. 형식은 꼭 지켜줘. 이모지는 절대 사용하지 말 것.\n\n"
            f"<정리된 정보>\n{slot_summary}\n</정리된 정보>\n\n"
            f"<대화 내용>\n{dialogue}\n</대화 내용>\n\n"
            "<일기 형식>\n"
            "제목: [제목]\n"
            "오늘의 기분: [기분]\n\n"
            "[일기 내용]\n"
        )
        try:
            res = self.client.models.generate_content(model=MODEL_NAME, contents=prompt)
            self.diary_content = (res.text or "").replace("*", "").strip()
            return self.diary_content
        except Exception as e:
            return f"일기 생성 중 오류가 발생했어: {e}"

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
# diary_assistant.py
# -*- coding: utf-8 -*-
import os, re, json, random
from typing import List, Optional, Dict
from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash-lite"

SYSTEM_INSTRUCTION = (
    "너는 초등학생의 눈높이에 맞춰 대화하는 친절한 AI 친구 '다이어리 봇'이야.\n"
    "규칙:\n"
    "1) 매 답변은 '공감 1문장' + '질문 1문장'으로 말해.\n"
    "2) 아이가 말하지 않은 사실·감정을 단정하거나 전제로 물어보지 마.\n"
    "3) 질문은 항상 개방형(무엇/어디/언제/누가). '왜' 질문 금지.\n"
    "4) 개인정보 요구 금지.\n"
    "5) 질문은 자연스러운 구어체 반말 한 문장, '~이야기해 줄 수 있니?/말해줄 수 있니?/알려줄 수 있니?' 같은 문형 금지.\n"
    "6) 이모지는 0~1개만 사용하고, 꼭 필요할 때만 넣어.\n"
)

class TextDiaryAssistant:
    slot_order = ["who", "when", "where", "what"]
    slot_kor = {
        "who": "누가(함께한 사람)",
        "when": "언제(시점)",
        "where": "어디서(장소)",
        "what": "무엇을(활동)",
    }

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        self.client = genai.Client(api_key=api_key)
        self.chat_history = []
        self.diary_content = ""
        self.slots = {k: None for k in self.slot_order}
        self.pending_confirm = False
        self.user_turns = 0

    def _history_text(self) -> str:
        if not self.chat_history:
            return "없음"
        lines = []
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
            "아래 아이의 한 문장에서 기본 요소(누가, 언제, 어디서, 무엇을)를 추출해.\n"
            'JSON으로만 출력하고, 모르면 null로 둬. 키는 ["who","when","where","what"].\n'
            '예) {"who":"친구들","when":"오늘 오후","where":null,"what":"축구"}\n\n'
            f'아이의 말: """{user_text}"""'
        )
        try:
            res = self.client.models.generate_content(model=MODEL_NAME, contents=ask)
            raw = (res.text or "").strip()
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e != -1 and e > s:
                obj = json.loads(raw[s:e+1])
                return {k: (obj.get(k) or None) for k in self.slot_order}
        except Exception:
            pass
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

    def _fallback_question(self, slot: Optional[str], context: str) -> str:
        bank = {
            "who": ["누구랑 같이 있었어?","함께한 사람은 누구였어?","친구 중에 누가 있었어?"],
            "when": ["몇 시쯤이었어?","오늘 중에 언제였어?","수업 전이었어, 아니면 끝나고였어?"],
            "where": ["정확히 어디에서 있었어?","그곳은 어떤 곳이었어?"],
            "what": ["무엇을 했는지 한 가지만 더 말해줄래?","그때 뭐가 제일 재미있었어?","구체적으로 어떤 활동이었어?"],
        }
        choices = bank.get(slot) or ["어떤 부분이 제일 기억에 남았어?"]
        return random.choice(choices)

    def _diary_offer(self) -> str:
        return random.choice([
            "이 정도면 충분해! 지금 내용으로 일기 써볼까?",
            "좋아, 이야기 많이 모였어. 바로 일기로 옮겨볼까?",
            "이야기 고마워! 이걸로 일기 써볼까?",
        ])

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
        guidance = "\n".join([
            "ready_to_write가 true이고 min_turns_met이 true일 때만, 자연스럽게 일기 쓰자고 제안해.",
            "그 외에는 부족한 focus_slot 하나만 자연스럽게 물어봐.",
            "질문은 자연스러운 구어체 반말 한 문장, 고정 문형 금지.",
            "이전 턴 질문과 같은 서두/끝어미 반복 금지.",
            "이모지는 0~1개만 사용(선택).",
        ])
        prompt = (
            f"{SYSTEM_INSTRUCTION}\n\n"
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
            self.pending_confirm = bool(ready and min_turns_met)
            self.chat_history.append((user_text, ai))
            return ai
        except Exception:
            retry_prompt = (
                f"{SYSTEM_INSTRUCTION}\n\n"
                f"지금까지의 대화(순서 유지):\n{self._history_text()}\n\n"
                f"마지막 아이의 말: {user_text}\n\n"
                f"수집된 정보(누구/언제/어디/무엇): {json.dumps(self.slots, ensure_ascii=False)}\n"
                f"ready_to_write: {str(ready).lower()} / min_turns_met: {str(min_turns_met).lower()}\n"
                f"focus_slot: {focus_slot or '없음'}\n\n"
                "두 문장만 출력: 1. 공감 1문장 2. 질문/제안 1문장.\n"
                "개방형만, '왜' 금지, 개인정보 금지, 자연스러운 구어체 반말, 이모지 0~1개, 고정 문형 금지."
            )
            try:
                res2 = self.client.models.generate_content(model=MODEL_NAME, contents=retry_prompt)
                ai = (res2.text or "").strip()
                self.pending_confirm = bool(ready and min_turns_met)
            except Exception:
                if ready and min_turns_met:
                    ai = f"이야기 잘 들었어! {self._diary_offer()}"
                    self.pending_confirm = True
                else:
                    target = self._missing_all()[0] if self._missing_all() else None
                    question = self._fallback_question(target, user_text)
                    ai = f"이해했어! {question}"
                    self.pending_confirm = False
            self.chat_history.append((user_text, ai))
            return ai

    # ---------- 번역 유틸 추가 (이미지 프롬프트용) ----------
    def extract_diary(self, diary_text: Optional[str] = None) -> Dict[str, str]:
        """
        일기 텍스트에서 제목과 [일기 내용] 구간을 분리.
        diary_text가 없으면 self.diary_content를 사용.
        """
        text = (diary_text or self.diary_content or "").strip()
        # 제목
        title = "오늘의 이야기"
        m = re.search(r"제목\s*:\s*(.+)", text)
        if m:
            title = m.group(1).strip()
        # 본문
        parts = re.split(r"\[일기\s*내용\]\s*", text, maxsplit=1)
        body = parts[1].strip() if len(parts) == 2 else text
        return {"title": title, "body": body}

    def translate_to_english(self, text_ko: str) -> str:
        """
        한국어 일기 본문을 자연스러운 영어 문장으로 번역
        이모지/이상한 기호 제거
        이미지 생성기 입력을 고려해 간결하고 묘사 중심
        """
        if not text_ko or not text_ko.strip():
            return ""
        ask = (
            "Translate the following Korean diary passage into concise, natural English for an image generation prompt. "
            "Keep concrete details (who/where/what/when) and actions. Avoid emojis, honorifics, or extra commentary. "
            "Return only the English sentences.\n\n"
            f'Korean:\n"""{text_ko.strip()}"""'
        )
        try:
            res = self.client.models.generate_content(model=MODEL_NAME, contents=ask)
            return (res.text or "").strip()
        except Exception:
            # 실패 시 원문을 그대로 반환
            return text_ko.strip()

    def diary_body_english(self, diary_text: Optional[str] = None) -> str:
        # 일기에서 [일기 내용]만 뽑아 영어로 번역하여 반환
        parts = self.extract_diary(diary_text)
        body = (parts.get("body") or "").strip()
        if not body:
            return ""

        # 한글이 없으면 이미 영어(또는 비한글)로 판단 → 그대로 반환
        if not re.search(r"[\uac00-\ud7a3]", body):
            return body

        return (self.translate_to_english(body) or "").strip()

    # ---------- 일기/코멘트 ----------
    def generate_diary_text(self) -> str:
        dialogue = self._history_text()
        slot_summary = "\n".join(
            f"- {self.slot_kor[k]}: {self.slots.get(k) or '미상'}" for k in self.slot_order
        )
        prompt = (
            "다음 대화와 정리된 정보를 바탕으로 어린이의 1인칭(나) 시점에서 일기 한 편을 작성해줘."
            "반말, 간결한 문장, 실제 행동과 감정이 드러나게. 형식은 꼭 지켜줘. 이모지는 절대 사용하지 말 것.\n\n"
            f"<정리된 정보>\n{slot_summary}\n</정리된 정보>\n\n"
            f"<대화 내용>\n{dialogue}\n</대화 내용>\n\n"
            "<일기 형식>\n"
            "제목: [제목]\n"
            "오늘의 기분: [기분]\n\n"
            "[일기 내용]\n"
        )
        try:
            res = self.client.models.generate_content(model=MODEL_NAME, contents=prompt)
            self.diary_content = (res.text or "").replace("*", "").strip()
            return self.diary_content
        except Exception as e:
            return f"일기 생성 중 오류가 발생했어: {e}"

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

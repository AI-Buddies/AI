import os, io, hashlib
from datetime import datetime
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel

#from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForText2Image, DEISMultistepScheduler

import boto3

# 환경 변수
#MODEL_ID = "sd-legacy/stable-diffusion-v1-5"
MODEL_ID = "Lykon/dreamshaper-7" 

LORA_DIR = "./lora"
IMAGE_SIZE = 512

S3_BUCKET = os.environ.get("S3_BUCKET", "sketchtalk-s3")   # 버킷명
S3_PREFIX = os.environ.get("S3_PREFIX", "image/") 

DEFAULT_DIARY = "A dog driving a sports car"
ALLOWED_LORAS = {"pastel", "childbook", "coolkids"}
DEFAULT_LORA = "coolkids" #default항상 바뀔 수 있음

LORA_FILES = {
    "pastel":    "PastelKa.safetensors",
    "childbook": "J_huiben.safetensors",
    "coolkids":  "COOLKIDS_MERGE_V2.5.safetensors",
}

# 프롬포트
POSITIVE_PROMPT = ("cute, storybook, illustration, soft lighting, kids, storybook illustration, cute and wholesome, clean line art, simple shapes, paper texture, soft lighting, clear focal point, child-friendly, flat 2D shading, gentle background, high readability")
NEGATIVE_PROMPT = ("minimal background, nsfw, violence, photorealistic,duplicate objects, extra eyes, (worst quality:2), (low quality:2), lowres, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, ((watermark:2)), (white letters:1), (multi nipples), bad anatomy, bad hands, text, error, missing fingers, missing arms, missing legs, extra digit, fewer digits, cropped, worst quality, jpeg artifacts, signature, username, bad feet, blurry, poorly drawn hands, poorly drawn face, mutation, deformed, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed, mutated hands, polar lowres, bad body, bad proportions, gross proportions, wrong feet bottom render, abdominal stretch, briefs, knickers, kecks, thong, {{fused fingers}}, {{bad body}}, bad-picture-chill-75v, ng_deepnegative_v1_75t, EasyNegative, bad proportion body to legs, wrong toes, extra toes, missing toes, weird toes, 2 body, 2 pussy, 2 upper, 2 lower, 2 head, 3 hand, 3 feet, extra long leg, super long leg, mirrored image, mirrored noise, single eye")

#pastel lora에 필요한 Trigger word
# 수정 코드
TRIGGER_WORDS = {
    "childbook": "J_huiben, childbook",
    "pastel": "soft pastel colors, pastel",
}

def _short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]

# Diffusers 파이프라인 초기화(프로세스 1회)
# Diffusers 파이프라인 초기화(프로세스 1회)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    variant="fp16" if dtype == torch.float16 else None
)

pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to(device)
pipe.vae.enable_slicing()
pipe.enable_vae_tiling()


# LoRA 3종 미리 로드 → 요청마다 set_adapters로 전환
for adapter_name, weight_file in LORA_FILES.items():
    full_path = os.path.join(LORA_DIR, weight_file)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"LoRA file not found: {full_path}")

    pipe.load_lora_weights(
        full_path,
        adapter_name=adapter_name
    )

pipe.set_adapters([DEFAULT_LORA])


# S3 클라이언트
s3 = boto3.client("s3")

resp = s3.get_bucket_location(Bucket=S3_BUCKET)
region = resp.get("LocationConstraint") or "us-east-1"

def upload_png_and_get_url(png_bytes: bytes, key: str) -> str:
    s3.put_object(
        Bucket=S3_BUCKET, Key=key, Body=png_bytes,
        ContentType="image/png"
    )
    return f"https://{S3_BUCKET}.s3.{region}.amazonaws.com/{key}"

def _generate_and_upload(diary_text: str, lora_choice: str = DEFAULT_LORA) -> dict:
    # 0) lora_choice 정리 (없거나 잘못 들어오면 기본값)
    lora_choice = (lora_choice or DEFAULT_LORA).lower()
    if lora_choice not in ALLOWED_LORAS:
        lora_choice = DEFAULT_LORA
    
    # 1) LoRA 전환
    pipe.set_adapters([lora_choice])

    # 2) trigger word + 고정 positive를 포함한 프롬프트 구성
    emphasized_diary = f"{diary_text}, {diary_text}"

    base_prompt = emphasized_diary

    # (1) 트리거 단어
    trigger = TRIGGER_WORDS.get(lora_choice)  # 트리거 단어 추가
    if trigger:
        base_prompt = f"{trigger}, {base_prompt}"

    full_prompt = f"{base_prompt}, {POSITIVE_PROMPT}" if POSITIVE_PROMPT else base_prompt

    full_prompt_for_hash = full_prompt   # 해시 생성에 쓰는 프롬프트도 동일하게

    neg_prompt = NEGATIVE_PROMPT

    text_lower = diary_text.lower()
    if ("book" not in text_lower):
        neg_prompt = (
            neg_prompt
            + ", book, books, reading a book, holding a book, open book, child holding a book"
        )
    
    # 3) 파일명/키 생성
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    h = _short_hash(full_prompt_for_hash)
    local_name = f"{lora_choice}_mydiaryimage_{ts}.png"
    s3_key = f"{S3_PREFIX}{lora_choice}/{ts}_{h}.png"

    # 4) 이미지 생성 (프롬프트는 위에서 만든 full_prompt 사용)
    with torch.inference_mode():
        image = pipe(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            guidance_scale=7.5,
            num_inference_steps=50,
        ).images[0]

    # 4) PNG 바이트화 (+선택 로컬 저장)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    png_bytes = buf.read()
    try:
        os.makedirs("./outputs", exist_ok=True)
        image.save(os.path.join("./outputs", local_name))
    except Exception:
        pass

    # 5) S3 업로드 & URL
    image_url = upload_png_and_get_url(png_bytes, s3_key)

    # 6) 명세서 응답
    return {
        "statusCode": "200",
        "message": "요청 성공",
        "data": {"image_url": image_url},
        "isSuccess": True
    }



# FastAPI
app = FastAPI(title="Diary Image Generator", version="1.0.0")

class DiaryBody(BaseModel):
    imageStyle: Optional[str] = None  # pastel / childbook / coolkids
    diary: str

@app.post("/image")
def generate_image_post(body: DiaryBody):
    diary_text = body.diary.strip() if body.diary else DEFAULT_DIARY
    lora_choice = (body.imageStyle or DEFAULT_LORA).strip().lower()

    return _generate_and_upload(diary_text, lora_choice=lora_choice)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

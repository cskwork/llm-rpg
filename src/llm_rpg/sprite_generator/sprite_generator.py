import asyncio
import hashlib
import re
from abc import ABC, abstractmethod
from pathlib import Path
import random
import time
from typing import Optional, Tuple

import pygame
import torch
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    LCMScheduler,
    EulerAncestralDiscreteScheduler,
)
import unfake
from PIL import Image
from llm_rpg.systems.battle.enemy import Enemy
from llm_rpg.llm.llm import LLM
from llm_rpg.utils.logger import get_logger

logger = get_logger(__name__)


class SpriteGenerator(ABC):
    @abstractmethod
    def generate_sprite(self, enemy: Enemy) -> pygame.Surface: ...


def _clean_sprite(sprite: Image.Image) -> Image.Image:
    result = unfake.process_image_sync(sprite, transparent_background=True)
    return result["image"]


def _pil_to_surface(sprite: Image.Image) -> pygame.Surface:
    rgba_sprite = sprite.convert("RGBA")
    size: Tuple[int, int] = rgba_sprite.size
    surface = pygame.image.frombuffer(rgba_sprite.tobytes(), size, "RGBA")
    return surface.convert_alpha()


class DummySpriteGenerator(SpriteGenerator):
    def __init__(self, latency_seconds: float = 0.0):
        self.latency_seconds = latency_seconds
        self._cache: dict[str, pygame.Surface] = {}
        self._sprites_dir = Path(__file__).parent / "dummy_sprites"
        self._sprite_paths = list(self._sprites_dir.glob("*devil_dog.png"))

    def generate_sprite(self, enemy: Enemy) -> pygame.Surface:
        if not self._sprite_paths:
            raise ValueError("No dummy sprites found")

        sprite_path = random.choice(self._sprite_paths)
        cache_key = sprite_path.name
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if self.latency_seconds > 0:
            time.sleep(self.latency_seconds)

        with Image.open(sprite_path) as raw_sprite:
            cleaned_sprite = _clean_sprite(raw_sprite.convert("RGBA"))
        surface = _pil_to_surface(cleaned_sprite)
        self._cache[cache_key] = surface
        return surface


class SDSpriteGenerator(SpriteGenerator):
    def __init__(
        self,
        base_model: str,
        lora_path: str,
        trigger_prompt: str,
        prompt_llm: LLM,
        prompt_template: str,
        lcm_lora_path: Optional[str] = None,
        guidance_scale: float = 7,
        num_inference_steps: int = 20,
        inference_height: int = 512,
        inference_width: int = 512,
        vae_path: Optional[str] = None,
        use_lcm: bool = False,
        negative_prompt: Optional[str] = None,
        debug: bool = False,
    ):
        self.base_model = base_model
        self.lora_path = lora_path
        self.trigger_prompt = trigger_prompt
        self.prompt_llm = prompt_llm
        self.prompt_template = prompt_template
        self.lcm_lora_path = lcm_lora_path
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.inference_height = inference_height
        self.inference_width = inference_width
        self.vae_path = vae_path
        self.use_lcm = use_lcm
        self.negative_prompt = negative_prompt
        self.debug = debug
        self.device = self._get_device()

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _build_sprite_prompt(self, enemy: Enemy) -> str:
        attempts = 0
        while attempts < 3:
            prompt = self.prompt_template.format(
                enemy_name=enemy.name, enemy_description=enemy.description
            )
            try:
                if self.debug:
                    print("////////////DEBUG SpritePrompt LLM prompt////////////")
                    print(prompt)
                    print("////////////DEBUG SpritePrompt LLM prompt////////////")
                output = self.prompt_llm.generate_completion(prompt=prompt)
                if self.debug:
                    print("////////////DEBUG SpritePrompt LLM response////////////")
                    print(output)
                    print("////////////DEBUG SpritePrompt LLM response////////////")
                return output.strip()
            except Exception:
                attempts += 1
                continue
        return enemy.description

    def generate_sprite(self, enemy: Enemy) -> pygame.Surface:
        pipe = StableDiffusionPipeline.from_single_file(
            self.base_model,
            torch_dtype=torch.float16,
        )
        if self.vae_path:
            vae = AutoencoderKL.from_pretrained(
                self.vae_path, torch_dtype=torch.float16
            )
            pipe.vae = vae

        if self.use_lcm and self.lcm_lora_path:
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipe.scheduler.config
            )

        pipe.to(self.device)

        if self.use_lcm and self.lcm_lora_path:
            pipe.load_lora_weights(self.lora_path, adapter_name="style")
            pipe.load_lora_weights(self.lcm_lora_path, adapter_name="lcm")
            pipe.set_adapters(["style", "lcm"], adapter_weights=[1.0, 1.0])
        else:
            pipe.load_lora_weights(self.lora_path)
        sprite_prompt = self._build_sprite_prompt(enemy)
        prompt = f"{self.trigger_prompt}, {sprite_prompt}"
        if self.debug:
            print("////////////DEBUG Diffusion prompt////////////")
            print(prompt)
            print("////////////DEBUG Diffusion prompt////////////")
        sprite = pipe(
            prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            num_images_per_prompt=1,
            negative_prompt=self.negative_prompt,
            height=self.inference_height,
            width=self.inference_width,
            safety_checker=None,
        ).images[0]
        sprite = _clean_sprite(sprite)
        return _pil_to_surface(sprite)


def _sanitize_filename(name: str) -> str:
    """파일명으로 사용 가능하도록 정제"""
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = name.strip()
    if not name:
        return "untitled"
    return name[:50]


def _get_cache_key(enemy: Enemy) -> str:
    """enemy.name 기반 캐시 키 생성"""
    return hashlib.md5(enemy.name.encode()).hexdigest()


class GeminiSpriteGenerator(SpriteGenerator):
    """Gemini 웹 인터페이스를 사용한 스프라이트 생성기"""

    GEMINI_URL = "https://gemini.google.com/app"

    SELECTORS = {
        "input_textarea": [
            'textarea[placeholder*="메시지"]',
            'textarea[placeholder*="message"]',
            "rich-textarea",
            "textarea[data-id]",
            ".ql-editor",
            'div[contenteditable="true"]',
        ],
        "send_button": [
            'button[aria-label*="전송"]',
            'button[aria-label*="send"]',
            'button[data-testid="send-button"]',
            'button[type="submit"]',
        ],
        "generated_images": [
            ".response-content img",
            'img[src*="blob"]',
            'img[alt*="생성"]',
            'img[src*="googleusercontent"]',
        ],
    }

    def __init__(
        self,
        prompt_llm: LLM,
        prompt_template: str,
        cache_dir: str = "cache/sprites",
        state_dir: str = ".playwright_state",
        headless: bool = True,
        debug: bool = False,
    ):
        self.prompt_llm = prompt_llm
        self.prompt_template = prompt_template
        self.cache_dir = Path(cache_dir)
        self.state_dir = Path(state_dir)
        self.headless = headless
        self.debug = debug
        self._cache: dict[str, pygame.Surface] = {}
        self._playwright_initialized = False

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _build_gemini_prompt(self, enemy: Enemy) -> str:
        """LLM을 사용하여 enemy 설명을 Gemini 프롬프트로 변환"""
        attempts = 0
        while attempts < 3:
            prompt = self.prompt_template.format(
                enemy_name=enemy.name,
                enemy_description=enemy.description,
            )
            try:
                if self.debug:
                    print("////////////DEBUG GeminiPrompt LLM prompt////////////")
                    print(prompt)
                    print("////////////DEBUG GeminiPrompt LLM prompt////////////")
                output = self.prompt_llm.generate_completion(prompt=prompt)
                if self.debug:
                    print("////////////DEBUG GeminiPrompt LLM response////////////")
                    print(output)
                    print("////////////DEBUG GeminiPrompt LLM response////////////")
                return output.strip()
            except Exception as e:
                if self.debug:
                    print(
                        f"[DEBUG] GeminiPrompt LLM error (attempt {attempts + 1}): {e}"
                    )
                attempts += 1
        return f"A pixel art RPG enemy sprite for {enemy.name}"

    async def _find_element(self, page, selectors: list, timeout: int = 5000):
        """여러 선택자 중 하나라도 찾으면 반환"""
        for selector in selectors:
            try:
                element = await page.wait_for_selector(selector, timeout=timeout)
                if element:
                    return element
            except Exception:
                continue
        return None

    async def _generate_with_gemini(
        self, prompt: str, enemy_name: str
    ) -> Optional[Path]:
        """Playwright로 Gemini 이미지 생성"""
        try:
            from playwright.async_api import (
                async_playwright,
                TimeoutError as PlaywrightTimeoutError,
            )
        except ImportError:
            raise ImportError(
                "Playwright가 설치되지 않았습니다. "
                "`poetry add playwright aiofiles` 실행 후 "
                "`playwright install chromium`를 실행하세요."
            )

        state_file = self.state_dir / "state.json"
        use_saved_state = state_file.exists()

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                args=(
                    ["--no-sandbox", "--disable-setuid-sandbox"]
                    if not self.headless
                    else []
                ),
            )
            context = await browser.new_context(
                storage_state=str(state_file) if use_saved_state else None,
                viewport={"width": 1280, "height": 720},
            )

            try:
                page = await context.new_page()

                if self.debug:
                    print(f"[DEBUG] Gemini 페이지로 이동...")
                await page.goto(
                    self.GEMINI_URL, wait_until="networkidle", timeout=30000
                )

                if not use_saved_state:
                    if self.headless:
                        print(
                            "[INFO] 첫 실행입니다. 헤드리스 모드를 해제하고 로그인을 완료하세요."
                        )
                        await browser.close()
                        return None
                    print(
                        "[INFO] 수동 로그인이 필요합니다. 브라우저에서 로그인을 완료한 후 Enter를 누르세요..."
                    )
                    input()

                    await context.storage_state(path=str(state_file))
                    self._playwright_initialized = True
                    print("[INFO] 인증 상태가 저장되었습니다.")

                if self.debug:
                    print(f"[DEBUG] 프롬프트 전송: {prompt[:100]}...")

                input_element = await self._find_element(
                    page, self.SELECTORS["input_textarea"], timeout=10000
                )
                if not input_element:
                    print("[ERROR] 입력창을 찾을 수 없습니다")
                    return None

                await input_element.click()
                await input_element.fill("")
                await input_element.type(prompt, delay=30)

                send_button = await self._find_element(
                    page, self.SELECTORS["send_button"], timeout=5000
                )
                if send_button:
                    await send_button.click()
                else:
                    await input_element.press("Enter")

                if self.debug:
                    print("[DEBUG] 이미지 생성 대기 중...")

                start_time = asyncio.get_event_loop().time()
                timeout = 120000

                while True:
                    elapsed = (asyncio.get_event_loop().time() - start_time) * 1000
                    if elapsed > timeout:
                        print("[WARN] 이미지 생성 타임아웃")
                        return None

                    for selector in self.SELECTORS["generated_images"]:
                        try:
                            images = await page.query_selector_all(selector)
                            for img in images:
                                src = await img.get_attribute("src")
                                if src and src.startswith(("http", "blob:")):
                                    if self.debug:
                                        print(
                                            f"[DEBUG] 이미지 생성 확인됨: {src[:50]}..."
                                        )

                                    safe_name = _sanitize_filename(enemy_name)
                                    cache_path = self.cache_dir / f"{safe_name}.png"

                                    response = await page.request.get(src)
                                    content = await response.body()

                                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                                    with open(cache_path, "wb") as f:
                                        f.write(content)

                                    if self.debug:
                                        print(f"[DEBUG] 이미지 저장 완료: {cache_path}")

                                    return cache_path
                        except Exception:
                            pass

                    await asyncio.sleep(1)

            finally:
                await browser.close()

    def _async_generate(self, prompt: str, enemy_name: str) -> Optional[Path]:
        """동기 컨텍스트에서 비동기 함수 실행"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self._generate_with_gemini(prompt, enemy_name)
            )
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Gemini 생성 오류: {e}")
            return None

    def generate_sprite(self, enemy: Enemy) -> pygame.Surface:
        cache_key = _get_cache_key(enemy)
        cached = self._cache.get(cache_key)
        if cached is not None:
            if self.debug:
                print(f"[DEBUG] 캐시된 스프라이트 사용: {enemy.name}")
            return cached

        cache_path = self.cache_dir / f"{_sanitize_filename(enemy.name)}.png"
        if cache_path.exists():
            if self.debug:
                print(f"[DEBUG] 디스크 캐시 사용: {cache_path}")
            with Image.open(cache_path) as img:
                surface = _pil_to_surface(img.convert("RGBA"))
            self._cache[cache_key] = surface
            return surface

        prompt = self._build_gemini_prompt(enemy)

        if self.debug:
            print(f"[DEBUG] Gemini로 스프라이트 생성 시작: {enemy.name}")

        generated_path = self._async_generate(prompt, enemy.name)

        if generated_path and generated_path.exists():
            with Image.open(generated_path) as img:
                surface = _pil_to_surface(img.convert("RGBA"))
            self._cache[cache_key] = surface
            return surface

        print(f"[WARN] Gemini 생성 실패, 더미 스프라이트 사용: {enemy.name}")
        dummy_surface = pygame.Surface((64, 64), pygame.SRCALPHA)
        dummy_surface.fill((128, 128, 128, 255))
        return dummy_surface


class AsciiSpriteGenerator(SpriteGenerator):
    """LLM을 사용하여 적 이름/설명 기반으로 픽셀 아트 스프라이트를 동적으로 생성"""

    SPRITE_SIZE = 64
    GRID_SIZE = 8

    def __init__(self, prompt_llm: LLM, debug: bool = False):
        self.prompt_llm = prompt_llm
        self.debug = debug
        self._cache: dict[str, pygame.Surface] = {}
        logger.info("AsciiSpriteGenerator initialized")

    def _generate_pixel_pattern(self, enemy: Enemy) -> list[str]:
        """LLM을 사용하여 픽셀 패턴 생성"""
        prompt = f"""Create an 8x8 pixel art pattern for this RPG enemy.

Enemy Name: {enemy.name}
Enemy Description: {enemy.description}

Output ONLY 8 lines, each with exactly 8 characters.
Use 'X' for filled pixels and ' ' (space) for empty pixels.
Do not include any explanation, just the 8x8 grid.

Example format:
XXXXXXXX
XXXXXXXX
XXXXXXXX
XXXXXXXX
XXXXXXXX
XXXXXXXX
XXXXXXXX
XXXXXXXX"""

        attempts = 0
        while attempts < 2:
            try:
                if self.debug:
                    logger.debug("Generating pixel pattern with LLM...")
                output = self.prompt_llm.generate_completion(prompt=prompt)

                # 출력에서 8x8 그리드 추출
                lines = []
                for line in output.strip().split("\n"):
                    line = line.strip()
                    # 유효한 행만 추출 (X와 공백만 포함)
                    if set(line) <= {"X", " "} and len(line) == 8:
                        lines.append(line)

                if len(lines) >= 8:
                    pattern = lines[:8]
                    if self.debug:
                        logger.debug(f"Generated pattern:\n" + "\n".join(pattern))
                    return pattern

            except Exception as e:
                if self.debug:
                    logger.error(f"Pattern generation error (attempt {attempts + 1}): {e}", exc_info=True)
                attempts += 1
                continue

        # 실패 시 기본 패턴 반환
        logger.warning("Using default pattern after 2 failed attempts")
        return [
            "  XXXX  ",
            " XXXXXX ",
            "XXXXXXXX",
            "XXXXXXXX",
            "XXXXXXXX",
            " XXXXXX ",
            "  XXXX  ",
            "  XX XX  ",
        ]

    def _get_color(self, enemy: Enemy) -> Tuple[int, int, int]:
        """적 이름/설명 기반으로 색상 생성"""
        prompt = f"Return ONLY an RGB tuple in this exact format: (r,g,b) representing: {enemy.name} - {enemy.description[:30]}"

        attempts = 0
        while attempts < 2:
            try:
                output = self.prompt_llm.generate_completion(prompt=prompt)

                # RGB 튜플 파싱
                import ast

                match = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", output)
                if match:
                    r = min(255, max(0, int(match.group(1))))
                    g = min(255, max(0, int(match.group(2))))
                    b = min(255, max(0, int(match.group(3))))
                    if self.debug:
                        logger.debug(f"Generated color: ({r},{g},{b})")
                    return (r, g, b)

            except Exception:
                attempts += 1
                continue

        # 기본 색상
        logger.warning("Using default color after 2 failed attempts")
        return (200, 100, 50)

    def _get_secondary_color(
        self, base_color: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """보조 색상 생성 (더 밝은 버전)"""
        return (
            min(255, base_color[0] + 60),
            min(255, base_color[1] + 60),
            min(255, base_color[2] + 60),
        )

    def _draw_pixelated_sprite(
        self, enemy: Enemy, pattern: list[str], main_color: Tuple[int, int, int]
    ) -> pygame.Surface:
        """픽셀 아트 스프라이트 그리기"""
        surface = pygame.Surface((self.SPRITE_SIZE, self.SPRITE_SIZE), pygame.SRCALPHA)

        secondary_color = self._get_secondary_color(main_color)
        shadow_color = (main_color[0] // 2, main_color[1] // 2, main_color[2] // 2)

        pixel_w = self.SPRITE_SIZE // self.GRID_SIZE
        pixel_h = self.SPRITE_SIZE // self.GRID_SIZE

        # 패턴 그리기
        for y, row in enumerate(pattern):
            for x, char in enumerate(row):
                if char == "X":
                    # 기본 색상
                    color = main_color

                    # 가장자리는 더 밝게 (하이라이트)
                    if x == 0 or y == 0:
                        color = secondary_color
                    # 오른쪽/아래쪽은 그림자
                    elif x == len(row) - 1 or y == len(pattern) - 1:
                        color = shadow_color

                    px = x * pixel_w
                    py = y * pixel_h
                    pygame.draw.rect(surface, color, (px, py, pixel_w, pixel_h))

        return surface

    def generate_sprite(self, enemy: Enemy) -> pygame.Surface:
        cache_key = _get_cache_key(enemy)
        cached = self._cache.get(cache_key)
        if cached is not None:
            if self.debug:
                logger.debug(f"캐시된 ASCII 스프라이트 사용: {enemy.name}")
            return cached

        logger.info(f"Generating ASCII sprite for: {enemy.name}")

        # 임시 디버깅을 위해 고정 패턴/색상 사용 (LLM 없이)
        import os
        use_dummy = os.environ.get("USE_DUMMY_SPRITE", "false").lower() in ("true", "1")

        if use_dummy:
            logger.warning(f"Using dummy sprite for {enemy.name} (USE_DUMMY_SPRITE=true)")
            pattern = [
                "  XXXX  ",
                " XXXXXX ",
                "XXXXXXXX",
                "XXXXXXXX",
                " XXXXXX ",
                "  XXXX  ",
                "  XX XX  ",
            ]
            main_color = (200, 100, 50)
        else:
            pattern = self._generate_pixel_pattern(enemy)
            main_color = self._get_color(enemy)

        sprite = self._draw_pixelated_sprite(enemy, pattern, main_color)
        self._cache[cache_key] = sprite

        if self.debug:
            logger.debug(f"ASCII 스프라이트 생성 완료: {enemy.name}")

        return sprite

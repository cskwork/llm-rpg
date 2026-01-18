# 2026-01-18

## Changes

### Features
- Zai API (智谱AI) 지원 추가
  - 새로운 `ZaiLLM` 클래스 구현 (`src/llm_rpg/llm/llm.py`)
  - OpenAI SDK 호환 API 지원 (base_url: https://api.z.ai/api/paas/v4/)
  - 지원 모델: glm-4.5-flash (무료), glm-4.7 (유료), glm-4.5-air (유료)
  - 토큰 비용 추적 지원

### Configuration
- `game_config.py` 업데이트
  - Zai 타입 지원 추가
  - `_build_llm()` 메서드에 ZaiLLM 생성 로직 추가
  - `_is_llm_block()` 메서드에 "zai" 타입 추가

- `game_config.yaml` 업데이트
  - action_judge, narrator, enemy_action, enemy_generation, prompt_llm 설정을 Zai API로 변경
  - 기본 모델: glm-4.5-flash (무료 모델)

### Documentation
- `README.md` 업데이트
  - Zai API 사용법 섹션 추가
  - 환경 변수 설정에 ZAI_API_KEY 설명 추가

## Later Changes

### Bug Fixes
- ZaiLLM API 타임아웃 문제 해결
  - OpenAI 클라이언트에 60초 타임아웃 추가 (기본값으로 인터넷 지연 시 응답 대기 중단)
  - 스프라이트 생성용 LLM 모델을 `glm-4.5-flash`로 변경 (무료 + 빠름)
  - Zai API 응답 지연 문제로 발생한 70+초 대기 시간 해결

- ZaiLLM `generate_structured_completion()` 파라미터 이름 수정
  - `output_model` → `output_schema`로 변경하여 LLM 기본 클래스 인터페이스 호환성 확보
  - 적 생성 실패 문제 해결 (`TypeError: got an unexpected keyword argument 'output_schema'`)

- 모든 LLM 구현체 파라미터 이름 `output_schema`로 통일
  - `GroqLLM`: `output_model` → `output_schema` (llm.py:79, 86)
  - `OllamaLLM`: `output_model` → `output_schema` (llm.py:118, 123, 126)
  - `ActionJudge` 호출 사이트: `output_model=` → `output_schema=` (action_judges.py:125)
  - LLM 기본 클래스 인터페이스 표준화로 모든 백엔드(Zai, Groq, Ollama) 호환성 확보

### Infrastructure
- 로깅 시스템 구현
  - 새로운 `src/llm_rpg/utils/logger.py` 모듈 추가
  - 콘솔 및 파일 기반 로깅 지원 (llm_rpg.log, llm_rpg_errors.log)
  - RotatingFileHandler로 로그 파일 크기 관리 (최대 5MB, 백업 3개)

### Logging Updates
- `__main__.py` - 로깅 초기화 추가, 예외 처리 개선
- `enemy_generator.py` - print 문을 logger 호출로 대체
- `sprite_generator.py` - AsciiSpriteGenerator에 로깅 추가
- `battle_start_state.py` - 적 생성 스레드에 로깅 추가
- `llm.py` - ZaiLLM에 디버그 및 에러 로깅 추가

### Features
- Gemini 기반 스프라이트 생성기 추가 (`GeminiSpriteGenerator`)
  - Stable Diffusion 모델 다운로드 없이 Google Gemini 이미지 생성 사용
  - Playwright를 사용한 Gemini 웹 인터페이스 자동화
  - enemy.name 기반 스프라이트 캐싱 지원 (메모리 + 디스크)
  - LLM을 사용한 enemy.description -> 이미지 프롬프트 변환

### Configuration
- `game_config.yaml` 업데이트
  - `sprite_generator.type: "gemini"` 옵션 추가
  - cache_dir, state_dir, headless 설정 추가

### Dependencies
- `pyproject.toml` 업데이트
  - playwright ^1.48.0 의존성 추가

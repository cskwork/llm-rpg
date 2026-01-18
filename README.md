```

▄█        ▄█         ▄▄▄▄███▄▄▄▄           ▄████████    ▄███████▄    ▄██████▄
███       ███       ▄██▀▀▀███▀▀▀██▄        ███    ███   ███    ███   ███    ███
███       ███       ███   ███   ███        ███    ███   ███    ███   ███    █▀
███       ███       ███   ███   ███       ▄███▄▄▄▄██▀   ███    ███  ▄███
███       ███       ███   ███   ███      ▀▀███▀▀▀▀▀   ▀█████████▀  ▀▀███ ████▄
███       ███       ███   ███   ███      ▀███████████   ███          ███    ███
███▌    ▄ ███▌    ▄ ███   ███   ███        ███    ███   ███          ███    ███
█████▄▄██ █████▄▄██  ▀█   ███   █▀         ███    ███  ▄████▀        ████████▀
▀         ▀                                ███    ███
```

LLM-RPG is intended to be a role-playing game that leverages large language models to create dynamic and engaging gameplay experiences. Currently it is still in the early stages of development and only has a battle scene implemented.

## Current / future features

- **Dynamic Battles**: Engage in battles where both heroes and enemies use AI to determine actions and effects.
- **Character Customization**: Define your hero's stats and abilities.
- **AI-Powered Creativity**: Use creative language to influence battle outcomes.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/vossenwout/llm-rpg.git
   cd llm-rpg
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry install
   ```

3. Set up your environment variables. You need to set either `GROQ_API_KEY` or `ZAI_API_KEY` depending on which LLM backend you use. You can do this by creating a `.env` file in the `config` directory:

    ```plaintext
    GROQ_API_KEY=your_api_key_here
    # or
    ZAI_API_KEY=your_api_key_here
    ```
You can get a Groq API key from [here](https://groq.com/). This gives you free tokens each day.
You can get a Zai API key from [here](https://z.ai/model-api). Zai provides free models like glm-4.5-flash.

4. Create `/models/sprite` dir then download and place the following models:
- models/sprite/earthbound_lora.safetensors: [link](https://civitai.com/models/167491)
- models/sprite/westernBeautiful_v10.safetensors [link](https://civitai.com/models/264807?modelVersionId=298593)
- models/sprite/LCM_LoRA_Weights_SD15.safetensors: [link](https://civitai.com/models/195519?modelVersionId=424706)
## Usage

To start the game, run the following command:

```bash
poetry run python -m llm_rpg
```

## Local LLMs with ollama

Using local llms with ollama:

1. Install ollama https://ollama.com

2. Install a model, I would recommend qwen3 models.

3. Start ollama

4. In game_config.yaml, set the models under `action_judge`, `narrator`, and `enemy_action` to the ollama model you installed.

```bash
action_judge:
  backend: "llm"
  llm:
    model: "qwen3:4b"
    type: "ollama"
narrator:
  llm:
    model: "qwen3:4b"
    type: "ollama"
enemy_action:
  llm:
    model: "qwen3:4b"
    type: "ollama"
```

5. Run the game

```bash
poetry run python -m llm_rpg
```

## Using Zai API

Using Zai (智谱AI) API:

1. Get API key from https://z.ai/model-api

2. Set `ZAI_API_KEY` environment variable in `config/.env`:

    ```plaintext
    ZAI_API_KEY=your_api_key_here
    ```

3. In game_config.yaml, set models under `action_judge`, `narrator`, and `enemy_action` to Zai model:

    ```yaml
    action_judge:
      backend: "llm"
      llm:
        model: "glm-4.5-flash"
        type: "zai"
    narrator:
      llm:
        model: "glm-4.5-flash"
        type: "zai"
    enemy_action:
      llm:
        model: "glm-4.5-flash"
        type: "zai"
    ```

4. Run the game

    ```bash
    poetry run python -m llm_rpg
    ```

Available Zai models:
- `glm-4.5-flash` (FREE) - Recommended for testing
- `glm-4.7` - Flagship model (paid)
- `glm-4.5-air` - Cost-optimized (paid)

## Maintaining the codebase

Install pre-commit hooks:

```bash
pre-commit install
```

Run tests:

```bash
poetry run pytest -s -v
```

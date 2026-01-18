from dotenv import load_dotenv

from llm_rpg.game.game import Game
from llm_rpg.game.game_config import GameConfig
from llm_rpg.utils.logger import setup_logging, get_logger

env_files = [
    ".env",
    "config/.env.secret",
]

for env_file in env_files:
    load_dotenv(env_file)


if __name__ == "__main__":
    logger = setup_logging(log_level="DEBUG", enable_console=True, enable_file=True)
    logger.info("LLM-RPG starting...")

    try:
        game = Game(config=GameConfig("config/game_config.yaml"))
        game.run()
    except Exception as exc:
        logger.exception("Fatal error during game execution")
        raise

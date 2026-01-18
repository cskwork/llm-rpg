from __future__ import annotations

from dataclasses import dataclass
import json
import random
import traceback
from typing import Annotated

from pydantic import BaseModel, Field

from llm_rpg.llm.llm import LLM
import pygame

from llm_rpg.sprite_generator.sprite_generator import (
    SpriteGenerator,
)
from llm_rpg.systems.battle.enemy import Enemy, EnemyArchetypes
from llm_rpg.systems.battle.enemy_action_generators import EnemyActionGenerator
from llm_rpg.objects.character import Stats
from llm_rpg.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EnemyDescription:
    name: str
    description: str


class LLMEnemyDescriptionOutput(BaseModel):
    name: Annotated[
        str,
        Field(
            min_length=2,
            max_length=40,
            description="Short enemy name (one or two words).",
        ),
    ]
    description: Annotated[
        str,
        Field(
            description="One or two sentences describing the enemy.",
        ),
    ]


class EnemyGenerator:
    def __init__(
        self,
        llm: LLM,
        prompt: str,
        enemy_action_generator: EnemyActionGenerator,
        base_stats: Stats,
        sprite_generator: SpriteGenerator,
        characters: list[str],
        adjectives: list[str],
        places: list[str],
        debug: bool = False,
    ):
        self.llm = llm
        self.prompt = prompt
        self.enemy_action_generator = enemy_action_generator
        self.sprite_generator = sprite_generator
        self.base_stats = base_stats
        self.characters = characters
        self.adjectives = adjectives
        self.places = places
        self.debug = debug

    def generate_enemy(self) -> tuple[Enemy, pygame.Surface]:
        logger.info("Starting enemy generation...")
        enemy_description = self._generate_enemy_description()
        logger.info(f"Got description - name={enemy_description.name}")
        archetype = random.choice(list(EnemyArchetypes))
        logger.debug(f"Creating Enemy with archetype={archetype}")
        enemy = Enemy(
            name=enemy_description.name,
            description=enemy_description.description,
            level=1,
            base_stats=self.base_stats,
            archetype=archetype,
            enemy_action_generator=self.enemy_action_generator,
        )
        logger.debug("Enemy created, generating sprite...")
        
        try:
            sprite = self.sprite_generator.generate_sprite(enemy)
            logger.info("Sprite generated successfully!")
        except Exception as e:
            logger.error(f"Sprite generation failed for {enemy_description.name}: {e}", exc_info=True)
            logger.warning("Using default placeholder sprite due to generation error")
            # Create fallback sprite
            import pygame
            sprite = pygame.Surface((64, 64), pygame.SRCALPHA)
            sprite.fill((128, 128, 128))
        
        return enemy, sprite

    def _pick_word(self, words: list[str], label: str) -> str:
        if not words:
            raise ValueError(
                f"Enemy generation {label} words must be configured for enemy generation"
            )
        return random.choice(words)

    def _get_prompt(self) -> str:
        enemy_character = self._pick_word(self.characters, "character")
        enemy_adjective = self._pick_word(self.adjectives, "adjective")
        enemy_place = self._pick_word(self.places, "place")
        prompt = self.prompt.format(
            enemy_character=enemy_character,
            enemy_adjective=enemy_adjective,
            enemy_place=enemy_place,
        )
        schema = json.dumps(LLMEnemyDescriptionOutput.model_json_schema(), indent=2)
        return f"{prompt}{schema}"

    def _generate_enemy_description(self) -> EnemyDescription:
        attempts = 0
        while attempts < 3:
            prompt = self._get_prompt()
            if self.debug:
                logger.debug(f"EnemyGeneration prompt: {prompt[:500]}...")
            try:
                output = self.llm.generate_structured_completion(
                    prompt=prompt, output_schema=LLMEnemyDescriptionOutput
                )
                logger.debug(f"Parsed output - name='{output.name}', description='{output.description[:50]}...'")
                return EnemyDescription(
                    name=output.name.strip(),
                    description=output.description.strip(),
                )
            except Exception as exc:
                logger.error(f"Attempt {attempts + 1} failed: {type(exc).__name__}: {exc}", exc_info=True)
                attempts += 1
        raise ValueError("Failed to generate enemy description")

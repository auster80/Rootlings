"""
Rootlings Prototype
===================
How to run: pip install pygame; python rootlings_prototype.py
Controls: Left click to select a Rootling. Right click to cancel a blocker or clear selection.
Hotkeys: 1 = Dig, 2 = Bridge, 3 = Block, R = Restart, ESC = Quit, F1 = Toggle debug overlay.
Win condition: Save at least the required number of Rootlings before the timer expires.
Lose condition: Timer expires without enough saves or all Rootlings are gone.
Known limitations: Simplified physics and AI, single hardcoded level, minimal audio/visual feedback.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import pygame


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
TILE = 32
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60
FIXED_DT = 1.0 / FPS
GRAVITY = 1500.0
TERMINAL_VEL = 900.0
WALK_SPEED = 90.0
FALL_EPSILON = 2.0
STEP_HEIGHT_PIXELS = TILE
FATAL_FALL_TILES = 7
SPAWN_INTERVAL = 0.5
TOTAL_TO_SPAWN = 16
REQUIRED_TO_SAVE = 12
TIME_LIMIT = 180.0
BRIDGE_STEPS_MAX = 12
BRIDGE_BUILD_TURNS = 90  # simulation ticks (60 per second) per bridge segment
DIG_INTERVAL = 1 #0.15
ROOTLING_WIDTH = 14
ROOTLING_HEIGHT = 22
ROOTLING_HITBOX = pygame.Rect(0, 0, ROOTLING_WIDTH, ROOTLING_HEIGHT)
SELECTION_RADIUS = 24
HAZARD_KILL_DELAY = 0.05

MAX_DIGGERS = 4
MAX_BRIDGERS = 6
MAX_BLOCKERS = 2
MAX_BOMBERS = 2
BOMBER_COUNTDOWN = 5.0

COLOR_BG_TOP = (50, 65, 90)
COLOR_BG_BOTTOM = (15, 20, 30)
COLOR_GROUND = (115, 80, 50)
COLOR_ROCK = (70, 55, 40)
COLOR_HAZARD_BASE = (200, 45, 50)
COLOR_BRIDGE = (190, 170, 120)
COLOR_SPAWN = (120, 200, 120)
COLOR_EXIT = (200, 220, 120)
COLOR_ROOTLING = (230, 220, 205)
COLOR_ROOTLING_OUTLINE = (45, 30, 20)
COLOR_SELECTION = (230, 240, 80)
COLOR_HUD_BG = (10, 10, 10, 180)
COLOR_HUD_TEXT = (230, 230, 230)
COLOR_HUD_WARNING = (240, 120, 120)
COLOR_DEBUG = (120, 200, 240)

LEVEL_LAYOUT = """
########################################
#......................................#
#..............@@@@@@..................#
#..............@....@..................#
#..............@....@..................#
#......S.......@....@.................E#
#..............@....@..................#
#..............@....@..................#
#..............@....@..................#
#..............@....@..................#
#..............@....@..................#
#..............@....@..................#
#.......#####..@....@..................#
#........~~~...@....@..................#
#........~~~...@....@..................#
#........~~~...@....@..................#
#..............@....@..................#
#.....#########@....@..................#
#..............@....@..................#
#..................@@@.................#
#......................................#
########################################
""".strip("\n")


# --------------------------------------------------------------------------------------
# Helper Enums
# --------------------------------------------------------------------------------------
class RootlingState(Enum):
    """Finite states for Rootling behaviour."""

    WALK = auto()
    FALL = auto()
    DIG = auto()
    BRIDGE = auto()
    BLOCK = auto()
    EXITED = auto()
    DEAD = auto()


class GameState(Enum):
    """High level game states."""

    RUNNING = auto()
    WIN = auto()
    LOSE = auto()


# --------------------------------------------------------------------------------------
# Level and environment classes
# --------------------------------------------------------------------------------------
class Level:
    """Stores tile grid data and provides collision helpers."""

    def __init__(self, ascii_map: str) -> None:
        rows = ascii_map.splitlines()
        self.height = len(rows)
        self.width = len(rows[0]) if rows else 0
        self.tiles: List[List[str]] = [list(row) for row in rows]
        self.bridge_tiles: Dict[Tuple[int, int], int] = {}
        self.spawn_pos = self._find_tile('S')
        self.exit_pos = self._find_tile('E')
        if not self.spawn_pos:
            raise ValueError("Level missing spawn point 'S'")
        if not self.exit_pos:
            raise ValueError("Level missing exit point 'E'")
        self.spawn_rect = pygame.Rect(
            self.spawn_pos[0] * TILE,
            self.spawn_pos[1] * TILE,
            TILE,
            TILE,
        )
        self.exit_rect = pygame.Rect(
            self.exit_pos[0] * TILE,
            self.exit_pos[1] * TILE,
            TILE,
            TILE,
        )

    def _find_tile(self, char: str) -> Optional[Tuple[int, int]]:
        for y, row in enumerate(self.tiles):
            for x, cell in enumerate(row):
                if cell == char:
                    return (x, y)
        return None

    def reset_dynamic(self) -> None:
        """Clear dynamic tiles such as bridges."""
        self.bridge_tiles.clear()

    def get_tile(self, tx: int, ty: int) -> str:
        if 0 <= tx < self.width and 0 <= ty < self.height:
            return self.tiles[ty][tx]
        return '#'

    def set_tile(self, tx: int, ty: int, char: str) -> None:
        if 0 <= tx < self.width and 0 <= ty < self.height:
            self.tiles[ty][tx] = char

    def is_solid(self, tx: int, ty: int) -> bool:
        tile = self.get_tile(tx, ty)
        if tile in ('#', '@'):
            return True
        if (tx, ty) in self.bridge_tiles:
            return True
        return tile == 'S'  # treat spawn tile as ground

    def is_hazard(self, tx: int, ty: int) -> bool:
        return self.get_tile(tx, ty) == '~'

    def is_diggable(self, tx: int, ty: int) -> bool:
        tile = self.get_tile(tx, ty)
        return tile == '#'

    def is_exit(self, tx: int, ty: int) -> bool:
        return self.get_tile(tx, ty) == 'E'

    def dig_tile(self, tx: int, ty: int) -> bool:
        """Remove a diggable tile, returning True if removed."""
        if self.is_diggable(tx, ty):
            self.set_tile(tx, ty, '.')
            return True
        return False

    def add_bridge_tile(self, tx: int, ty: int, direction: int) -> bool:
        """Add a temporary bridge tile if the space is empty."""
        if self.is_solid(tx, ty) or self.is_hazard(tx, ty):
            return False
        if self.get_tile(tx, ty) != '.':
            return False
        self.bridge_tiles[(tx, ty)] = direction
        return True

    def remove_bridge_tile(self, tx: int, ty: int) -> None:
        self.bridge_tiles.pop((tx, ty), None)

    def rect_collides_solid(self, rect: pygame.Rect, direction: Optional[int] = None) -> bool:
        min_tx = rect.left // TILE
        max_tx = rect.right // TILE
        min_ty = rect.top // TILE
        max_ty = rect.bottom // TILE
        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                if self.is_solid(tx, ty):
                    if direction is not None and (tx, ty) in self.bridge_tiles:
                        bridge_dir = self.bridge_tiles[(tx, ty)]
                        if bridge_dir != direction:
                            continue
                    tile_rect = pygame.Rect(tx * TILE, ty * TILE, TILE, TILE)
                    if rect.colliderect(tile_rect):
                        return True
        return False

    def rect_overlaps_hazard(self, rect: pygame.Rect) -> bool:
        min_tx = rect.left // TILE
        max_tx = rect.right // TILE
        min_ty = rect.top // TILE
        max_ty = rect.bottom // TILE
        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                if self.is_hazard(tx, ty):
                    tile_rect = pygame.Rect(tx * TILE, ty * TILE, TILE, TILE)
                    if rect.colliderect(tile_rect):
                        return True
        return False

    def world_to_tile(self, x: float, y: float) -> Tuple[int, int]:
        return int(x // TILE), int(y // TILE)


@dataclass
class Spawner:
    """Handles timed emission of Rootlings."""

    level: Level
    total: int = TOTAL_TO_SPAWN
    interval: float = SPAWN_INTERVAL
    timer: float = 0.0
    spawned: int = 0

    def spawn_position(self) -> Tuple[float, float]:
        center_x = self.level.spawn_rect.centerx - ROOTLING_WIDTH / 2
        y = self.level.spawn_rect.bottom - ROOTLING_HEIGHT
        return float(center_x), float(y)

    def update(self, dt: float, rootlings: List['Rootling']) -> None:
        if self.spawned >= self.total:
            return
        self.timer += dt
        if self.timer >= self.interval:
            self.timer -= self.interval
            spawn_pos = self.spawn_position()
            rootlings.append(Rootling(spawn_pos[0], spawn_pos[1]))
            self.spawned += 1


@dataclass
class ExitZone:
    """Defines the exit area for Rootlings."""

    rect: pygame.Rect


# --------------------------------------------------------------------------------------
# Rootling behaviour
# --------------------------------------------------------------------------------------
class Rootling:
    """Rootling agent handling movement and ability logic."""

    def __init__(self, x: float, y: float) -> None:
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(WALK_SPEED, 0.0)
        self.state = RootlingState.WALK
        self.direction = 1
        self.fall_start_y = self.pos.y
        self.fall_time = 0.0
        self.dig_timer = 0.0
        self.bridge_progress = 0.0
        self.bridge_steps = 0
        self.bridge_anchor: Optional[Tuple[int, int]] = None
        self.selected = False
        self.dead_timer = 0.0
        self.on_ground = False
        self.bomber_timer: Optional[float] = None

    # ----------------------------------------------------------------------------------
    # Utility properties
    # ----------------------------------------------------------------------------------
    @property
    def rect(self) -> pygame.Rect:
        rect = ROOTLING_HITBOX.copy()
        rect.topleft = (int(self.pos.x), int(self.pos.y))
        return rect

    def set_state(self, state: RootlingState) -> None:
        if self.state != state:
            if state == RootlingState.FALL:
                self.fall_start_y = self.rect.bottom
                self.fall_time = 0.0
            if state != RootlingState.BRIDGE:
                self.bridge_steps = 0
                self.bridge_anchor = None
                self.bridge_progress = 0.0
            if state != RootlingState.DIG:
                self.dig_timer = 0.0
            self.state = state

    # ----------------------------------------------------------------------------------
    # Ability assignments
    # ----------------------------------------------------------------------------------
    def assign_dig(self) -> bool:
        if self.state in (RootlingState.WALK, RootlingState.FALL):
            self.set_state(RootlingState.DIG)
            self.dig_timer = 0.0
            return True
        return False

    def assign_bridge(self) -> bool:
        if self.state == RootlingState.WALK and self.on_ground:
            self.set_state(RootlingState.BRIDGE)
            self.bridge_steps = 0
            self.bridge_anchor = None
            self.bridge_progress = 0.0
            return True
        return False

    def assign_block(self) -> bool:
        if self.state == RootlingState.WALK and self.on_ground:
            self.set_state(RootlingState.BLOCK)
            self.vel.xy = 0, 0
            return True
        return False

    def assign_bomber(self) -> bool:
        if self.state in (RootlingState.EXITED, RootlingState.DEAD):
            return False
        if self.bomber_timer is not None:
            return False
        self.bomber_timer = BOMBER_COUNTDOWN
        return True

    # ----------------------------------------------------------------------------------
    # Update logic
    # ----------------------------------------------------------------------------------
    def update(self, dt: float, level: Level, others: List['Rootling']) -> None:
        if self.state in (RootlingState.EXITED, RootlingState.DEAD):
            if self.state == RootlingState.DEAD:
                self.dead_timer += dt
            return

        if level.rect_overlaps_hazard(self.rect):
            self.die()
            return

        if self.bomber_timer is not None:
            self.bomber_timer -= dt
            if self.bomber_timer <= 0:
                self.explode(level)
                return

        if self.state == RootlingState.BLOCK:
            self.vel.xy = 0, 0
            self.on_ground = self.check_ground(level)
            return

        if self.state == RootlingState.DIG:
            self.update_dig(dt, level)
            return

        if self.state == RootlingState.BRIDGE:
            self.update_bridge(dt, level)
            return

        if self.state == RootlingState.FALL:
            self.update_fall(dt, level)
        elif self.state == RootlingState.WALK:
            self.update_walk(dt, level, others)

    def update_walk(self, dt: float, level: Level, others: List['Rootling']) -> None:
        self.on_ground = self.check_ground(level)
        if not self.on_ground:
            self.set_state(RootlingState.FALL)
            self.update_fall(dt, level)
            return

        if self.wall_ahead(level):
            self.direction *= -1

        if self.check_blockers_ahead(others):
            self.direction *= -1

        self.vel.x = WALK_SPEED * self.direction
        self.vel.y = 0

        self.apply_horizontal_movement(level, self.vel.x * dt)

        if not self.check_ground(level):
            self.set_state(RootlingState.FALL)

    def update_fall(self, dt: float, level: Level) -> None:
        self.vel.y = min(self.vel.y + GRAVITY * dt, TERMINAL_VEL)
        dy = self.vel.y * dt
        self.apply_vertical_movement(level, dy)
        if self.on_ground:
            fall_distance = max(0, (self.fall_start_y - self.rect.bottom) / TILE)
            if fall_distance > FATAL_FALL_TILES:
                self.die()
                return
            self.set_state(RootlingState.WALK)
        else:
            self.fall_time += dt

    def update_dig(self, dt: float, level: Level) -> None:
        below_tile = level.world_to_tile(self.rect.centerx, self.rect.bottom + 1)
        if not level.is_diggable(*below_tile):
            self.set_state(RootlingState.FALL)
            return
        self.dig_timer += dt
        if self.dig_timer >= DIG_INTERVAL:
            self.dig_timer -= DIG_INTERVAL
            if level.dig_tile(*below_tile):
                self.pos.y += 1
            else:
                self.set_state(RootlingState.FALL)

        self.on_ground = False
        self.vel.xy = 0, 0

    def update_bridge(self, dt: float, level: Level) -> None:
        if self.bridge_anchor is None:
            anchor_tx, anchor_ty = level.world_to_tile(self.rect.centerx, self.rect.bottom - 1)
            self.bridge_anchor = (anchor_tx, anchor_ty)
            self.bridge_progress = 0.0

        if self.bridge_steps >= BRIDGE_STEPS_MAX:
            self.set_state(RootlingState.WALK)
            return

        self.vel.xy = 0, 0
        self.on_ground = True
        self.bridge_progress += dt / FIXED_DT
        if self.bridge_progress < BRIDGE_BUILD_TURNS:
            return
        self.bridge_progress -= BRIDGE_BUILD_TURNS

        step_index = self.bridge_steps
        target_tx = self.bridge_anchor[0] + self.direction * (step_index + 1)
        target_ty = self.bridge_anchor[1] - step_index
        if not level.add_bridge_tile(target_tx, target_ty, self.direction):
            self.set_state(RootlingState.WALK)
            return

        self.bridge_steps += 1
        tile_left = target_tx * TILE
        self.pos.x = tile_left + (TILE - ROOTLING_WIDTH) / 2
        self.pos.y = target_ty * TILE - ROOTLING_HEIGHT
        self.on_ground = True

        if not level.rect_collides_solid(self.rect.move(0, 1)):
            self.set_state(RootlingState.FALL)

    def apply_horizontal_movement(self, level: Level, dx: float) -> None:
        if abs(dx) < 1e-4:
            return
        step_direction = 1 if dx > 0 else -1
        step_amount = abs(dx)
        remaining = step_amount
        while remaining > 0:
            step = min(remaining, 4)
            offset = step * step_direction
            moved = self.try_move_horizontal(level, offset)
            if not moved:
                self.direction *= -1
                self.vel.x = WALK_SPEED * self.direction
                break
            remaining -= step

    def try_move_horizontal(self, level: Level, dx: float) -> bool:
        if dx == 0:
            return True
        rect = self.rect
        target_rect = rect.move(dx, 0)
        if not self.on_ground:
            if level.rect_collides_solid(target_rect):
                return False
            self.pos.x += dx
            return True

        max_step = STEP_HEIGHT_PIXELS
        for step_up in range(0, max_step + 1, 4):
            test_rect = target_rect.move(0, -step_up)
            if not level.rect_collides_solid(test_rect, self.direction):
                self.pos.x += dx
                self.pos.y -= step_up
                self.on_ground = True
                return True
        return False

    def apply_vertical_movement(self, level: Level, dy: float) -> None:
        if abs(dy) < 1e-4:
            self.on_ground = self.check_ground(level)
            return
        step_direction = 1 if dy > 0 else -1
        step_amount = abs(dy)
        remaining = step_amount
        self.on_ground = False
        while remaining > 0:
            step = min(remaining, 4)
            offset = step * step_direction
            new_rect = self.rect.move(0, offset)
            if level.rect_collides_solid(new_rect):
                if step_direction > 0:
                    self.pos.y = (new_rect.bottom // TILE) * TILE - ROOTLING_HEIGHT
                    self.on_ground = True
                    self.vel.y = 0
                else:
                    self.pos.y = ((new_rect.top // TILE) + 1) * TILE
                    self.vel.y = 0
                return
            else:
                self.pos.y += offset
            remaining -= step

    def check_ground(self, level: Level) -> bool:
        rect = self.rect.move(0, 1)
        return level.rect_collides_solid(rect)

    def wall_ahead(self, level: Level) -> bool:
        offset = 1 if self.direction > 0 else -1
        probe_rect = self.rect.move(offset, 0)
        if not self.on_ground:
            return level.rect_collides_solid(probe_rect)
        for step_up in range(0, STEP_HEIGHT_PIXELS + 1, 4):
            test_rect = probe_rect.move(0, -step_up)
            if not level.rect_collides_solid(test_rect, self.direction):
                return False
        return True

    def check_blockers_ahead(self, others: List['Rootling']) -> bool:
        front_rect = self.rect.inflate(6, 6)
        front_rect.x += self.direction * 6
        for other in others:
            if other is self:
                continue
            if other.state == RootlingState.BLOCK and front_rect.colliderect(other.rect):
                return True
        return False

    def die(self) -> None:
        if self.state != RootlingState.DEAD:
            self.state = RootlingState.DEAD
            self.dead_timer = 0.0
            self.bomber_timer = None

    def explode(self, level: Level) -> None:
        center_tx, center_ty = level.world_to_tile(self.rect.centerx, self.rect.centery)
        radius = 1
        for ty in range(center_ty - radius, center_ty + radius + 1):
            for tx in range(center_tx - radius, center_tx + radius + 1):
                if abs(tx - center_tx) <= radius and abs(ty - center_ty) <= radius:
                    if level.is_diggable(tx, ty):
                        level.set_tile(tx, ty, '.')
                    level.remove_bridge_tile(tx, ty)
        self.bomber_timer = None
        self.die()


# --------------------------------------------------------------------------------------
# HUD and rendering helpers
# --------------------------------------------------------------------------------------
class HUD:
    """Renders textual information and overlays."""

    def __init__(self, screen: pygame.Surface) -> None:
        self.font_small = pygame.font.Font(None, 22)
        self.font_large = pygame.font.Font(None, 30)
        self.screen = screen

    def draw(self, game: 'Game') -> None:
        hud_rect = pygame.Rect(0, 0, SCREEN_WIDTH, 40)
        hud_surface = pygame.Surface(hud_rect.size, pygame.SRCALPHA)
        hud_surface.fill(COLOR_HUD_BG)

        text_lines = [
            f"Spawned {game.spawner.spawned}/{game.spawner.total} | Saved {game.saved} | Dead {game.dead}",
            (
                f"Dig {game.abilities['dig']}/{MAX_DIGGERS} | "
                f"Bridge {game.abilities['bridge']}/{MAX_BRIDGERS} | "
                f"Block {game.abilities['block']}/{MAX_BLOCKERS} | "
                f"Bomber {game.abilities['bomber']}/{MAX_BOMBERS}"
            ),
            f"Time Left: {int(max(0, game.time_left))}s"
        ]
        x = 12
        y = 6
        for line in text_lines:
            text = self.font_large.render(line, True, COLOR_HUD_TEXT)
            hud_surface.blit(text, (x, y))
            y += 14

        hint = "1:Dig 2:Bridge 3:Block 4:Bomber | R:Restart | ESC:Quit"
        text = self.font_small.render(hint, True, COLOR_HUD_TEXT)
        hud_surface.blit(text, (SCREEN_WIDTH - text.get_width() - 12, 12))

        self.screen.blit(hud_surface, (0, 0))

    def draw_end_screen(self, game: 'Game', message: str) -> None:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        title = self.font_large.render(message, True, COLOR_HUD_TEXT)
        prompt = self.font_small.render("Press R to restart", True, COLOR_HUD_TEXT)
        self.screen.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20)))
        self.screen.blit(prompt, prompt.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 10)))

    def draw_debug(self, game: 'Game', fps: float) -> None:
        lines = [
            f"FPS: {fps:.1f}",
            f"Rootlings: {len(game.rootlings)}",
        ]
        if game.selected:
            fall_distance = max(0.0, (game.selected.fall_start_y - game.selected.rect.bottom) / TILE)
            lines.append(f"Selected State: {game.selected.state.name}")
            lines.append(f"Fall distance: {fall_distance:.2f} tiles")
        mouse_pos = pygame.mouse.get_pos()
        tile = game.level.world_to_tile(*mouse_pos)
        lines.append(f"Cursor tile: {tile}")

        for i, line in enumerate(lines):
            text = self.font_small.render(line, True, COLOR_DEBUG)
            self.screen.blit(text, (10, 50 + i * 16))


# --------------------------------------------------------------------------------------
# Main Game class
# --------------------------------------------------------------------------------------
class Game:
    """Main game loop and state management."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Rootlings Prototype")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.level = Level(LEVEL_LAYOUT)
        self.spawner = Spawner(self.level)
        self.exit_zone = ExitZone(self.level.exit_rect)
        self.rootlings: List[Rootling] = []
        self.hud = HUD(self.screen)
        self.game_state = GameState.RUNNING
        self.abilities = {
            'dig': MAX_DIGGERS,
            'bridge': MAX_BRIDGERS,
            'block': MAX_BLOCKERS,
            'bomber': MAX_BOMBERS,
        }
        self.saved = 0
        self.dead = 0
        self.time_left = TIME_LIMIT
        self.accumulator = 0.0
        self.running = True
        self.debug_overlay = False
        self.selected: Optional[Rootling] = None

    # ----------------------------------------------------------------------------------
    def reset(self) -> None:
        self.level = Level(LEVEL_LAYOUT)
        self.level.reset_dynamic()
        self.spawner = Spawner(self.level)
        self.exit_zone = ExitZone(self.level.exit_rect)
        self.rootlings.clear()
        self.abilities = {
            'dig': MAX_DIGGERS,
            'bridge': MAX_BRIDGERS,
            'block': MAX_BLOCKERS,
            'bomber': MAX_BOMBERS,
        }
        self.saved = 0
        self.dead = 0
        self.time_left = TIME_LIMIT
        self.game_state = GameState.RUNNING
        self.selected = None
        self.accumulator = 0.0

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.accumulator += dt
            self.handle_events()
            while self.accumulator >= FIXED_DT:
                self.update(FIXED_DT)
                self.accumulator -= FIXED_DT
            self.draw()
        pygame.quit()
        sys.exit()

    # ----------------------------------------------------------------------------------
    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_F1:
                    self.debug_overlay = not self.debug_overlay
                elif self.game_state == GameState.RUNNING:
                    if event.key == pygame.K_1:
                        self.try_assign('dig')
                    elif event.key == pygame.K_2:
                        self.try_assign('bridge')
                    elif event.key == pygame.K_3:
                        self.try_assign('block')
                    elif event.key == pygame.K_4:
                        self.try_assign('bomber')
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.select_rootling(event.pos)
                elif event.button == 3:
                    self.handle_right_click()

    def try_assign(self, ability: str) -> None:
        if not self.selected or self.game_state != GameState.RUNNING:
            return
        if self.abilities[ability] <= 0:
            return
        assigned = False
        if ability == 'dig':
            assigned = self.selected.assign_dig()
        elif ability == 'bridge':
            assigned = self.selected.assign_bridge()
        elif ability == 'block':
            assigned = self.selected.assign_block()
        elif ability == 'bomber':
            assigned = self.selected.assign_bomber()
        if assigned:
            self.abilities[ability] -= 1

    def select_rootling(self, pos: Tuple[int, int]) -> None:
        best: Optional[Rootling] = None
        best_dist = SELECTION_RADIUS
        for rootling in self.rootlings:
            if rootling.state in (RootlingState.EXITED, RootlingState.DEAD):
                continue
            rect = rootling.rect.inflate(10, 10)
            if rect.collidepoint(pos):
                dist = pygame.Vector2(rootling.rect.center).distance_to(pos)
                if dist < best_dist:
                    best = rootling
                    best_dist = dist
        if self.selected:
            self.selected.selected = False
        self.selected = best
        if self.selected:
            self.selected.selected = True

    def handle_right_click(self) -> None:
        if self.selected and self.selected.state == RootlingState.BLOCK:
            self.selected.set_state(RootlingState.WALK)
            self.selected.selected = False
            self.selected = None
        else:
            if self.selected:
                self.selected.selected = False
            self.selected = None

    # ----------------------------------------------------------------------------------
    def update(self, dt: float) -> None:
        if self.game_state != GameState.RUNNING:
            return

        self.time_left -= dt
        if self.time_left <= 0:
            self.time_left = 0
            if self.saved < REQUIRED_TO_SAVE:
                self.game_state = GameState.LOSE
                return

        self.spawner.update(dt, self.rootlings)

        for rootling in list(self.rootlings):
            prev_state = rootling.state
            rootling.update(dt, self.level, self.rootlings)
            if rootling.state == RootlingState.DEAD and prev_state != RootlingState.DEAD:
                self.dead += 1
            if rootling.state == RootlingState.EXITED:
                continue
            if rootling.state != RootlingState.DEAD and rootling.rect.colliderect(self.exit_zone.rect):
                rootling.set_state(RootlingState.EXITED)
                self.saved += 1

        active = [r for r in self.rootlings if r.state not in (RootlingState.EXITED, RootlingState.DEAD)]
        if self.saved >= REQUIRED_TO_SAVE:
            self.game_state = GameState.WIN
        elif self.time_left <= 0 and self.saved < REQUIRED_TO_SAVE:
            self.game_state = GameState.LOSE
        elif self.spawner.spawned >= self.spawner.total and not active and self.saved < REQUIRED_TO_SAVE:
            self.game_state = GameState.LOSE

    # ----------------------------------------------------------------------------------
    def draw(self) -> None:
        self.draw_background()
        self.draw_level()
        self.draw_rootlings()
        self.hud.draw(self)

        if self.game_state == GameState.WIN:
            self.hud.draw_end_screen(self, "Level Complete!")
        elif self.game_state == GameState.LOSE:
            self.hud.draw_end_screen(self, "Level Failed")

        if self.debug_overlay:
            fps = self.clock.get_fps()
            self.hud.draw_debug(self, fps)

        pygame.display.flip()

    def draw_background(self) -> None:
        for y in range(SCREEN_HEIGHT):
            t = y / SCREEN_HEIGHT
            r = int(COLOR_BG_TOP[0] * (1 - t) + COLOR_BG_BOTTOM[0] * t)
            g = int(COLOR_BG_TOP[1] * (1 - t) + COLOR_BG_BOTTOM[1] * t)
            b = int(COLOR_BG_TOP[2] * (1 - t) + COLOR_BG_BOTTOM[2] * t)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))

    def draw_level(self) -> None:
        hazard_phase = pygame.time.get_ticks() / 300.0
        for y in range(self.level.height):
            for x in range(self.level.width):
                tile = self.level.get_tile(x, y)
                rect = pygame.Rect(x * TILE, y * TILE, TILE, TILE)
                if tile == '#':
                    pygame.draw.rect(self.screen, COLOR_GROUND, rect)
                elif tile == '@':
                    pygame.draw.rect(self.screen, COLOR_ROCK, rect)
                elif tile == '~':
                    offset = math.sin(hazard_phase + x * 0.5) * 4
                    hazard_rect = rect.inflate(0, -8).move(0, offset)
                    pygame.draw.rect(self.screen, COLOR_HAZARD_BASE, hazard_rect)
                elif tile == 'S':
                    pygame.draw.rect(self.screen, COLOR_SPAWN, rect)
                elif tile == 'E':
                    pygame.draw.rect(self.screen, COLOR_EXIT, rect)
        for (tx, ty), direction in self.level.bridge_tiles.items():
            rect = pygame.Rect(tx * TILE, ty * TILE, TILE, TILE)
            pygame.draw.rect(self.screen, COLOR_BRIDGE, rect)
            if direction > 0:
                pygame.draw.line(self.screen, (120, 110, 80), rect.bottomleft, rect.topright, 3)
            else:
                pygame.draw.line(self.screen, (120, 110, 80), rect.topleft, rect.bottomright, 3)

    def draw_rootlings(self) -> None:
        for rootling in self.rootlings:
            rect = rootling.rect
            if rootling.state == RootlingState.DEAD:
                alpha = max(0, 255 - int(rootling.dead_timer * 255))
                surface = pygame.Surface(rect.size, pygame.SRCALPHA)
                surface.fill((*COLOR_HAZARD_BASE, alpha))
                self.screen.blit(surface, rect)
                continue
            color = COLOR_ROOTLING
            pygame.draw.rect(self.screen, COLOR_ROOTLING_OUTLINE, rect.inflate(4, 4), border_radius=6)
            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            eye_offset = 4
            eye_y = rect.centery - 4
            eye_x = rect.centerx + rootling.direction * 4
            pygame.draw.circle(self.screen, (40, 40, 40), (eye_x, eye_y), 2)
            if rootling.state == RootlingState.BLOCK:
                block_rect = rect.inflate(10, 4)
                pygame.draw.rect(self.screen, COLOR_SELECTION, block_rect, width=2, border_radius=4)
            if rootling.selected:
                pygame.draw.rect(self.screen, COLOR_SELECTION, rect.inflate(8, 8), width=2, border_radius=8)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def main() -> None:
    game = Game()
    game.run()


if __name__ == "__main__":
    main()

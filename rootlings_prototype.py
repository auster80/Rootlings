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
import os
import random
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
COLOR_PARTICLE_DIRT = (170, 130, 90, 255)
COLOR_PARTICLE_SPARK = (208, 190, 130, 255)

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
# Animation helpers and asset generation
# --------------------------------------------------------------------------------------
class Animation:
    """Simple time-based animation sequence."""

    def __init__(self, frames: List[pygame.Surface], fps: float, loop: bool = True) -> None:
        self.frames = frames
        self.fps = fps
        self.loop = loop
        self.time = 0.0
        self.index = 0
        self.finished = False

    def reset(self) -> None:
        self.time = 0.0
        self.index = 0
        self.finished = False

    def update(self, dt: float) -> None:
        if self.finished or len(self.frames) <= 1 or self.fps <= 0:
            return
        step = 1.0 / self.fps
        self.time += dt
        while self.time >= step:
            self.time -= step
            self.index += 1
            if self.index >= len(self.frames):
                if self.loop:
                    self.index = 0
                else:
                    self.index = len(self.frames) - 1
                    self.finished = True

    @property
    def frame(self) -> pygame.Surface:
        return self.frames[self.index] if self.frames else pygame.Surface((1, 1), pygame.SRCALPHA)


def load_sheet(path: str, frame_w: int, frame_h: int) -> List[pygame.Surface]:
    frames: List[pygame.Surface] = []
    try:
        sheet = pygame.image.load(path).convert_alpha()
    except Exception:
        return frames
    sw, sh = sheet.get_size()
    for y in range(0, sh, frame_h):
        for x in range(0, sw, frame_w):
            frames.append(sheet.subsurface(pygame.Rect(x, y, frame_w, frame_h)).copy())
    return frames


def ensure_assets() -> None:
    """Create placeholder sprite sheets on first run."""

    os.makedirs("assets/rootling", exist_ok=True)
    os.makedirs("assets/fx", exist_ok=True)

    if pygame.display.get_surface() is None:
        pygame.display.set_mode((1, 1))

    BODY = (215, 203, 190, 255)
    OUTLINE = (42, 30, 22, 255)
    LEAF = (121, 197, 107, 255)
    GLOW = (239, 216, 142, 200)
    DIRT = (170, 130, 90, 255)
    SPARK = (208, 190, 130, 255)

    def make_sheet(path: str, frames: int, painter) -> None:
        if os.path.exists(path):
            return
        sheet = pygame.Surface((frames * 32, 32), pygame.SRCALPHA)
        for i in range(frames):
            frame = pygame.Surface((32, 32), pygame.SRCALPHA)
            painter(frame, i, frames)
            sheet.blit(frame, (i * 32, 0))
        pygame.image.save(sheet, path)

    def draw_rootling_pose(surf: pygame.Surface, x: int, y: int, w: int = 16, h: int = 20,
                           mouth: bool = False, stretch: int = 0, eye_wide: bool = False) -> None:
        rect = pygame.Rect(x, y - stretch, w, h + stretch)
        pygame.draw.ellipse(surf, OUTLINE, rect.inflate(4, 4))
        pygame.draw.ellipse(surf, BODY, rect)
        leaf_base = (x + w // 2, y - 6 - stretch)
        pygame.draw.line(surf, OUTLINE, leaf_base, (leaf_base[0] + 4, leaf_base[1] - 4), 2)
        pygame.draw.circle(surf, LEAF, (leaf_base[0] + 5, leaf_base[1] - 4), 3)
        eye_radius = 3 if eye_wide else 2
        pygame.draw.circle(surf, (40, 40, 40), (x + w // 2 + 3, y + 4 - stretch), eye_radius)
        if mouth:
            pygame.draw.rect(surf, (40, 40, 40), pygame.Rect(x + w // 2, y + 8 - stretch, 6, 3))
        glow = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.circle(glow, GLOW, (x + w // 2, y + h // 2), 7)
        surf.blit(glow, (0, 0), special_flags=pygame.BLEND_PREMULTIPLIED)

    def idle_painter(frame: pygame.Surface, i: int, n: int) -> None:
        bob = int(2 * math.sin((i / max(1, n)) * math.tau))
        draw_rootling_pose(frame, 8, 10 + bob)

    def walk_painter(frame: pygame.Surface, i: int, n: int) -> None:
        bob = int(2 * math.sin((i / max(1, n)) * math.tau))
        draw_rootling_pose(frame, 8, 9 + bob)
        pygame.draw.rect(frame, OUTLINE, pygame.Rect(8 + (i % 2) * 2, 26, 5, 2))
        pygame.draw.rect(frame, OUTLINE, pygame.Rect(18 - (i % 2) * 2, 26, 5, 2))

    def fall_painter(frame: pygame.Surface, i: int, n: int, panic: bool = False) -> None:
        flutter = (i % 2) * 2
        draw_rootling_pose(frame, 8, 8, stretch=3, mouth=panic, eye_wide=panic)
        pygame.draw.circle(frame, LEAF, (16 + flutter, 4), 3)
        if panic:
            pygame.draw.rect(frame, (40, 40, 40), pygame.Rect(18, 16, 6, 3))

    def dig_painter(frame: pygame.Surface, i: int, n: int) -> None:
        draw_rootling_pose(frame, 8, 12 + (i % 2))
        for _ in range(6):
            px = random.randint(10, 22)
            py = random.randint(22, 28)
            frame.set_at((px, py), DIRT)

    def bridge_painter(frame: pygame.Surface, i: int, n: int) -> None:
        draw_rootling_pose(frame, 8, 10)
        chip_x = 18 + (i % 3)
        chip_y = 12 - (i % 2)
        pygame.draw.circle(frame, SPARK, (chip_x, chip_y), 2)

    def block_painter(frame: pygame.Surface, i: int, n: int) -> None:
        draw_rootling_pose(frame, 8, 10)
        pygame.draw.rect(frame, OUTLINE, pygame.Rect(6, 16, 6, 3))
        pygame.draw.rect(frame, OUTLINE, pygame.Rect(20, 16, 6, 3))

    def explode_painter(frame: pygame.Surface, i: int, n: int) -> None:
        t = i / max(1, n - 1)
        if t < 0.35:
            stretch = int(8 * t)
            draw_rootling_pose(frame, 8, 12, stretch=stretch)
        else:
            radius = int(6 + 20 * (t - 0.35) / 0.65)
            for _ in range(30):
                ang = random.random() * math.tau
                dist = random.uniform(radius * 0.5, radius)
                x = int(16 + math.cos(ang) * dist)
                y = int(16 + math.sin(ang) * dist)
                color = SPARK if random.random() < 0.5 else DIRT
                if 0 <= x < 32 and 0 <= y < 32:
                    frame.set_at((x, y), color)

    def particles_painter(frame: pygame.Surface, i: int, n: int) -> None:
        frame.fill((0, 0, 0, 0))
        cols = 8
        rows = 2
        for y in range(rows):
            for x in range(cols):
                cx = x * 4 + 2
                cy = y * 4 + 2
                color = SPARK if y == 0 else DIRT
                pygame.draw.circle(frame, color, (cx, cy), 2)

    make_sheet("assets/rootling/rootling_idle.png", 6, idle_painter)
    make_sheet("assets/rootling/rootling_walk.png", 8, walk_painter)
    make_sheet("assets/rootling/rootling_fall.png", 6, lambda f, i, n: fall_painter(f, i, n, False))
    make_sheet("assets/rootling/rootling_fall_panic.png", 6, lambda f, i, n: fall_painter(f, i, n, True))
    make_sheet("assets/rootling/rootling_dig.png", 6, dig_painter)
    make_sheet("assets/rootling/rootling_bridge.png", 8, bridge_painter)
    make_sheet("assets/rootling/rootling_block.png", 4, block_painter)
    make_sheet("assets/rootling/rootling_explode.png", 10, explode_painter)
    make_sheet("assets/fx/particles.png", 2, particles_painter)


def load_rootling_animations() -> Dict[str, List[pygame.Surface]]:
    animations: Dict[str, List[pygame.Surface]] = {}
    animations["idle"] = load_sheet("assets/rootling/rootling_idle.png", 32, 32)
    animations["walk"] = load_sheet("assets/rootling/rootling_walk.png", 32, 32)
    animations["fall"] = load_sheet("assets/rootling/rootling_fall.png", 32, 32)
    animations["fall_p"] = load_sheet("assets/rootling/rootling_fall_panic.png", 32, 32)
    animations["dig"] = load_sheet("assets/rootling/rootling_dig.png", 32, 32)
    animations["bridge"] = load_sheet("assets/rootling/rootling_bridge.png", 32, 32)
    animations["block"] = load_sheet("assets/rootling/rootling_block.png", 32, 32)
    animations["explode"] = load_sheet("assets/rootling/rootling_explode.png", 32, 32)
    return animations


@dataclass
class Particle:
    pos: pygame.Vector2
    vel: pygame.Vector2
    life: float
    max_life: float
    size: int
    color: Tuple[int, int, int, int]
    gravity: float
    kind: str


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
        max_tx = (rect.right - 1) // TILE
        min_ty = rect.top // TILE
        max_ty = (rect.bottom - 1) // TILE
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
        max_tx = (rect.right - 1) // TILE
        min_ty = rect.top // TILE
        max_ty = (rect.bottom - 1) // TILE
        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                if self.is_hazard(tx, ty):
                    tile_rect = pygame.Rect(tx * TILE, ty * TILE, TILE, TILE)
                    if rect.colliderect(tile_rect):
                        return True
        return False

    def world_to_tile(self, x: float, y: float) -> Tuple[int, int]:
        return int(x // TILE), int(y // TILE)

    def predicted_fall_tiles(self, rect: pygame.Rect) -> int:
        tx = rect.centerx // TILE
        ty = rect.bottom // TILE
        dist = 0
        while ty < self.height:
            if self.is_solid(int(tx), int(ty)):
                break
            ty += 1
            dist += 1
        return dist


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
        self.animations: Dict[str, Animation] = {}
        self.exploding = False
        self.explode_timer = 0.0
        self.panic = False
        self.glow_phase = 0.0
        self._last_anim_key: Optional[str] = None
        self.game: Optional['Game'] = None
        self.level_ref: Optional[Level] = None

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
            if state != RootlingState.FALL:
                self.panic = False
            self.state = state
            self._last_anim_key = None

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

    def current_animation_key(self) -> str:
        if self.state == RootlingState.WALK:
            return "walk"
        if self.state == RootlingState.FALL:
            return "fall_p" if self.panic else "fall"
        if self.state == RootlingState.DIG:
            return "dig"
        if self.state == RootlingState.BRIDGE:
            return "bridge"
        if self.state == RootlingState.BLOCK:
            return "block"
        if self.state == RootlingState.DEAD and self.exploding:
            return "explode"
        return "idle"

    def explode(self) -> None:
        if self.state in (RootlingState.DEAD, RootlingState.EXITED):
            return
        level = self.level_ref
        self.exploding = True
        self.set_state(RootlingState.DEAD)
        self.dead_timer = 0.0
        self.explode_timer = 0.0
        self.vel.xy = 0, 0
        self.selected = False
        if level is not None:
            center_tx, center_ty = level.world_to_tile(self.rect.centerx, self.rect.centery)
            radius = 1
            for ty in range(center_ty - radius, center_ty + radius + 1):
                for tx in range(center_tx - radius, center_tx + radius + 1):
                    if abs(tx - center_tx) <= radius and abs(ty - center_ty) <= radius:
                        if level.is_diggable(tx, ty):
                            level.set_tile(tx, ty, '.')
                        level.remove_bridge_tile(tx, ty)
        if "explode" in self.animations:
            self.animations["explode"].reset()
        if self.game:
            self.game.spawn_explosion_particles(self.rect.center)
            self.game.shake(0.15, 3)

    # ----------------------------------------------------------------------------------
    # Update logic
    # ----------------------------------------------------------------------------------
    def update(self, dt: float, level: Level, others: List['Rootling']) -> None:
        self.level_ref = level
        if self.state == RootlingState.EXITED:
            return
        if self.state == RootlingState.DEAD:
            self.dead_timer += dt
            if self.exploding:
                self.explode_timer += dt
            self.update_animation(dt)
            return

        if level.rect_overlaps_hazard(self.rect):
            self.die()
            self.update_animation(dt)
            return

        if self.bomber_timer is not None:
            self.bomber_timer -= dt
            if self.bomber_timer <= 0:
                self.explode()
                self.update_animation(dt)
                return

        if self.state == RootlingState.BLOCK:
            self.vel.xy = 0, 0
            self.on_ground = self.check_ground(level)
            self.update_animation(dt)
            return

        if self.state == RootlingState.DIG:
            self.update_dig(dt, level)
            self.update_animation(dt)
            return

        if self.state == RootlingState.BRIDGE:
            self.update_bridge(dt, level)
            self.update_animation(dt)
            return

        if self.state == RootlingState.FALL:
            self.update_fall(dt, level)
        elif self.state == RootlingState.WALK:
            self.update_walk(dt, level, others)
        self.update_animation(dt)

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
        self.panic = level.predicted_fall_tiles(self.rect) > FATAL_FALL_TILES
        if self.on_ground:
            fall_tiles = max(0.0, (self.rect.bottom - self.fall_start_y) / TILE)
            if fall_tiles > FATAL_FALL_TILES:
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
                if self.game:
                    center = (self.rect.centerx, self.rect.bottom)
                    self.game.spawn_dig_particles(center)
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
        dx = step_index + 1
        dy = (step_index + 1) // 2
        target_tx = self.bridge_anchor[0] + self.direction * dx
        target_ty = self.bridge_anchor[1] - dy
        if not level.add_bridge_tile(target_tx, target_ty, self.direction):
            self.set_state(RootlingState.WALK)
            return

        self.bridge_steps += 1
        tile_left = target_tx * TILE
        self.pos.x = tile_left + (TILE - ROOTLING_WIDTH) / 2
        self.pos.y = target_ty * TILE - ROOTLING_HEIGHT
        self.on_ground = True

        if self.game:
            center = (target_tx * TILE + TILE // 2, target_ty * TILE + TILE // 2)
            self.game.spawn_bridge_particles(center, self.direction)

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
            self.exploding = False
            self.selected = False

    def update_animation(self, dt: float) -> None:
        key = self.current_animation_key()
        if key != self._last_anim_key:
            anim = self.animations.get(key)
            if anim:
                anim.reset()
            self._last_anim_key = key
        anim = self.animations.get(key)
        if anim:
            anim.update(dt)

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

        hint = "1:Dig 2:Bridge 3:Block 4:Bomber | X:Explode selected | R:Restart | ESC:Quit"
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
        ensure_assets()
        self.anim_defs = load_rootling_animations()
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
        self.bomber_font = pygame.font.Font(None, 24)
        self.selected: Optional[Rootling] = None
        self.particles: List[Particle] = []
        self.shake_time = 0.0
        self.shake_mag = 0
        self.render_offset: Tuple[int, int] = (0, 0)
        self.sprite_cache: Dict[Tuple[int, int], pygame.Surface] = {}

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
        self.particles.clear()
        self.shake_time = 0.0
        self.shake_mag = 0

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
                    elif event.key == pygame.K_x and self.selected:
                        self.selected.explode()
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

    def shake(self, duration: float, magnitude: int) -> None:
        self.shake_time = max(self.shake_time, duration)
        self.shake_mag = max(self.shake_mag, magnitude)

    def spawn_dig_particles(self, center: Tuple[int, int]) -> None:
        cx, cy = center
        for _ in range(random.randint(4, 8)):
            vel = pygame.Vector2(random.uniform(-40, 40), random.uniform(60, 120))
            life = random.uniform(0.35, 0.6)
            self.particles.append(
                Particle(
                    pos=pygame.Vector2(cx + random.uniform(-4, 4), cy + random.uniform(-4, 4)),
                    vel=vel,
                    life=life,
                    max_life=life,
                    size=random.randint(2, 3),
                    color=COLOR_PARTICLE_DIRT,
                    gravity=500.0,
                    kind="dust",
                )
            )

    def spawn_bridge_particles(self, center: Tuple[int, int], direction: int) -> None:
        cx, cy = center
        for _ in range(random.randint(2, 4)):
            speed = random.uniform(70, 130)
            vel = pygame.Vector2(direction * speed, random.uniform(-150, -60))
            life = random.uniform(0.25, 0.45)
            self.particles.append(
                Particle(
                    pos=pygame.Vector2(cx, cy - 6),
                    vel=vel,
                    life=life,
                    max_life=life,
                    size=2,
                    color=COLOR_PARTICLE_SPARK,
                    gravity=220.0,
                    kind="spark",
                )
            )

    def spawn_explosion_particles(self, center: Tuple[int, int]) -> None:
        cx, cy = center
        count = random.randint(20, 30)
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(80, 220)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = random.uniform(0.35, 0.65)
            if random.random() < 0.4:
                color = COLOR_PARTICLE_SPARK
                gravity = 150.0
                size = 2
                kind = "spark"
            else:
                color = COLOR_PARTICLE_DIRT
                gravity = 360.0
                size = 3
                kind = "dust"
            self.particles.append(
                Particle(
                    pos=pygame.Vector2(cx, cy),
                    vel=vel,
                    life=life,
                    max_life=life,
                    size=size,
                    color=color,
                    gravity=gravity,
                    kind=kind,
                )
            )

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

        for rootling in self.rootlings:
            if not rootling.animations:
                rootling.animations = {
                    "idle": Animation(self.anim_defs.get("idle", []), 8, True),
                    "walk": Animation(self.anim_defs.get("walk", []), 11, True),
                    "fall": Animation(self.anim_defs.get("fall", []), 10, True),
                    "fall_p": Animation(self.anim_defs.get("fall_p", []), 12, True),
                    "dig": Animation(self.anim_defs.get("dig", []), 10, True),
                    "bridge": Animation(self.anim_defs.get("bridge", []), 10, True),
                    "block": Animation(self.anim_defs.get("block", []), 6, True),
                    "explode": Animation(self.anim_defs.get("explode", []), 16, False),
                }
                rootling.game = self
                rootling.glow_phase = math.tau * ((hash(id(rootling)) & 255) / 255.0)
                rootling._last_anim_key = None

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

        for particle in list(self.particles):
            particle.life -= dt
            if particle.life <= 0:
                self.particles.remove(particle)
                continue
            particle.vel.y += particle.gravity * dt
            particle.pos += particle.vel * dt

        if self.shake_time > 0:
            self.shake_time = max(0.0, self.shake_time - dt)

        active = [r for r in self.rootlings if r.state not in (RootlingState.EXITED, RootlingState.DEAD)]
        if self.saved >= REQUIRED_TO_SAVE:
            self.game_state = GameState.WIN
        elif self.time_left <= 0 and self.saved < REQUIRED_TO_SAVE:
            self.game_state = GameState.LOSE
        elif self.spawner.spawned >= self.spawner.total and not active and self.saved < REQUIRED_TO_SAVE:
            self.game_state = GameState.LOSE

    # ----------------------------------------------------------------------------------
    def draw(self) -> None:
        ox = oy = 0
        if self.shake_time > 0:
            ox = random.randint(-self.shake_mag, self.shake_mag)
            oy = random.randint(-self.shake_mag, self.shake_mag)
        self.render_offset = (ox, oy)

        self.draw_background()
        self.draw_level()
        self.draw_rootlings()
        self.draw_particles()
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
        ox, oy = self.render_offset
        for y in range(self.level.height):
            for x in range(self.level.width):
                tile = self.level.get_tile(x, y)
                rect = pygame.Rect(x * TILE, y * TILE, TILE, TILE).move(ox, oy)
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
            rect = pygame.Rect(tx * TILE, ty * TILE, TILE, TILE).move(ox, oy)
            pygame.draw.rect(self.screen, COLOR_BRIDGE, rect)
            if direction > 0:
                pygame.draw.line(self.screen, (120, 110, 80), rect.bottomleft, rect.topright, 3)
            else:
                pygame.draw.line(self.screen, (120, 110, 80), rect.topleft, rect.bottomright, 3)

    def draw_rootlings(self) -> None:
        ox, oy = self.render_offset
        time_factor = pygame.time.get_ticks() / 400.0
        for rootling in self.rootlings:
            rect = rootling.rect
            draw_pos = rect.move(ox, oy)
            glow_alpha = int(90 + 60 * math.sin(time_factor + rootling.glow_phase))
            if glow_alpha > 0 and rootling.state != RootlingState.DEAD:
                glow_surface = pygame.Surface((48, 48), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (239, 216, 142, glow_alpha), (24, 26), 18)
                glow_rect = glow_surface.get_rect(center=(rect.centerx + ox, rect.centery + oy + 6))
                self.screen.blit(glow_surface, glow_rect)

            if rootling.state == RootlingState.DEAD and rootling.exploding:
                anim = rootling.animations.get("explode")
                if anim and anim.frames:
                    frame = anim.frame
                    sprite = frame.copy()
                    alpha = max(0, 255 - int(rootling.dead_timer * 180))
                    sprite.set_alpha(alpha)
                    self.screen.blit(sprite, (rect.x - 8 + ox, rect.y - 6 + oy))
                else:
                    surface = pygame.Surface(rect.size, pygame.SRCALPHA)
                    alpha = max(0, 255 - int(rootling.dead_timer * 255))
                    surface.fill((*COLOR_HAZARD_BASE, alpha))
                    self.screen.blit(surface, draw_pos)
                continue

            if rootling.state == RootlingState.DEAD:
                alpha = max(0, 255 - int(rootling.dead_timer * 255))
                surface = pygame.Surface(rect.size, pygame.SRCALPHA)
                surface.fill((*COLOR_HAZARD_BASE, alpha))
                self.screen.blit(surface, draw_pos)
                continue

            key = rootling.current_animation_key()
            anim = rootling.animations.get(key)
            sprite_drawn = False
            if anim and anim.frames:
                frame = anim.frame
                sprite = frame
                if rootling.direction < 0:
                    cache_key = (id(frame), -1)
                    sprite = self.sprite_cache.get(cache_key)
                    if sprite is None:
                        sprite = pygame.transform.flip(frame, True, False)
                        self.sprite_cache[cache_key] = sprite
                draw_point = (rect.x - 8 + ox, rect.y - 6 + oy)
                self.screen.blit(sprite, draw_point)
                sprite_drawn = True

            if not sprite_drawn:
                pygame.draw.rect(self.screen, COLOR_ROOTLING_OUTLINE, draw_pos.inflate(4, 4), border_radius=6)
                pygame.draw.rect(self.screen, COLOR_ROOTLING, draw_pos, border_radius=6)
                eye_y = draw_pos.centery - 4
                eye_x = draw_pos.centerx + rootling.direction * 4
                pygame.draw.circle(self.screen, (40, 40, 40), (eye_x, eye_y), 2)

            if rootling.state == RootlingState.BLOCK:
                block_rect = draw_pos.inflate(10, 4)
                pygame.draw.rect(self.screen, COLOR_SELECTION, block_rect, width=2, border_radius=4)
            if rootling.selected:
                pygame.draw.circle(self.screen, COLOR_SELECTION, (rect.centerx + ox, rect.bottom + oy + 4), 20, 2)
                if rootling.panic:
                    panic_text = self.bomber_font.render("Panic!", True, COLOR_HUD_WARNING)
                    text_rect = panic_text.get_rect(midbottom=(rect.centerx + ox, rect.top + oy - 6))
                    self.screen.blit(panic_text, text_rect)
            if rootling.bomber_timer is not None:
                remaining = max(rootling.bomber_timer, 0.0)
                countdown_text = f"{remaining:.1f}"
                text_surface = self.bomber_font.render(countdown_text, True, COLOR_HUD_WARNING)
                text_rect = text_surface.get_rect(midbottom=(rect.centerx + ox, rect.top + oy - 4))
                self.screen.blit(text_surface, text_rect)

    def draw_particles(self) -> None:
        ox, oy = self.render_offset
        for particle in self.particles:
            alpha = max(0, int(255 * (particle.life / particle.max_life)))
            if alpha <= 0:
                continue
            size = particle.size
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            r, g, b, _ = particle.color
            pygame.draw.circle(surf, (r, g, b, alpha), (size, size), size)
            self.screen.blit(surf, (int(particle.pos.x - size + ox), int(particle.pos.y - size + oy)))


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def main() -> None:
    game = Game()
    game.run()


if __name__ == "__main__":
    main()

import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(scope="session")
def app():
    """One headless App shared by every test (fonts and maths are cached per process)."""
    import pygame
    pygame.init()
    pygame.display.set_mode((64, 64))
    from main import App
    a = App(headless=True)
    a.prerender()
    return a

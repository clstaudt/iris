"""Unit tests for switching tool groups on and off at runtime."""

from iris import tools as tools_module
from iris.tools import Tools


def test_groups_switch_their_tools(monkeypatch):
    monkeypatch.setattr(tools_module, "HAS_CAMERA", True)
    monkeypatch.setattr(tools_module, "HAS_SCREEN", True)
    tools = Tools()
    tools.enable_camera()
    tools.enable_screen_capture()

    assert tools.group_states()["camera"] == {"enabled": True, "available": True}
    assert tools.set_group("camera", False) is True
    assert "look" not in tools._enabled_tools
    assert "capture_screen" in tools._enabled_tools
    assert tools.group_states()["camera"]["enabled"] is False

    assert tools.set_group("camera", True) is True
    assert "look" in tools._enabled_tools


def test_web_group_covers_search_and_browse():
    tools = Tools()
    if not tools_module.HAS_SEARCH:
        return
    tools.set_group("web", True)
    assert {"search", "browse"} <= set(tools._enabled_tools)
    tools.set_group("web", False)
    assert not {"search", "browse"} & set(tools._enabled_tools)
    assert not tools.has_tools()


def test_unavailable_or_unknown_groups_refuse(monkeypatch):
    monkeypatch.setattr(tools_module, "HAS_SCREEN", False)
    tools = Tools()
    assert tools.set_group("screen", True) is False
    assert tools.group_states()["screen"] == {"enabled": False, "available": False}
    assert tools.set_group("teleport", True) is False

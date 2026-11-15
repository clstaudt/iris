"""Roles: a focus a thread can be given, such as research or a language to practise.

A role is a name and a focus, instructions that follow the persona's own in
the prompt. It never changes the pipeline, and a thread need not have one.
Iris ships a few (iris/roles); the user's own are kept in ~/.iris/roles, one
YAML file each. Editing a shipped role saves the user's copy, which replaces
it until it is reset.
"""

import re
from pathlib import Path
from typing import Optional

import yaml

# Roles shipped with Iris, and those created or edited by the user
BUILTIN_DIR = Path(__file__).parent / "roles"
USER_DIR = Path.home() / ".iris" / "roles"


class Role:
    """A focus for a thread: a name and instructions."""

    def __init__(self, name: str, focus: str, role_id: str, path: Optional[Path] = None):
        self.name = name
        self.focus = focus
        self.id = role_id
        self.path = path

    @classmethod
    def from_file(cls, path: Path) -> "Role":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(
            str(data["name"]).strip(),
            str(data.get("focus") or "").strip(),
            path.stem.lower(),
            path,
        )

    @property
    def builtin(self) -> bool:
        """Whether the role is one shipped with Iris, as shipped."""
        return self.path is not None and self.path.parent == BUILTIN_DIR

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "focus": self.focus,
            # Shipped with Iris: it can be reset once edited, but not deleted
            "shipped": shipped(self.id),
            "edited": shipped(self.id) and not self.builtin,
        }


class _RoleDumper(yaml.SafeDumper):
    """Writes multi-line text as a block, the way the role files are written."""


_RoleDumper.add_representer(
    str,
    lambda dumper, text: dumper.represent_scalar(
        "tag:yaml.org,2002:str", text, style="|" if "\n" in text else None
    ),
)


def slug(name: str) -> str:
    """A file name for a role called this."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "role"


def load_all_roles() -> dict[str, Role]:
    """The shipped roles, and the user's, which replace shipped ones of the
    same file name."""
    roles: dict[str, Role] = {}
    for folder in (BUILTIN_DIR, USER_DIR):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.yaml")):
            try:
                role = Role.from_file(path)
            except Exception as e:
                print(f"Warning: Could not load role from {path}: {e}")
                continue
            roles[role.id] = role
    return roles


ROLES = load_all_roles()


def reload_roles() -> list[Role]:
    """Read the role files again; the roles, by name."""
    fresh = load_all_roles()
    ROLES.clear()
    ROLES.update(fresh)
    return list_roles()


def list_roles() -> list[Role]:
    """Every role, by name."""
    return sorted(ROLES.values(), key=lambda role: role.name.casefold())


def find_role(role_id: Optional[str]) -> Optional[Role]:
    """The role of this id, or None (also for no id)."""
    return ROLES.get(role_id) if role_id else None


def shipped(role_id: str) -> bool:
    """Whether Iris ships a role of this id (which the user may have edited)."""
    return (BUILTIN_DIR / f"{role_id}.yaml").exists()


def save_role(settings: dict, role_id: Optional[str] = None) -> str:
    """Write a role to the user's folder and load it: a new one (no id) gets
    a file name from its name, a shipped one is saved as the user's copy.

    Returns:
        The role's id
    """
    if not role_id:
        base = slug(settings["name"])
        role_id, n = base, 2
        while role_id in ROLES:
            role_id, n = f"{base}-{n}", n + 1
    path = USER_DIR / f"{role_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    focus = (settings.get("focus") or "").strip()
    data = {"name": settings["name"].strip(), "focus": focus + "\n" if focus else ""}
    with open(path, "w") as f:
        yaml.dump(data, f, Dumper=_RoleDumper, sort_keys=False, allow_unicode=True)
    reload_roles()
    return role_id


def delete_role(role_id: str) -> bool:
    """Remove the user's role file: the user's own role is gone, an edited
    shipped one is as shipped again.

    Returns:
        Whether a role of that id is still there (the shipped one)
    """
    role = ROLES.get(role_id)
    if role is None or role.builtin or role.path is None:
        return role is not None
    role.path.unlink(missing_ok=True)
    reload_roles()
    return role_id in ROLES

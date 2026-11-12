"""An example plugin: a clock the model can read.

Copy this file into ~/.iris/plugins (or the folder IRIS_PLUGINS_DIR names)
and start Iris: "Clock" appears under Settings > Tools, and Iris can tell
the time when asked.
"""

import time

from iris.plugin import Plugin, Tool


def current_time(**_) -> dict:
    """The local time and date, for the model to put into words."""
    return {"success": True, "result": time.strftime("%A %d %B %Y, %H:%M")}


class Clock(Plugin):
    id = "clock"
    name = "Clock"

    def tools(self, context):
        return [
            Tool(
                name="current_time",
                description="The current local date and time.",
                run=current_time,
                guidance="- 'current_time' function: call it when the user asks for the time or date\n",
                label="Looked at the clock",
            )
        ]


plugin = Clock()

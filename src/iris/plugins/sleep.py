"""Going to sleep: Iris ends the conversation itself once the user is done,
as the sleep word would. It says its parting words, then listens only for
its wake word again.

The tool is offered only while the wake word is on: whether the model ends
a conversation is up to it, and it may only where the user chose to wake
Iris by its wake word. The sleep word, said by the user, always works.
"""

from iris.plugin import Plugin, PluginContext, Tool

DESCRIPTION = (
    "End the conversation and go to sleep: you stop listening until the user calls you"
    " by your wake word. Use it when the user is clearly finished for now, such as when"
    " they say goodbye or good night, thank you in a way that closes the conversation,"
    " or ask you to go to sleep or stop listening. Say your goodbye in the same reply;"
    " it is spoken before you fall asleep."
)
GUIDANCE = (
    "- 'end_conversation' function: Ends the conversation; you go to sleep until the user"
    " calls you by your wake word.\n"
    "  * Call it when the user is clearly done for now: they say goodbye or good night,"
    " thank you to close the conversation, say that was all they needed, or ask you to go"
    " to sleep or stop listening\n"
    "  * Say your goodbye in the same reply, as you would to someone who is leaving; it is"
    " spoken before you fall asleep\n"
    "  * Do not call it after an ordinary answer, nor when the user thanks you in passing"
    " or the conversation may well go on\n"
)
# What the model is told when it called the tool without a word: it answers
# once more, and that is the goodbye
RESULT = "You go to sleep once this reply has been spoken. Say goodbye to the user now."


class Sleep(Plugin):
    id = "sleep"
    name = "Going to sleep"

    def tools(self, context: PluginContext) -> list[Tool]:
        if context.sleep is None:
            return []
        sleep = context.sleep

        def end_conversation(**_) -> dict:
            sleep()
            return {"success": True, "result": RESULT, "ends_reply": True}

        return [
            Tool(
                name="end_conversation",
                description=DESCRIPTION,
                run=end_conversation,
                guidance=GUIDANCE,
                label="Went to sleep",
                detail=lambda arguments: "",
                offered=context.can_sleep,
            )
        ]


plugin = Sleep()

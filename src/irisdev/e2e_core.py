"""The core as the app's end-to-end test (ui/e2e) runs it: iris.rpc_server,
with a virtual room for its audio in place of the computer's microphone and
speaker.

The room holds nothing but Iris's own voice, which comes back into its
microphone as from a laptop's speakers. What goes on in the real room (a video
playing, people talking) never becomes a turn, the test still hears the voice
come back as the app would, and nothing sounds on the speakers.

    python -m irisdev.e2e_core
"""

from iris import audio, rpc_server
from irisdev.voicetest.acoustics import Space


def main():
    # The microphone and the speaker are taken from here when the assistant
    # is built, after the app has asked for it
    audio.SYSTEM = Space().endpoint("Iris")
    rpc_server.main()


if __name__ == "__main__":
    main()

import { mount } from "svelte";
import App from "./App.svelte";
import Presence from "./Presence.svelte";
import { setOrbResting } from "./lib/orb.svelte";
import "./app.css";

/** The presence window loads this page too, as ?window=presence */
const isPresence = new URLSearchParams(location.search).get("window") === "presence";

// The orb draws only in the window on screen: presence starts hidden, and
// so does the conversation when Iris opens as presence
setOrbResting(isPresence || (window.iris?.presence ?? false));

const app = mount(isPresence ? Presence : App, {
  target: document.getElementById("app")!,
});

export default app;

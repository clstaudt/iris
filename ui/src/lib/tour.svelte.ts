/** The feature tour as the app runs it (tour.ts): it opens and closes the
 *  threads and the settings as the user would, and keeps whether it was
 *  taken in the app's settings. */

import { loadSettings, saveSettings, store } from "./state.svelte";
import { TourGuide, type TourPosition } from "./tour";

const now = $state<{ position: TourPosition | null; away: boolean }>({ position: null, away: false });

export const tour = new TourGuide(
  {
    view: () => ({ settingsPage: store.settingsPage, sidebar: store.sidebar, panel: store.panel }),
    show(view) {
      if (store.settingsPage !== view.settingsPage) store.openSettings(view.settingsPage);
      if (store.sidebar !== view.sidebar) store.toggleSidebar();
      if (view.panel === "memory" && store.panel !== "memory") store.togglePanel("memory");
      else if (view.panel === null) store.panel = null;
    },
    saved: loadSettings,
    save: saveSettings,
  },
  now
);

/** The feature tour as the app runs it (tour.ts): it opens and closes the
 *  threads and the settings as the user would, and keeps whether it was
 *  taken in the app's settings. */

import { loadSettings, saveSettings, store } from "./state.svelte";
import { TourGuide, type TourPosition } from "./tour";

const now = $state<{ position: TourPosition | null }>({ position: null });

export const tour = new TourGuide(
  {
    view: () => ({ settingsPage: store.settingsPage, sidebar: store.sidebar, panel: store.panel }),
    show(view) {
      if (store.settingsPage !== view.settingsPage) store.openSettings(view.settingsPage);
      if (store.sidebar !== view.sidebar) store.toggleSidebar();
      store.panel = view.panel;
    },
    saved: loadSettings,
    save: saveSettings,
  },
  now
);

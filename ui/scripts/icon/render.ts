/** The app icon: the fully open iris of the startup bloom on a macOS-style
 *  tile. Rendered by make-icon.cjs (`npm run icon`). */

import { drawIris, type IrisStyle } from "../../src/lib/iris";

const SIZE = 1024;
// Apple's icon grid: the tile fills 824 of the 1024 px, with ~22% corners
const INSET = 100;
const TILE = SIZE - 2 * INSET;
const CORNER = TILE * 0.225;

// The bloom is airy; at 16 px a flower needs more body to read
const ICON_STYLE: IrisStyle = {
  fill: 1.5,
  edgeWidth: 7,
  edgeBlur: 26,
  veinWidth: 2.2,
  heart: 0.55,
};

function tile(c: CanvasRenderingContext2D) {
  c.beginPath();
  c.roundRect(INSET, INSET, TILE, TILE, CORNER);
}

function render(): string {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = SIZE;
  const c = canvas.getContext("2d")!;

  // The tile, with the soft shadow macOS icons sit on
  c.save();
  c.shadowColor = "rgba(0, 0, 0, 0.45)";
  c.shadowBlur = 30;
  c.shadowOffsetY = 12;
  tile(c);
  const bg = c.createLinearGradient(0, INSET, 0, INSET + TILE);
  bg.addColorStop(0, "#26272d");
  bg.addColorStop(1, "#111216");
  c.fillStyle = bg;
  c.fill();
  c.restore();

  // The flower, kept inside the tile
  c.save();
  tile(c);
  c.clip();
  drawIris(
    c,
    // A touch below centre: the top petal is the longest
    { cx: SIZE / 2, cy: SIZE / 2 + 20, R: 190, open: 1, merge: 0, sway: 0, fade: 1, glowRadius: 420 },
    ICON_STYLE
  );
  c.restore();

  // A hairline edge, so the tile holds against dark backgrounds too
  tile(c);
  c.strokeStyle = "rgba(255, 255, 255, 0.08)";
  c.lineWidth = 2;
  c.stroke();

  return canvas.toDataURL("image/png");
}

(window as unknown as { renderIcon: () => string }).renderIcon = render;

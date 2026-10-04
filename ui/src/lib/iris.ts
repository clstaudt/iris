/** The iris flower: drawn by the orb's startup bloom and by the app icon. */

export type Rgb = [number, number, number];

const hex = (h: string): Rgb => {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};

// The title image's petals: violet to blue, lit edges, a warm beard
export const PETAL_BASE = hex("#8d84ee");
export const PETAL_TIP = hex("#79a3ee");
export const PETAL_EDGE = hex("#b2dcf2");
export const PETAL_BEARD = hex("#e6ab7a");

const TAU = Math.PI * 2;
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const rgba = (c: Rgb, a: number) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;

/** How strongly the flower is drawn. The bloom is airy; an icon must hold up
 *  at 16 px, so it gets more body. */
export interface IrisStyle {
  fill: number; // petal opacity multiplier
  edgeWidth: number; // lit edge, in px
  edgeBlur: number; // glow around the edge, in px
  veinWidth: number; // in px; 0 leaves the veins out
  heart: number; // the bright centre's opacity
}

export const BLOOM_STYLE: IrisStyle = {
  fill: 1,
  edgeWidth: 1.4,
  edgeBlur: 6,
  veinWidth: 0.5,
  heart: 0.4,
};

export interface IrisPose {
  cx: number;
  cy: number;
  /** The orb ring's radius; the petals are sized from it */
  R: number;
  /** 0 a closed bud, 1 fully open */
  open: number;
  /** 0 the flower, 1 merged into the ring (petals flat at the ring's size) */
  merge: number;
  /** Small rotation that sways the petals, in radians */
  sway: number;
  /** Overall opacity */
  fade: number;
  /** Reach of the soft glow behind the flower */
  glowRadius: number;
}

/** One petal along -y from the centre: narrow at the base, broad and rounded
 *  at the tip, translucent with a lit edge and fine veins. */
function petal(
  c: CanvasRenderingContext2D,
  pose: IrisPose,
  style: IrisStyle,
  angle: number,
  length: number,
  width: number,
  beard: boolean
) {
  if (length < 0.5) return;
  const alpha = pose.fade;
  c.save();
  c.translate(pose.cx, pose.cy);
  c.rotate(angle);
  c.beginPath();
  c.moveTo(0, 0);
  c.bezierCurveTo(width * 0.9, -length * 0.22, width * 1.15, -length * 0.82, 0, -length);
  c.bezierCurveTo(-width * 1.15, -length * 0.82, -width * 0.9, -length * 0.22, 0, 0);
  c.closePath();

  const fill = c.createLinearGradient(0, 0, 0, -length);
  fill.addColorStop(0, rgba(PETAL_BASE, Math.min(1, 0.34 * style.fill) * alpha));
  fill.addColorStop(1, rgba(PETAL_TIP, Math.min(1, 0.12 * style.fill) * alpha));
  c.fillStyle = fill;
  c.fill();

  c.shadowColor = rgba(PETAL_EDGE, 0.5 * alpha);
  c.shadowBlur = style.edgeBlur;
  c.strokeStyle = rgba(PETAL_EDGE, 0.55 * alpha);
  c.lineWidth = style.edgeWidth;
  c.stroke();
  c.shadowBlur = 0;

  // Veins fan out from the base
  if (style.veinWidth > 0) {
    c.strokeStyle = rgba(PETAL_EDGE, 0.16 * alpha);
    c.lineWidth = style.veinWidth;
    for (const v of [-0.55, -0.25, 0, 0.25, 0.55]) {
      c.beginPath();
      c.moveTo(0, -length * 0.06);
      c.quadraticCurveTo(width * v * 1.2, -length * 0.5, width * v * 0.75, -length * 0.88);
      c.stroke();
    }
  }

  if (beard) {
    c.fillStyle = rgba(PETAL_BEARD, Math.min(1, 0.4 * style.fill) * alpha);
    c.beginPath();
    c.ellipse(0, -length * 0.26, width * 0.16, length * 0.14, 0, 0, TAU);
    c.fill();
  }
  c.restore();
}

/** The iris: three broad falls opening flat, three standards between them
 *  staying more upright. Seen from above, a petal tilting open looks longer,
 *  so its drawn length follows the sine of its tilt. */
export function drawIris(c: CanvasRenderingContext2D, pose: IrisPose, style = BLOOM_STYLE) {
  const { cx, cy, R, open, merge: m, sway, fade } = pose;
  if (fade <= 0) return;
  const twist = (1 - open) * 0.6 + m * 0.45; // petals turn as they open

  c.save();
  c.globalCompositeOperation = "lighter"; // overlaps glow, like the photo

  const glow = c.createRadialGradient(cx, cy, 0, cx, cy, pose.glowRadius);
  glow.addColorStop(0, rgba(PETAL_BASE, 0.18 * fade * (0.4 + 0.6 * open)));
  glow.addColorStop(1, rgba(PETAL_BASE, 0));
  c.fillStyle = glow;
  c.fillRect(cx - pose.glowRadius, cy - pose.glowRadius, pose.glowRadius * 2, pose.glowRadius * 2);

  for (let i = 0; i < 3; i++) {
    const tilt = (0.1 + 0.4 * open) * Math.PI;
    const length = lerp(R * 1.6, R, m) * Math.sin(tilt);
    petal(c, pose, style, (i / 3) * TAU + twist + sway, length, R * (0.3 + 0.28 * open), true);
  }
  for (let i = 0; i < 3; i++) {
    const tilt = (0.06 + 0.26 * open + 0.18 * m) * Math.PI;
    const length = lerp(R * 1.15, R * 0.92, m) * Math.sin(tilt);
    const angle = (i / 3 + 1 / 6) * TAU + twist * 0.7 - sway;
    petal(c, pose, style, angle, length, R * (0.24 + 0.2 * open), false);
  }

  c.fillStyle = rgba([255, 255, 255], style.heart * fade);
  c.beginPath();
  c.arc(cx, cy, Math.max(1.2, R * 0.07), 0, TAU);
  c.fill();
  c.restore();
}

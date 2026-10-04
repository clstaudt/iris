// Renders the app icon into ui/build: icon.png (1024 px) and, on macOS,
// icon.icns. Run through `npm run icon`, which bundles render.ts first.
const { app, BrowserWindow } = require("electron");
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const OUT = path.join(__dirname, "..", "..", "build");

function icns(png) {
  // macOS wants each size twice, the second for Retina screens
  const set = fs.mkdtempSync(path.join(os.tmpdir(), "iris-")) + "/icon.iconset";
  fs.mkdirSync(set);
  for (const size of [16, 32, 128, 256, 512]) {
    for (const [scale, suffix] of [[1, ""], [2, "@2x"]]) {
      const px = String(size * scale);
      const file = path.join(set, `icon_${size}x${size}${suffix}.png`);
      execFileSync("sips", ["-z", px, px, png, "--out", file], { stdio: "ignore" });
    }
  }
  execFileSync("iconutil", ["-c", "icns", set, "-o", path.join(OUT, "icon.icns")]);
}

app.whenReady().then(async () => {
  const win = new BrowserWindow({ show: false });
  await win.loadFile(path.join(__dirname, "render.html"));
  const dataUrl = await win.webContents.executeJavaScript("window.renderIcon()");
  fs.mkdirSync(OUT, { recursive: true });
  const png = path.join(OUT, "icon.png");
  fs.writeFileSync(png, Buffer.from(dataUrl.split(",")[1], "base64"));
  console.log("wrote", png);
  if (process.platform === "darwin") {
    icns(png);
    console.log("wrote", path.join(OUT, "icon.icns"));
  }
  app.quit();
});

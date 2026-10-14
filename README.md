# AI Hand Energy Mesh Studio

Browser-based real-time 2-hand tracking project with a dense animated multi-color line mesh.

## Main upgrades

- Two-hand real-time tracking
- 21 landmarks per hand used as line anchors
- No particle circles / landmark dots in the visual effect
- High-density animated cross-hand rainbow web
- Line endpoints follow the hands and fingers in real time
- Adaptive smoothing for stable motion with less lag
- Gesture-reactive mesh:
  - Open palm: expands the web
  - Pinch: tightens/focuses lines
  - Fist: boosts glow/energy
  - Peace: changes the color mode
- Prism, Aurora, Cyber Laser, Solar Fusion and Ice White modes
- Adjustable line density, width, glow, stabilization and motion trails
- Auto-quality system that reduces visual load when FPS drops
- Screenshot and video recording
- Pause, fullscreen, hide UI and keyboard shortcuts
- Camera alignment accounts for `object-fit: cover` cropping and mirror mode

## Run on Windows

### Easiest method

1. Extract the project folder.
2. Double-click `start_server.bat`.
3. Chrome/Edge will open `http://localhost:5500`.
4. Click **Start AI Camera**.
5. Allow camera permission.

### Manual method

Open PowerShell inside this project folder and run:

```powershell
py -m http.server 5500
```

If `py` does not work:

```powershell
python -m http.server 5500
```

Then open:

```text
http://localhost:5500
```

## Keyboard shortcuts

- `Space` — Pause / resume
- `F` — Fullscreen
- `H` — Hide / show controls
- `S` — Screenshot
- `R` — Start / stop recording

## Important

The project loads MediaPipe Hands from a CDN, so internet is required when the AI model files are loaded for the first time.

For the strongest visual effect, keep both hands visible, with some distance between them, and use even lighting.
   

   
"use strict";

// =============================================================
// AI HAND ENERGY MESH STUDIO
// High-performance two-hand line visualization.
// No circular particles or landmark dots are rendered anywhere.
// =============================================================

const $ = (id) => document.getElementById(id);

const app = $("app");
const video = $("webcam");
const fxCanvas = $("fxCanvas");
const fx = fxCanvas.getContext("2d", { alpha: true, desynchronized: true });
const captureCanvas = $("captureCanvas");
const captureCtx = captureCanvas.getContext("2d", { alpha: false });

const fpsEl = $("fps");
const handCountEl = $("handCount");
const gestureEl = $("gesture");
const distanceValueEl = $("distanceValue");
const cameraStatusEl = $("cameraStatus");
const liveDot = $("liveDot");
const qualityBadge = $("qualityBadge");

const effectModeEl = $("effectMode");
const lineDensityEl = $("lineDensity");
const lineWidthEl = $("lineWidth");
const glowAmountEl = $("glowAmount");
const smoothingEl = $("smoothing");
const trailLengthEl = $("trailLength");

const densityValueEl = $("densityValue");
const widthValueEl = $("widthValue");
const glowValueEl = $("glowValue");
const smoothingValueEl = $("smoothingValue");
const trailValueEl = $("trailValue");

const wireframeToggle = $("wireframeToggle");
const trailsToggle = $("trailsToggle");
const autoQualityToggle = $("autoQualityToggle");
const mirrorToggle = $("mirrorToggle");

const screenshotButton = $("screenshotButton");
const recordButton = $("recordButton");
const recordIcon = $("recordIcon");
const recordLabel = $("recordLabel");
const pauseButton = $("pauseButton");
const pauseIcon = $("pauseIcon");
const pauseLabel = $("pauseLabel");
const fullscreenButton = $("fullscreenButton");
const hideUiButton = $("hideUiButton");
const showUiButton = $("showUiButton");
const collapsePanelButton = $("collapsePanelButton");
const controlPanel = document.querySelector(".control-panel");

const startScreen = $("startScreen");
const startButton = $("startButton");
const startNote = $("startNote");
const toastEl = $("toast");

// MediaPipe's 21-point hand topology. We draw lines only.
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17]
];

const INTERNAL_WEB_CONNECTIONS = [
    [0, 8], [0, 12], [0, 16], [0, 20],
    [4, 8], [4, 12], [4, 16], [4, 20],
    [8, 12], [8, 16], [8, 20],
    [12, 16], [12, 20], [16, 20],
    [5, 13], [5, 17], [9, 17], [5, 20],
    [1, 8], [1, 12], [1, 16], [1, 20]
];

const FINGERTIPS = [4, 8, 12, 16, 20];
const PALM_POINTS = [0, 5, 9, 13, 17];

const MODE_ORDER = ["prism", "aurora", "laser", "sunset", "mono"];

const MODE_CONFIG = {
    prism: {
        speed: 0.055,
        curve: 1.0,
        alpha: 0.82,
        saturation: 100,
        lightness: 62,
        hueBase: 0,
        hueRange: 360,
        dash: null
    },
    aurora: {
        speed: 0.026,
        curve: 1.42,
        alpha: 0.68,
        saturation: 92,
        lightness: 66,
        hueBase: 145,
        hueRange: 170,
        dash: null
    },
    laser: {
        speed: 0.09,
        curve: 0.42,
        alpha: 0.9,
        saturation: 100,
        lightness: 64,
        hueBase: 175,
        hueRange: 165,
        dash: [20, 9]
    },
    sunset: {
        speed: 0.036,
        curve: 0.88,
        alpha: 0.78,
        saturation: 100,
        lightness: 62,
        hueBase: 315,
        hueRange: 120,
        dash: null
    },
    mono: {
        speed: 0.018,
        curve: 0.72,
        alpha: 0.78,
        saturation: 34,
        lightness: 82,
        hueBase: 185,
        hueRange: 45,
        dash: null
    }
};

let viewportWidth = window.innerWidth;
let viewportHeight = window.innerHeight;
let canvasDpr = Math.min(window.devicePixelRatio || 1, 1.6);

let handsModel = null;
let cameraStream = null;
let engineRunning = false;
let enginePaused = false;
let inferenceBusy = false;
let modelReady = false;
let latestHands = [];
let previousSmoothedHands = [null, null];
let previousSlotCenters = [null, null];
let handHistory = [];
let videoFrameCallbackId = null;
let inferenceRafId = null;
let renderRafId = null;
let lastResultsAt = 0;

let displayFps = 0;
let fpsFrameCounter = 0;
let fpsWindowStart = performance.now();
let qualityFactor = 1;
let lastQualityAdjustment = 0;
let lastPeaceCycleAt = 0;
let previousStableGestures = ["none", "none"];
const gestureDebounce = [
    { candidate: "none", count: 0, stable: "none" },
    { candidate: "none", count: 0, stable: "none" }
];

let toastTimer = null;
let isRecording = false;
let mediaRecorder = null;
let recordedChunks = [];

const STRANDS = buildStrands(110);

// -------------------------------------------------------------
// Utilities
// -------------------------------------------------------------

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function lerp(a, b, t) {
    return a + (b - a) * t;
}

function pointLerp(a, b, t) {
    return {
        x: lerp(a.x, b.x, t),
        y: lerp(a.y, b.y, t)
    };
}

function pointDistance(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function normalizedDistance(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function angleAt(a, b, c) {
    const abx = a.x - b.x;
    const aby = a.y - b.y;
    const cbx = c.x - b.x;
    const cby = c.y - b.y;
    const dot = abx * cbx + aby * cby;
    const mag = Math.hypot(abx, aby) * Math.hypot(cbx, cby);
    if (mag < 1e-7) return 0;
    return Math.acos(clamp(dot / mag, -1, 1)) * 180 / Math.PI;
}

function showToast(message, duration = 1800) {
    clearTimeout(toastTimer);
    toastEl.textContent = message;
    toastEl.classList.add("show");
    toastTimer = setTimeout(() => toastEl.classList.remove("show"), duration);
}

function setStatus(message, state = "idle") {
    cameraStatusEl.textContent = message;
    liveDot.classList.remove("active", "error");
    if (state === "active") liveDot.classList.add("active");
    if (state === "error") liveDot.classList.add("error");
}

function updateControlLabels() {
    densityValueEl.textContent = lineDensityEl.value;
    widthValueEl.textContent = Number(lineWidthEl.value).toFixed(1);
    glowValueEl.textContent = glowAmountEl.value;
    smoothingValueEl.textContent = `${smoothingEl.value}%`;
    trailValueEl.textContent = trailLengthEl.value;
}

function resizeCanvas() {
    viewportWidth = window.innerWidth;
    viewportHeight = window.innerHeight;
    canvasDpr = Math.min(window.devicePixelRatio || 1, 1.6);

    fxCanvas.width = Math.max(1, Math.round(viewportWidth * canvasDpr));
    fxCanvas.height = Math.max(1, Math.round(viewportHeight * canvasDpr));
    fx.setTransform(canvasDpr, 0, 0, canvasDpr, 0, 0);
    fx.lineCap = "round";
    fx.lineJoin = "round";
}

function getVideoCoverLayout(targetWidth = viewportWidth, targetHeight = viewportHeight) {
    const sourceWidth = video.videoWidth || 1280;
    const sourceHeight = video.videoHeight || 720;
    const scale = Math.max(targetWidth / sourceWidth, targetHeight / sourceHeight);
    const renderedWidth = sourceWidth * scale;
    const renderedHeight = sourceHeight * scale;
    return {
        sourceWidth,
        sourceHeight,
        scale,
        offsetX: (targetWidth - renderedWidth) / 2,
        offsetY: (targetHeight - renderedHeight) / 2,
        renderedWidth,
        renderedHeight
    };
}

function landmarkToScreen(landmark) {
    const layout = getVideoCoverLayout();
    const mirroredX = mirrorToggle.checked ? 1 - landmark.x : landmark.x;
    return {
        x: layout.offsetX + mirroredX * layout.sourceWidth * layout.scale,
        y: layout.offsetY + landmark.y * layout.sourceHeight * layout.scale,
        z: landmark.z || 0
    };
}

function getPalmCenter(points) {
    const indices = [0, 5, 9, 13, 17];
    let x = 0;
    let y = 0;
    for (const index of indices) {
        x += points[index].x;
        y += points[index].y;
    }
    return { x: x / indices.length, y: y / indices.length };
}

function getNormalizedPalmCenter(points) {
    const indices = [0, 5, 9, 13, 17];
    let x = 0;
    let y = 0;
    for (const index of indices) {
        x += points[index].x;
        y += points[index].y;
    }
    return { x: x / indices.length, y: y / indices.length };
}

function deepCopyScreenHands(hands) {
    return hands.map((hand) => ({
        points: hand.points.map((p) => ({ x: p.x, y: p.y })),
        gesture: hand.gesture
    }));
}

// -------------------------------------------------------------
// Stable, adaptive landmark smoothing
// -------------------------------------------------------------

function smoothLandmarks(rawLandmarks, slot) {
    const previous = previousSmoothedHands[slot];
    if (!previous || previous.length !== rawLandmarks.length) {
        const fresh = rawLandmarks.map((p) => ({ x: p.x, y: p.y, z: p.z || 0 }));
        previousSmoothedHands[slot] = fresh;
        return fresh;
    }

    const stabilization = Number(smoothingEl.value) / 100;
    const baseAlpha = clamp(1 - stabilization * 0.78, 0.2, 0.86);

    const smoothed = rawLandmarks.map((point, index) => {
        const prev = previous[index];
        const speed = Math.hypot(point.x - prev.x, point.y - prev.y);
        const adaptiveAlpha = clamp(baseAlpha + speed * 5.5, baseAlpha, 0.94);
        return {
            x: lerp(prev.x, point.x, adaptiveAlpha),
            y: lerp(prev.y, point.y, adaptiveAlpha),
            z: lerp(prev.z || 0, point.z || 0, adaptiveAlpha)
        };
    });

    previousSmoothedHands[slot] = smoothed;
    return smoothed;
}

// -------------------------------------------------------------
// Gesture recognition
// -------------------------------------------------------------

function fingerExtended(landmarks, mcp, pip, dip, tip) {
    const wrist = landmarks[0];
    const pipAngle = angleAt(landmarks[mcp], landmarks[pip], landmarks[tip]);
    const dipAngle = angleAt(landmarks[pip], landmarks[dip], landmarks[tip]);
    const tipDistance = normalizedDistance(landmarks[tip], wrist);
    const pipDistance = normalizedDistance(landmarks[pip], wrist);
    return pipAngle > 145 && dipAngle > 145 && tipDistance > pipDistance * 1.04;
}

function classifyGesture(landmarks) {
    const palmWidth = Math.max(normalizedDistance(landmarks[5], landmarks[17]), 0.03);
    const pinchRatio = normalizedDistance(landmarks[4], landmarks[8]) / palmWidth;

    const index = fingerExtended(landmarks, 5, 6, 7, 8);
    const middle = fingerExtended(landmarks, 9, 10, 11, 12);
    const ring = fingerExtended(landmarks, 13, 14, 15, 16);
    const pinky = fingerExtended(landmarks, 17, 18, 19, 20);

    if (pinchRatio < 0.42) {
        return { key: "pinch", label: "🤏 Pinch", pinchRatio };
    }

    if (index && middle && ring && pinky) {
        return { key: "open", label: "🖐 Open Palm", pinchRatio };
    }

    if (!index && !middle && !ring && !pinky) {
        return { key: "fist", label: "✊ Fist", pinchRatio };
    }

    if (index && middle && !ring && !pinky) {
        return { key: "peace", label: "✌ Peace", pinchRatio };
    }

    if (index && !middle && !ring && !pinky) {
        return { key: "point", label: "☝ Point", pinchRatio };
    }

    return { key: "hand", label: "✋ Hand", pinchRatio };
}

function stabilizeGesture(slot, gesture) {
    const state = gestureDebounce[slot];
    if (state.candidate === gesture.key) {
        state.count += 1;
    } else {
        state.candidate = gesture.key;
        state.count = 1;
    }

    if (state.count >= 3) {
        state.stable = gesture.key;
    }

    const labels = {
        pinch: "🤏 Pinch",
        open: "🖐 Open Palm",
        fist: "✊ Fist",
        peace: "✌ Peace",
        point: "☝ Point",
        hand: "✋ Hand",
        none: "No hand"
    };

    return {
        ...gesture,
        key: state.stable,
        label: labels[state.stable] || gesture.label
    };
}

function maybeCycleModeFromGesture(hands) {
    const now = performance.now();
    const stableGestures = ["none", "none"];
    hands.forEach((hand) => {
        stableGestures[hand.slot] = hand.gesture.key;
    });

    const newlyPeace = stableGestures.some((gesture, index) =>
        gesture === "peace" && previousStableGestures[index] !== "peace"
    );

    if (newlyPeace && now - lastPeaceCycleAt > 1200) {
        const currentIndex = MODE_ORDER.indexOf(effectModeEl.value);
        effectModeEl.value = MODE_ORDER[(currentIndex + 1) % MODE_ORDER.length];
        lastPeaceCycleAt = now;
        showToast(`Effect: ${effectModeEl.options[effectModeEl.selectedIndex].text}`);
    }

    previousStableGestures = [stableGestures[0] || "none", stableGestures[1] || "none"];
}

// -------------------------------------------------------------
// Strand generation: stable endpoints, no random jitter per frame
// -------------------------------------------------------------

function seededUnit(seed) {
    const value = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
    return value - Math.floor(value);
}

function makeStrand(a, b, seed) {
    return {
        a,
        b,
        phase: seededUnit(seed + 1) * Math.PI * 2,
        curve: 0.55 + seededUnit(seed + 2) * 1.25,
        weight: 0.72 + seededUnit(seed + 3) * 0.62,
        hueOffset: seededUnit(seed + 4) * 360,
        sway: 0.35 + seededUnit(seed + 5) * 1.1
    };
}

function buildStrands(targetCount) {
    const strands = [];
    let seed = 1;

    // Matching landmarks create a clean structural bridge.
    for (let i = 0; i < 21; i += 1) {
        strands.push(makeStrand(i, i, seed++));
    }

    // Finger-tip fan creates the dense rainbow look from the reference.
    for (const a of FINGERTIPS) {
        for (const b of FINGERTIPS) {
            strands.push(makeStrand(a, b, seed++));
        }
    }

    // Palm-to-fingertip cross links add width and depth.
    for (const a of PALM_POINTS) {
        for (const b of FINGERTIPS) {
            strands.push(makeStrand(a, b, seed++));
        }
    }

    // Deterministic extra links for high-density mode.
    while (strands.length < targetCount) {
        const i = strands.length;
        const a = (i * 7 + 3) % 21;
        const b = (i * 11 + 8) % 21;
        strands.push(makeStrand(a, b, seed++));
    }

    return strands;
}


function assignHandsToStableSlots(rawHands) {
    const candidates = rawHands.map((landmarks) => {
        const center = getNormalizedPalmCenter(landmarks);
        return {
            landmarks,
            center: {
                x: mirrorToggle.checked ? 1 - center.x : center.x,
                y: center.y
            }
        };
    });

    const assigned = [null, null];

    if (candidates.length === 2) {
        if (previousSlotCenters[0] && previousSlotCenters[1]) {
            const direct =
                pointDistance(candidates[0].center, previousSlotCenters[0]) +
                pointDistance(candidates[1].center, previousSlotCenters[1]);
            const swapped =
                pointDistance(candidates[0].center, previousSlotCenters[1]) +
                pointDistance(candidates[1].center, previousSlotCenters[0]);

            if (direct <= swapped) {
                assigned[0] = candidates[0];
                assigned[1] = candidates[1];
            } else {
                assigned[0] = candidates[1];
                assigned[1] = candidates[0];
            }
        } else {
            candidates.sort((a, b) => a.center.x - b.center.x);
            assigned[0] = candidates[0];
            assigned[1] = candidates[1];
        }
    } else if (candidates.length === 1) {
        const candidate = candidates[0];
        if (previousSlotCenters[0] || previousSlotCenters[1]) {
            const d0 = previousSlotCenters[0]
                ? pointDistance(candidate.center, previousSlotCenters[0])
                : Number.POSITIVE_INFINITY;
            const d1 = previousSlotCenters[1]
                ? pointDistance(candidate.center, previousSlotCenters[1])
                : Number.POSITIVE_INFINITY;
            assigned[d0 <= d1 ? 0 : 1] = candidate;
        } else {
            assigned[0] = candidate;
        }
    }

    assigned.forEach((entry, slot) => {
        if (entry) previousSlotCenters[slot] = entry.center;
    });

    return assigned;
}

// -------------------------------------------------------------
// MediaPipe setup and camera loop
// -------------------------------------------------------------

function setupHandsModel() {
    if (typeof Hands === "undefined") {
        throw new Error("MediaPipe Hands library failed to load. Check your internet connection.");
    }

    handsModel = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    handsModel.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.72,
        minTrackingConfidence: 0.72
    });

    handsModel.onResults(handleResults);
}

async function startCamera() {
    if (engineRunning) return;

    startButton.disabled = true;
    startButton.querySelector("span").textContent = "Starting camera…";
    setStatus("Starting camera…");

    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("This browser does not support webcam access.");
        }

        setupHandsModel();

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30, max: 60 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = cameraStream;
        await video.play();
        updateMirrorState();
        resizeCanvas();

        setStatus("Loading AI model…");
        await handsModel.send({ image: video });
        modelReady = true;
        engineRunning = true;
        startScreen.classList.add("hidden");
        setStatus("Tracking active", "active");
        startInferenceLoop();
        showToast("Camera and hand tracking are ready");
    } catch (error) {
        console.error("Camera / model startup error:", error);
        setStatus("Camera error", "error");
        startButton.disabled = false;
        startButton.querySelector("span").textContent = "Try Again";
        startNote.textContent = error.message || "Could not start the camera.";
        stopCameraTracks();
    }
}

function stopCameraTracks() {
    if (!cameraStream) return;
    for (const track of cameraStream.getTracks()) {
        track.stop();
    }
    cameraStream = null;
}

function startInferenceLoop() {
    cancelInferenceLoop();

    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
        videoFrameCallbackId = video.requestVideoFrameCallback(processVideoFrame);
    } else {
        inferenceRafId = requestAnimationFrame(processVideoFrameFallback);
    }
}

function cancelInferenceLoop() {
    if (videoFrameCallbackId !== null && "cancelVideoFrameCallback" in video) {
        video.cancelVideoFrameCallback(videoFrameCallbackId);
        videoFrameCallbackId = null;
    }
    if (inferenceRafId !== null) {
        cancelAnimationFrame(inferenceRafId);
        inferenceRafId = null;
    }
}

async function processVideoFrame() {
    if (!engineRunning) return;

    if (!enginePaused && modelReady && !inferenceBusy && video.readyState >= 2) {
        inferenceBusy = true;
        try {
            await handsModel.send({ image: video });
        } catch (error) {
            console.error("Hand inference error:", error);
        } finally {
            inferenceBusy = false;
        }
    }

    if (engineRunning && "requestVideoFrameCallback" in HTMLVideoElement.prototype) {
        videoFrameCallbackId = video.requestVideoFrameCallback(processVideoFrame);
    }
}

async function processVideoFrameFallback() {
    if (!engineRunning) return;

    if (!enginePaused && modelReady && !inferenceBusy && video.readyState >= 2) {
        inferenceBusy = true;
        try {
            await handsModel.send({ image: video });
        } catch (error) {
            console.error("Hand inference error:", error);
        } finally {
            inferenceBusy = false;
        }
    }

    inferenceRafId = requestAnimationFrame(processVideoFrameFallback);
}

function handleResults(results) {
    lastResultsAt = performance.now();
    const rawHands = (results.multiHandLandmarks || []).slice(0, 2);

    if (rawHands.length === 0) {
        latestHands = [];
        previousSmoothedHands = [null, null];
        previousSlotCenters = [null, null];
        handCountEl.textContent = "0";
        gestureEl.textContent = "No hand";
        distanceValueEl.textContent = "—";
        previousStableGestures = ["none", "none"];
        return;
    }

    // Assign detections to persistent slots using nearest-palm motion, so
    // hand order stays stable even when MediaPipe changes detection order.
    const assignedSlots = assignHandsToStableSlots(rawHands);

    latestHands = assignedSlots
        .map((entry, slot) => {
            if (!entry) return null;
            const smoothed = smoothLandmarks(entry.landmarks, slot);
            const rawGesture = classifyGesture(smoothed);
            const gesture = stabilizeGesture(slot, rawGesture);
            return {
                slot,
                normalized: smoothed,
                points: smoothed.map(landmarkToScreen),
                gesture
            };
        })
        .filter(Boolean);

    handCountEl.textContent = String(latestHands.length);
    gestureEl.textContent = latestHands.map((hand) => hand.gesture.label).join("  •  ");

    if (latestHands.length === 2) {
        const palmA = getPalmCenter(latestHands[0].points);
        const palmB = getPalmCenter(latestHands[1].points);
        const normalized = clamp(pointDistance(palmA, palmB) / Math.max(viewportWidth, 1), 0, 1);
        distanceValueEl.textContent = `${Math.round(normalized * 100)}%`;
    } else {
        distanceValueEl.textContent = "Need 2 hands";
    }

    maybeCycleModeFromGesture(latestHands);

    handHistory.unshift(deepCopyScreenHands(latestHands));
    const historyLimit = 10;
    if (handHistory.length > historyLimit) handHistory.length = historyLimit;
}

// -------------------------------------------------------------
// Rendering engine
// -------------------------------------------------------------

function getGestureDynamics(hands) {
    const gestures = hands.map((hand) => hand.gesture.key);
    let spread = 1;
    let energy = 1;
    let focus = 1;

    if (gestures.includes("open")) spread *= 1.34;
    if (gestures.includes("fist")) energy *= 1.55;
    if (gestures.includes("pinch")) {
        focus *= 0.48;
        spread *= 0.72;
    }
    if (gestures.includes("point")) energy *= 1.12;

    return { spread, energy, focus };
}

function modeHue(config, index, time, offset = 0) {
    const moving = time * config.speed;
    const hue = config.hueBase + ((moving + index * 19 + offset) % config.hueRange);
    return ((hue % 360) + 360) % 360;
}

function hsl(hue, saturation, lightness, alpha = 1) {
    return `hsla(${hue.toFixed(1)}, ${saturation}%, ${lightness}%, ${alpha})`;
}

function drawCurvedStrand(ctx, pointA, pointB, strand, index, time, alphaMultiplier, widthMultiplier, dynamics, historical = false) {
    const mode = MODE_CONFIG[effectModeEl.value] || MODE_CONFIG.prism;
    const dx = pointB.x - pointA.x;
    const dy = pointB.y - pointA.y;
    const length = Math.hypot(dx, dy);
    if (length < 4) return;

    const nx = -dy / length;
    const ny = dx / length;
    const phase = time * 0.0024 * strand.sway + strand.phase;
    const oscillation = Math.sin(phase) * length * 0.028;
    const microWave = Math.sin(phase * 1.73 + index) * length * 0.012;
    const curveStrength = mode.curve * strand.curve * dynamics.spread * dynamics.focus;
    const bend = (oscillation + microWave) * curveStrength;

    const cp1Base = pointLerp(pointA, pointB, 0.33);
    const cp2Base = pointLerp(pointA, pointB, 0.67);
    const cp1 = {
        x: cp1Base.x + nx * bend,
        y: cp1Base.y + ny * bend
    };
    const cp2 = {
        x: cp2Base.x - nx * bend * 0.72,
        y: cp2Base.y - ny * bend * 0.72
    };

    const hue1 = modeHue(mode, index, time, strand.hueOffset);
    const hue2 = modeHue(mode, index + 5, time, strand.hueOffset + 85);
    const hue3 = modeHue(mode, index + 11, time, strand.hueOffset + 170);

    const gradient = ctx.createLinearGradient(pointA.x, pointA.y, pointB.x, pointB.y);
    gradient.addColorStop(0, hsl(hue1, mode.saturation, mode.lightness, 1));
    gradient.addColorStop(0.5, hsl(hue2, mode.saturation, Math.min(92, mode.lightness + 15), 1));
    gradient.addColorStop(1, hsl(hue3, mode.saturation, mode.lightness, 1));

    const baseWidth = Number(lineWidthEl.value) * strand.weight * widthMultiplier * dynamics.energy;
    const glow = Number(glowAmountEl.value) * (historical ? 0 : 1) * dynamics.energy;

    ctx.save();
    ctx.strokeStyle = gradient;
    ctx.globalAlpha = mode.alpha * alphaMultiplier;
    ctx.lineWidth = Math.max(0.35, baseWidth);
    ctx.shadowBlur = glow;
    ctx.shadowColor = hsl(hue2, mode.saturation, mode.lightness, 0.95);

    if (mode.dash && !historical) {
        ctx.setLineDash(mode.dash);
        ctx.lineDashOffset = -(time * 0.055 + index * 3);
    } else {
        ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.moveTo(pointA.x, pointA.y);
    ctx.bezierCurveTo(cp1.x, cp1.y, cp2.x, cp2.y, pointB.x, pointB.y);
    ctx.stroke();
    ctx.restore();
}

function drawHandWireframe(hand, handIndex, time, alphaMultiplier = 1) {
    if (!wireframeToggle.checked) return;

    const mode = MODE_CONFIG[effectModeEl.value] || MODE_CONFIG.prism;
    const points = hand.points;
    const baseIndex = handIndex * 31;

    for (let i = 0; i < HAND_CONNECTIONS.length; i += 1) {
        const [a, b] = HAND_CONNECTIONS[i];
        const p1 = points[a];
        const p2 = points[b];
        const hue1 = modeHue(mode, baseIndex + i, time, 0);
        const hue2 = modeHue(mode, baseIndex + i + 8, time, 95);
        const gradient = fx.createLinearGradient(p1.x, p1.y, p2.x, p2.y);
        gradient.addColorStop(0, hsl(hue1, mode.saturation, mode.lightness, 1));
        gradient.addColorStop(1, hsl(hue2, mode.saturation, Math.min(88, mode.lightness + 10), 1));

        fx.save();
        fx.strokeStyle = gradient;
        fx.globalAlpha = 0.88 * alphaMultiplier;
        fx.lineWidth = Math.max(1.05, Number(lineWidthEl.value) * 1.35);
        fx.shadowBlur = Number(glowAmountEl.value) * 0.7;
        fx.shadowColor = hsl(hue1, mode.saturation, mode.lightness, 0.9);
        fx.beginPath();
        fx.moveTo(p1.x, p1.y);
        fx.lineTo(p2.x, p2.y);
        fx.stroke();
        fx.restore();
    }

    // Extra internal lines create a geometric hand mesh without point markers.
    const internalCount = Math.max(6, Math.floor(INTERNAL_WEB_CONNECTIONS.length * qualityFactor));
    for (let i = 0; i < internalCount; i += 1) {
        const [a, b] = INTERNAL_WEB_CONNECTIONS[i];
        const p1 = points[a];
        const p2 = points[b];
        const hue = modeHue(mode, baseIndex + i * 2, time, 180);

        fx.save();
        fx.strokeStyle = hsl(hue, mode.saturation, mode.lightness, 0.58);
        fx.globalAlpha = 0.45 * alphaMultiplier;
        fx.lineWidth = Math.max(0.5, Number(lineWidthEl.value) * 0.62);
        fx.shadowBlur = Number(glowAmountEl.value) * 0.28;
        fx.shadowColor = hsl(hue, mode.saturation, mode.lightness, 0.7);
        fx.beginPath();
        fx.moveTo(p1.x, p1.y);
        fx.lineTo(p2.x, p2.y);
        fx.stroke();
        fx.restore();
    }
}

function drawTwoHandMesh(handA, handB, time, alphaMultiplier = 1, densityScale = 1, historical = false) {
    const userDensity = Number(lineDensityEl.value);
    const effectiveDensity = clamp(
        Math.floor(userDensity * qualityFactor * densityScale),
        historical ? 4 : 10,
        STRANDS.length
    );
    const dynamics = historical ? { spread: 1, energy: 1, focus: 1 } : getGestureDynamics([handA, handB]);

    for (let i = 0; i < effectiveDensity; i += 1) {
        const strand = STRANDS[i];
        const p1 = handA.points[strand.a];
        const p2 = handB.points[strand.b];
        drawCurvedStrand(
            fx,
            p1,
            p2,
            strand,
            i,
            time,
            alphaMultiplier,
            historical ? 0.72 : 1,
            dynamics,
            historical
        );
    }

    if (!historical) {
        drawCoreEnergyLines(handA, handB, time, dynamics);
        drawMovingStreaks(handA, handB, time, dynamics);
    }
}

function drawCoreEnergyLines(handA, handB, time, dynamics) {
    const corePairs = [
        [0, 0], [4, 4], [8, 8], [12, 12], [16, 16], [20, 20]
    ];
    const mode = MODE_CONFIG[effectModeEl.value] || MODE_CONFIG.prism;

    corePairs.forEach(([a, b], index) => {
        const p1 = handA.points[a];
        const p2 = handB.points[b];
        const hue = modeHue(mode, index * 13, time, 40);
        const hue2 = modeHue(mode, index * 17, time, 190);
        const gradient = fx.createLinearGradient(p1.x, p1.y, p2.x, p2.y);
        gradient.addColorStop(0, hsl(hue, 100, 65, 1));
        gradient.addColorStop(0.5, "rgba(255,255,255,0.96)");
        gradient.addColorStop(1, hsl(hue2, 100, 65, 1));

        fx.save();
        fx.strokeStyle = gradient;
        fx.globalAlpha = 0.22 * dynamics.energy;
        fx.lineWidth = Number(lineWidthEl.value) * (4.8 + index * 0.25) * dynamics.energy;
        fx.shadowBlur = Number(glowAmountEl.value) * 1.6;
        fx.shadowColor = hsl(hue2, 100, 65, 0.95);
        fx.beginPath();
        fx.moveTo(p1.x, p1.y);
        fx.lineTo(p2.x, p2.y);
        fx.stroke();
        fx.restore();
    });
}

function drawMovingStreaks(handA, handB, time, dynamics) {
    const anchorsA = [0, 4, 8, 12, 16, 20];
    const anchorsB = [0, 20, 16, 12, 8, 4];
    const mode = MODE_CONFIG[effectModeEl.value] || MODE_CONFIG.prism;
    const streakCount = Math.max(3, Math.floor(6 * qualityFactor));

    for (let i = 0; i < streakCount; i += 1) {
        const p1 = handA.points[anchorsA[i % anchorsA.length]];
        const p2 = handB.points[anchorsB[i % anchorsB.length]];
        const progress = (time * 0.00022 * (1 + i * 0.08) + i / streakCount) % 1;
        const center = pointLerp(p1, p2, progress);
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const len = Math.max(Math.hypot(dx, dy), 1);
        const ux = dx / len;
        const uy = dy / len;
        const streakLength = clamp(len * 0.075, 18, 70) * dynamics.energy;
        const start = { x: center.x - ux * streakLength * 0.5, y: center.y - uy * streakLength * 0.5 };
        const end = { x: center.x + ux * streakLength * 0.5, y: center.y + uy * streakLength * 0.5 };
        const hue = modeHue(mode, i * 23, time, 120);

        fx.save();
        fx.strokeStyle = hsl(hue, 100, 78, 0.95);
        fx.globalAlpha = 0.88;
        fx.lineWidth = Math.max(1, Number(lineWidthEl.value) * 1.4);
        fx.shadowBlur = Number(glowAmountEl.value) * 1.4;
        fx.shadowColor = "rgba(255,255,255,0.95)";
        fx.beginPath();
        fx.moveTo(start.x, start.y);
        fx.lineTo(end.x, end.y);
        fx.stroke();
        fx.restore();
    }
}

function drawMotionHistory(time) {
    if (!trailsToggle.checked) return;

    const requestedLength = Number(trailLengthEl.value);
    const effectiveLength = Math.min(
        requestedLength,
        Math.floor(8 * qualityFactor),
        handHistory.length - 1
    );

    if (effectiveLength <= 0) return;

    for (let age = effectiveLength; age >= 1; age -= 1) {
        const snapshot = handHistory[age];
        if (!snapshot || snapshot.length === 0) continue;
        const alpha = (1 - age / (effectiveLength + 1)) * 0.16;

        snapshot.forEach((hand, index) => {
            drawHandWireframe(hand, index, time - age * 24, alpha * 1.8);
        });

        if (snapshot.length === 2) {
            drawTwoHandMesh(snapshot[0], snapshot[1], time - age * 32, alpha, 0.22, true);
        }
    }
}

function renderFrame(time) {
    renderRafId = requestAnimationFrame(renderFrame);

    fx.clearRect(0, 0, viewportWidth, viewportHeight);

    if (engineRunning) {
        updateFpsAndQuality(time);
    }

    if (latestHands.length > 0) {
        drawMotionHistory(time);

        latestHands.forEach((hand, index) => {
            drawHandWireframe(hand, index, time, 1);
        });

        if (latestHands.length === 2) {
            drawTwoHandMesh(latestHands[0], latestHands[1], time);
        }
    }

    if (isRecording) {
        drawCaptureFrame();
    }
}

function updateFpsAndQuality(time) {
    fpsFrameCounter += 1;
    const elapsed = time - fpsWindowStart;

    if (elapsed >= 1000) {
        displayFps = Math.round(fpsFrameCounter * 1000 / elapsed);
        fpsEl.textContent = String(displayFps);
        fpsFrameCounter = 0;
        fpsWindowStart = time;

        if (autoQualityToggle.checked && time - lastQualityAdjustment > 1200) {
            if (displayFps < 27) {
                qualityFactor = clamp(qualityFactor - 0.12, 0.42, 1);
            } else if (displayFps > 48) {
                qualityFactor = clamp(qualityFactor + 0.06, 0.42, 1);
            }
            lastQualityAdjustment = time;
        } else if (!autoQualityToggle.checked) {
            qualityFactor = 1;
        }

        qualityBadge.textContent = autoQualityToggle.checked
            ? `AUTO ${Math.round(qualityFactor * 100)}%`
            : "MAX QUALITY";
    }
}

// -------------------------------------------------------------
// Screenshot and recording
// -------------------------------------------------------------

function configureCaptureCanvas() {
    const maxLongSide = 1280;
    const aspect = viewportWidth / Math.max(viewportHeight, 1);

    if (aspect >= 1) {
        captureCanvas.width = maxLongSide;
        captureCanvas.height = Math.max(1, Math.round(maxLongSide / aspect));
    } else {
        captureCanvas.height = maxLongSide;
        captureCanvas.width = Math.max(1, Math.round(maxLongSide * aspect));
    }
}

function drawVideoCover(ctx, targetWidth, targetHeight) {
    if (!video.videoWidth || !video.videoHeight) return;

    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;
    const targetAspect = targetWidth / targetHeight;
    const sourceAspect = sourceWidth / sourceHeight;

    let sx = 0;
    let sy = 0;
    let sw = sourceWidth;
    let sh = sourceHeight;

    if (sourceAspect > targetAspect) {
        sw = sourceHeight * targetAspect;
        sx = (sourceWidth - sw) / 2;
    } else {
        sh = sourceWidth / targetAspect;
        sy = (sourceHeight - sh) / 2;
    }

    ctx.save();
    if (mirrorToggle.checked) {
        ctx.translate(targetWidth, 0);
        ctx.scale(-1, 1);
    }
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, targetWidth, targetHeight);
    ctx.restore();
}

function drawCaptureFrame() {
    if (!captureCanvas.width || !captureCanvas.height) configureCaptureCanvas();
    const w = captureCanvas.width;
    const h = captureCanvas.height;
    captureCtx.clearRect(0, 0, w, h);
    drawVideoCover(captureCtx, w, h);
    captureCtx.drawImage(fxCanvas, 0, 0, fxCanvas.width, fxCanvas.height, 0, 0, w, h);
}

function takeScreenshot() {
    if (!engineRunning || !video.videoWidth) {
        showToast("Start the camera first");
        return;
    }

    configureCaptureCanvas();
    drawCaptureFrame();

    captureCanvas.toBlob((blob) => {
        if (!blob) return;
        downloadBlob(blob, `ai-hand-energy-mesh-${Date.now()}.png`);
        showToast("Screenshot saved");
    }, "image/png");
}

function getSupportedRecordingMimeType() {
    if (typeof MediaRecorder === "undefined") return "";
    const candidates = [
        "video/webm;codecs=vp9",
        "video/webm;codecs=vp8",
        "video/webm"
    ];
    return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function toggleRecording() {
    if (isRecording) {
        mediaRecorder?.stop();
        return;
    }

    if (!engineRunning) {
        showToast("Start the camera first");
        return;
    }

    const mimeType = getSupportedRecordingMimeType();
    if (!mimeType) {
        showToast("Video recording is not supported in this browser");
        return;
    }

    configureCaptureCanvas();
    drawCaptureFrame();
    recordedChunks = [];

    const stream = captureCanvas.captureStream(30);
    mediaRecorder = new MediaRecorder(stream, {
        mimeType,
        videoBitsPerSecond: 6_000_000
    });

    mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) recordedChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: mimeType });
        downloadBlob(blob, `ai-hand-energy-mesh-${Date.now()}.webm`);
        recordedChunks = [];
        isRecording = false;
        recordButton.classList.remove("recording");
        recordIcon.textContent = "⏺";
        recordLabel.textContent = "Record";
        showToast("Recording saved");
    };

    mediaRecorder.start(250);
    isRecording = true;
    recordButton.classList.add("recording");
    recordIcon.textContent = "■";
    recordLabel.textContent = "Stop";
    showToast("Recording started");
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// -------------------------------------------------------------
// UI actions
// -------------------------------------------------------------

function setPaused(paused) {
    enginePaused = paused;
    pauseIcon.textContent = paused ? "▶" : "⏸";
    pauseLabel.textContent = paused ? "Resume" : "Pause";
    setStatus(paused ? "Paused" : "Tracking active", paused ? "idle" : "active");
}

function togglePause() {
    if (!engineRunning) return;
    setPaused(!enginePaused);
}

async function toggleFullscreen() {
    try {
        if (!document.fullscreenElement) {
            await document.documentElement.requestFullscreen();
        } else {
            await document.exitFullscreen();
        }
    } catch (error) {
        console.error("Fullscreen error:", error);
        showToast("Fullscreen is not available");
    }
}

function setUiHidden(hidden) {
    document.body.classList.toggle("ui-hidden", hidden);
}

function updateMirrorState() {
    video.classList.toggle("mirrored", mirrorToggle.checked);
    previousSmoothedHands = [null, null];
    previousSlotCenters = [null, null];
    handHistory = [];
}

function resetTrackingSmoothing() {
    previousSmoothedHands = [null, null];
    previousSlotCenters = [null, null];
    handHistory = [];
}

function bindRange(input, callback = null) {
    input.addEventListener("input", () => {
        updateControlLabels();
        if (callback) callback();
    });
}

bindRange(lineDensityEl);
bindRange(lineWidthEl);
bindRange(glowAmountEl);
bindRange(smoothingEl, resetTrackingSmoothing);
bindRange(trailLengthEl);

startButton.addEventListener("click", startCamera);
screenshotButton.addEventListener("click", takeScreenshot);
recordButton.addEventListener("click", toggleRecording);
pauseButton.addEventListener("click", togglePause);
fullscreenButton.addEventListener("click", toggleFullscreen);
hideUiButton.addEventListener("click", () => setUiHidden(true));
showUiButton.addEventListener("click", () => setUiHidden(false));
mirrorToggle.addEventListener("change", updateMirrorState);

autoQualityToggle.addEventListener("change", () => {
    if (!autoQualityToggle.checked) qualityFactor = 1;
    qualityBadge.textContent = autoQualityToggle.checked ? `AUTO ${Math.round(qualityFactor * 100)}%` : "MAX QUALITY";
});

collapsePanelButton.addEventListener("click", () => {
    const collapsed = controlPanel.classList.toggle("collapsed");
    collapsePanelButton.textContent = collapsed ? "+" : "−";
    collapsePanelButton.title = collapsed ? "Expand panel" : "Collapse panel";
});

window.addEventListener("resize", resizeCanvas);

window.addEventListener("keydown", (event) => {
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;

    if (event.code === "Space") {
        event.preventDefault();
        togglePause();
    } else if (event.key.toLowerCase() === "f") {
        toggleFullscreen();
    } else if (event.key.toLowerCase() === "h") {
        setUiHidden(!document.body.classList.contains("ui-hidden"));
    } else if (event.key.toLowerCase() === "s") {
        takeScreenshot();
    } else if (event.key.toLowerCase() === "r") {
        toggleRecording();
    }
});

window.addEventListener("beforeunload", () => {
    cancelInferenceLoop();
    stopCameraTracks();
    if (handsModel && typeof handsModel.close === "function") {
        try {
            handsModel.close();
        } catch (_) {
            // Ignore shutdown errors.
        }
    }
});

document.addEventListener("visibilitychange", () => {
    if (document.hidden && engineRunning && !enginePaused) {
        setPaused(true);
    }
});

// -------------------------------------------------------------
// Boot
// -------------------------------------------------------------

updateControlLabels();
updateMirrorState();
resizeCanvas();
setStatus("Ready");
renderRafId = requestAnimationFrame(renderFrame);

console.log("AI Hand Energy Mesh Studio loaded — line-only rendering enabled.");

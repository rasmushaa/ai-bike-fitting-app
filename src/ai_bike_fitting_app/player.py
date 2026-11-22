"""Client-side frame playback: seeking never runs pose inference again."""

import base64
import json

from .analysis import FPS


def player_html(frames: list[bytes], fps: int = FPS) -> str:
    images = json.dumps([base64.b64encode(frame).decode("ascii") for frame in frames])
    return """<!doctype html><html><head><style>
body {margin:0; font:14px system-ui,sans-serif; color:#31333f; background:white}
img {display:block; width:100%; height:auto; border-radius:6px}
.controls {display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:12px 0}
button {background:white; border:1px solid #ddd; border-radius:6px; padding:8px 12px; cursor:pointer; color:inherit}
button:disabled {opacity:.4; cursor:default}
input {width:100%; accent-color:#2563eb; margin:4px 0 12px}
</style></head><body>
<img id="frame" alt="Selected-side pose above blue knee and red elbow trace and distribution charts">
<div class="controls"><button id="previous">Previous frame</button>
<button id="play">Auto playback</button><button id="next">Next frame</button>
<span id="position" aria-live="off"></span></div>
<label for="seek">Frame</label><input id="seek" type="range" min="0" step="1" aria-label="Seek to frame">
<script>
const frames = FRAMES;
const fps = FPS_VALUE;
const image = document.getElementById('frame');
const seek = document.getElementById('seek');
const play = document.getElementById('play');
const previous = document.getElementById('previous');
const next = document.getElementById('next');
let index = 0, timer = null;
seek.max = frames.length - 1;
function pause() {clearInterval(timer); timer=null; play.textContent='Auto playback';}
function show(value) {
 index = Math.max(0, Math.min(frames.length-1, value));
 image.src = 'data:image/jpeg;base64,' + frames[index];
 seek.value = index;
 document.getElementById('position').textContent = `Frame ${index+1} / ${frames.length} · ${(index/fps).toFixed(2)} s`;
 previous.disabled = index === 0; next.disabled = index === frames.length-1;
}
previous.onclick = () => {pause(); show(index-1);};
next.onclick = () => {pause(); show(index+1);};
seek.oninput = () => {pause(); show(Number(seek.value));};
play.onclick = () => {
 if (timer !== null) {pause(); return;}
 if (index === frames.length-1) show(0);
 play.textContent='Pause';
 timer=setInterval(() => {if(index < frames.length-1) show(index+1); else pause();}, 1000/fps);
};
window.addEventListener('pagehide', pause);
show(0);
</script></body></html>""".replace("FRAMES", images).replace("FPS_VALUE", str(fps))

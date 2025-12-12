#!/usr/bin/env python3

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Bot-Run Sector Map</title>
  <style>
    :root {
      color-scheme: dark;
    }
    body {
      margin: 0;
      padding: 0;
      background: #050816;
      color: #e0e6ff;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }
    header {
      padding: 8px 16px;
      background: #090f24;
      border-bottom: 1px solid #1a2344;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 14px;
    }
    .tick {
      font-family: monospace;
    }
    main {
      display: flex;
      flex: 1;
      overflow: hidden;
    }
    #map-container {
      flex: 2.2;
      padding: 12px;
      display: flex;
      flex-direction: column;
    }
    #canvas-wrapper {
      flex: 1;
      background: radial-gradient(circle at 20% 0%, #151b37 0, #050816 45%);
      border-radius: 12px;
      box-shadow: 0 0 30px rgba(0,0,0,0.7);
      position: relative;
      overflow: hidden;
    }
    #mapCanvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    #legend {
      margin-top: 8px;
      font-size: 13px;
      color: #a2afd8;
    }
    #legend span {
      margin-right: 16px;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    .badge {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }
    .badge-E { background: #ff5555; }
    .badge-P { background: #f1fa8c; }
    .badge-T { background: #8be9fd; }
    .badge-empty { background: #44475a; }
    .badge-siege {
      background: transparent;
      border: 2px solid #ff7b7b;
      box-shadow: 0 0 6px rgba(255, 123, 123, 0.6);
    }

    #events-container {
      flex: 1;
      padding: 12px;
      border-left: 1px solid #1a2344;
      display: flex;
      flex-direction: column;
    }
    #events-title {
      font-size: 13px;
      margin-bottom: 6px;
      color: #a2afd8;
    }
    #events {
      flex: 1;
      background: #050b18;
      border-radius: 12px;
      padding: 8px 12px;
      margin: 0;
      list-style: none;
      overflow-y: auto;
      font-size: 13px;
    }
    #events li {
      padding: 3px 0;
    }
    #status {
      font-size: 12px;
      color: #8be9fd;
    }
  </style>
</head>
<body>
  <header>
    <div>Bot-Run Sector Map</div>
    <div>
      Tick: <span class="tick" id="tick">0</span>
      &nbsp;|&nbsp;
      <span id="status">Connecting...</span>
    </div>
  </header>

  <main>
    <section id="map-container">
      <div id="canvas-wrapper">
        <canvas id="mapCanvas"></canvas>
      </div>
      <div id="legend">
        <span><span class="badge badge-E"></span> Empire</span>
        <span><span class="badge badge-P"></span> Pirates</span>
        <span><span class="badge badge-T"></span> Traders</span>
        <span><span class="badge badge-empty"></span> Unclaimed</span>
        <span><span class="badge badge-siege"></span> Under siege</span>
      </div>
    </section>

    <section id="events-container">
      <div id="events-title">Recent Events</div>
      <ul id="events"></ul>
    </section>
  </main>

  <script>
    const tickEl = document.getElementById("tick");
    const statusEl = document.getElementById("status");
    const canvas = document.getElementById("mapCanvas");
    const eventsEl = document.getElementById("events");

    function resizeCanvas() {
      const wrapper = document.getElementById("canvas-wrapper");
      const rect = wrapper.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();

    function render(data) {
      tickEl.textContent = data.tick;

      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const padding = 30;
      const innerW = w - 2 * padding;
      const innerH = h - 2 * padding;

      const byId = {};
      data.systems.forEach(s => byId[s.id] = s);

      const highlights = new Set(data.highlight_ids || []);

      function proj(sys) {
        return {
          x: padding + sys.x * innerW,
          y: padding + sys.y * innerH,
        };
      }

      // warp lanes
      ctx.save();
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(120, 140, 220, 0.4)";
      data.lanes.forEach(([a, b]) => {
        const sa = byId[a];
        const sb = byId[b];
        if (!sa || !sb) return;
        const pa = proj(sa);
        const pb = proj(sb);
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
      });
      ctx.restore();

      // systems
      data.systems.forEach(sys => {
        const p = proj(sys);
        const r = 6;
        let color = "#44475a";
        if (sys.owner === "E") color = "#ff5555";
        else if (sys.owner === "P") color = "#f1fa8c";
        else if (sys.owner === "T") color = "#8be9fd";

        const isHighlight = highlights.has(sys.id);
        const isBesieged = !!sys.is_besieged;

        // subtle background glow
        ctx.save();
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 3, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0,0,0,0.4)";
        ctx.fill();
        ctx.restore();

        // siege indicator ring (pulsing)
        if (isBesieged) {
          const t = (typeof performance !== "undefined" ? performance.now() : Date.now());
          const pulse = 0.6 + 0.4 * Math.sin(t / 400);
          ctx.save();
          ctx.beginPath();
          ctx.arc(p.x, p.y, r + 7 + pulse * 2, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(255, 123, 123, 0.95)";
          ctx.lineWidth = 2.5;
          ctx.shadowBlur = 6;
          ctx.shadowColor = "rgba(255, 123, 123, 0.9)";
          ctx.stroke();
          ctx.restore();
        }

        // highlight ring
        if (isHighlight) {
          ctx.save();
          ctx.beginPath();
          ctx.arc(p.x, p.y, r + 6, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
          ctx.lineWidth = 2;
          ctx.stroke();
          ctx.restore();
        }

        // core circle
        ctx.save();
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.restore();

        // siege marker glyph
        if (isBesieged) {
          ctx.save();
          ctx.fillStyle = "#ff7b7b";
          ctx.font = "bold 11px monospace";
          ctx.textAlign = "center";
          ctx.textBaseline = "bottom";
          ctx.fillText("!", p.x, p.y - r - 2);
          ctx.restore();
        }

        // id label
        ctx.save();
        ctx.fillStyle = "rgba(220, 230, 255, 0.75)";
        ctx.font = "10px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(sys.id, p.x, p.y + r + 2);
        ctx.restore();
      });

      // events
      eventsEl.innerHTML = "";
      data.events.forEach(ev => {
        const li = document.createElement("li");
        li.textContent = ev;
        eventsEl.appendChild(li);
      });
      eventsEl.scrollTop = eventsEl.scrollHeight;
    }

    function connect() {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const wsUrl = `${protocol}://${window.location.host}/ws`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        statusEl.textContent = "Connected";
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        render(data);
      };

      ws.onclose = () => {
        statusEl.textContent = "Disconnected. Reconnecting...";
        setTimeout(connect, 2000);
      };

      ws.onerror = () => {
        statusEl.textContent = "Error. Reconnecting...";
        ws.close();
      };
    }

    connect();
  </script>
</body>
</html>
"""

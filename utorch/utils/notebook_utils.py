"""Interactive 28x28 drawing canvas for VS Code, JupyterLab, and Colab."""

from IPython.display import HTML, display


def show_drawing_canvas() -> None:
    """Render an interactive 28x28 drawing canvas."""
    html_code = """
    <div style="background: #1e1e1e; padding: 15px; border-radius: 8px; display: inline-block; font-family: sans-serif;">
      <h4 style="color: #eee; margin: 0 0 10px 0;">Draw a digit (0-9) with your mouse:</h4>
      <canvas id="cvs" width="280" height="280" style="background: black; border: 2px solid #ee4c2c; cursor: crosshair; border-radius: 4px;"></canvas>
      <div style="margin-top: 10px;">
        <button onclick="clearCanvas()" style="padding: 6px 14px; font-weight: bold; background: #444; color: white; border: none; border-radius: 4px; cursor: pointer;">Clear</button>
        <button onclick="captureCanvas()" style="padding: 6px 14px; font-weight: bold; background: #ee4c2c; color: white; border: none; border-radius: 4px; cursor: pointer; margin-left: 8px;">1. Process Image</button>
      </div>
      <div id="output_box" style="margin-top: 12px; display: none;">
        <span style="color: #aaa; font-size: 12px;">Click below to copy pixels to clipboard, then paste in the next cell:</span><br/>
        <textarea id="pixel_txt" rows="3" cols="35" style="width: 270px; background: #111; color: #4af; font-size: 11px; border: 1px solid #333; margin-top: 4px;"></textarea>
        <br/>
        <button onclick="copyToClipboard()" style="margin-top: 6px; padding: 5px 12px; font-size: 12px; background: #2ca02c; color: white; border: none; border-radius: 4px; cursor: pointer;">Copy Code</button>
      </div>
    </div>

    <script>
      (function() {
        const c = document.getElementById('cvs');
        const ctx = c.getContext('2d');
        let drawing = false;

        ctx.strokeStyle = "white";
        ctx.lineWidth = 22;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        function getPos(e) {
          const r = c.getBoundingClientRect();
          return { x: e.clientX - r.left, y: e.clientY - r.top };
        }

        c.onmousedown = (e) => { drawing = true; const p = getPos(e); ctx.beginPath(); ctx.moveTo(p.x, p.y); };
        window.onmouseup = () => { drawing = false; };
        c.onmousemove = (e) => {
          if (!drawing) return;
          const p = getPos(e);
          ctx.lineTo(p.x, p.y);
          ctx.stroke();
        };

        window.clearCanvas = function() {
          ctx.fillStyle = "black";
          ctx.fillRect(0, 0, 280, 280);
          document.getElementById('output_box').style.display = "none";
        };
        clearCanvas();

        window.captureCanvas = function() {
          // Downscale to 28x28
          const s = document.createElement('canvas');
          s.width = 28; s.height = 28;
          const sctx = s.getContext('2d');
          sctx.drawImage(c, 0, 0, 28, 28);

          const img = sctx.getImageData(0, 0, 28, 28).data;
          const vals = [];
          for (let i = 0; i < img.length; i += 4) {
            vals.push(Math.round((img[i] / 255.0) * 100) / 100);
          }

          const code = "drawn_digit = np.array(" + JSON.stringify(vals) + ").reshape(1, 784)";
          const box = document.getElementById('output_box');
          const txt = document.getElementById('pixel_txt');
          txt.value = code;
          box.style.display = "block";
        };

        window.copyToClipboard = function() {
          const txt = document.getElementById('pixel_txt');
          txt.select();
          document.execCommand('copy');
          alert("Copied code to clipboard! Paste and run it in the next cell.");
        };
      })();
    </script>
    """
    display(HTML(html_code))  # type: ignore[no-untyped-call]

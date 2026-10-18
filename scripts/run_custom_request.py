#!/usr/bin/env python3
import argparse
import html
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str) -> str:
    out = []
    last_dash = False
    for char in value.lower():
        if char.isalnum():
            out.append(char)
            last_dash = False
        elif not last_dash:
            out.append("-")
            last_dash = True
    return "".join(out).strip("-") or "run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://vllm.test")
    parser.add_argument("--endpoint", default=None, choices=["chat", "completion"])
    parser.add_argument("--request-file", required=True)
    parser.add_argument("--run-name", default="landing-page-custom")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--custom-label", default="custom")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "900")),
    )
    return parser.parse_args()


def normalize_timeout_seconds(timeout_seconds: float) -> float | None:
    if timeout_seconds <= 0:
        return None
    return timeout_seconds


def normalize_artifact_extension(extension: str | None) -> str | None:
    if not extension:
        return None
    if extension.startswith("."):
        return extension
    return f".{extension}"


def extract_json_object(raw_text: str) -> dict | None:
    stripped = raw_text.strip()
    candidates = [stripped]
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidates.insert(0, "\n".join(lines).strip())
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(stripped[first_brace : last_brace + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def sanitize_ascii_text(value: object, default: str, max_length: int) -> str:
    if not isinstance(value, str):
        return default
    cleaned = "".join(char for char in value if 32 <= ord(char) <= 126)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return default
    return cleaned[:max_length]


def sanitize_hex_color(value: object, default: str) -> str:
    if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return value.lower()
    return default


def render_pong_html(spec: dict | None) -> tuple[str, dict]:
    spec = spec or {}
    safe_spec = {
        "title": sanitize_ascii_text(spec.get("title"), "PONG", 18),
        "intro": "First to 5 points. Move with W/S or Arrow keys.",
        "startLabel": sanitize_ascii_text(spec.get("startLabel"), "Start Game", 20),
        "restartLabel": sanitize_ascii_text(
            spec.get("restartLabel"), "Restart Game", 20
        ),
        "winLabel": sanitize_ascii_text(spec.get("winLabel"), "YOU WIN", 24),
        "loseLabel": sanitize_ascii_text(spec.get("loseLabel"), "AI WINS", 24),
        "accent": sanitize_hex_color(spec.get("accent"), "#8df0c8"),
        "bgTop": sanitize_hex_color(spec.get("bgTop"), "#20252f"),
        "bgBottom": sanitize_hex_color(spec.get("bgBottom"), "#0d1015"),
    }

    title_html = html.escape(safe_spec["title"])
    intro_html = html.escape(safe_spec["intro"])
    start_label_html = html.escape(safe_spec["startLabel"])
    accent = safe_spec["accent"]
    bg_top = safe_spec["bgTop"]
    bg_bottom = safe_spec["bgBottom"]
    title_js = json.dumps(safe_spec["title"])
    intro_js = json.dumps(safe_spec["intro"])
    start_label_js = json.dumps(safe_spec["startLabel"])
    restart_label_js = json.dumps(safe_spec["restartLabel"])
    win_label_js = json.dumps(safe_spec["winLabel"])
    lose_label_js = json.dumps(safe_spec["loseLabel"])
    accent_js = json.dumps(accent)

    rendered = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"UTF-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>{title_html}</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;min-height:100vh;display:grid;place-items:center;background:radial-gradient(circle at top,{bg_top},{bg_bottom} 62%);font-family:Arial,sans-serif;color:#eaf2ff}}
.wrap{{position:relative;width:min(92vw,800px)}}
canvas{{display:block;width:100%;height:auto;background:#081018;border:2px solid #32455f;border-radius:12px;box-shadow:0 18px 50px rgba(0,0,0,.35)}}
.hud{{position:absolute;inset:16px 18px auto;display:flex;justify-content:space-between;align-items:center;font-weight:700;letter-spacing:.08em;font-size:clamp(12px,2vw,16px);pointer-events:none}}
.score{{font-size:clamp(24px,4vw,38px);font-weight:800}}
.overlay{{position:absolute;inset:0;display:grid;place-items:center;background:rgba(3,6,10,.72);border-radius:12px}}
.panel{{padding:24px 28px;border:1px solid rgba(255,255,255,.12);border-radius:14px;background:rgba(18,24,34,.92);text-align:center;max-width:min(84vw,420px)}}
.panel h1{{margin:0 0 10px;font-size:clamp(26px,5vw,42px);letter-spacing:.12em}}
.panel p{{margin:0 0 16px;color:#b7c7df;line-height:1.45}}
button{{border:0;border-radius:999px;padding:12px 22px;background:{accent};color:#04211a;font-weight:800;cursor:pointer}}
</style>
</head>
<body>
<div class=\"wrap\">
<div class=\"hud\"><div>{title_html}</div><div class=\"score\"><span id=\"leftScore\">0</span> : <span id=\"rightScore\">0</span></div><div>W/S or Arrows</div></div>
<canvas id=\"game\" width=\"800\" height=\"500\"></canvas>
<div class=\"overlay\" id=\"overlay\"><div class=\"panel\"><h1 id=\"title\">{title_html}</h1><p id=\"message\">{intro_html}</p><button id=\"action\">{start_label_html}</button></div></div>
</div>
<script>
const copy={{title:{title_js},intro:{intro_js},startLabel:{start_label_js},restartLabel:{restart_label_js},winLabel:{win_label_js},loseLabel:{lose_label_js},accent:{accent_js}}};
const canvas=document.getElementById("game");
const ctx=canvas.getContext("2d");
const overlay=document.getElementById("overlay");
const titleEl=document.getElementById("title");
const messageEl=document.getElementById("message");
const actionBtn=document.getElementById("action");
const leftScoreEl=document.getElementById("leftScore");
const rightScoreEl=document.getElementById("rightScore");
const width=canvas.width,height=canvas.height,paddleWidth=12,paddleHeight=96,paddleInset=22,paddleSpeed=7,aiSpeed=5.2,ballRadius=8,maxScore=5;
const keys={{up:false,down:false}};
const player={{x:paddleInset,y:(height-paddleHeight)/2,w:paddleWidth,h:paddleHeight}};
const ai={{x:width-paddleInset-paddleWidth,y:(height-paddleHeight)/2,w:paddleWidth,h:paddleHeight}};
const ball={{x:width/2,y:height/2,vx:0,vy:0,r:ballRadius,speed:5}};
let leftScore=0,rightScore=0,running=false;
function clamp(value,min,max){{return Math.max(min,Math.min(max,value));}}
function setOverlay(title,message,buttonLabel){{titleEl.textContent=title;messageEl.textContent=message;actionBtn.textContent=buttonLabel;overlay.hidden=false;}}
function updateScore(){{leftScoreEl.textContent=String(leftScore);rightScoreEl.textContent=String(rightScore);}}
function resetPaddles(){{player.y=(height-paddleHeight)/2;ai.y=(height-paddleHeight)/2;}}
function resetBall(direction=1){{ball.x=width/2;ball.y=height/2;ball.speed=5;const angle=Math.random()*1.2-.6;ball.vx=Math.cos(angle)*ball.speed*direction;ball.vy=Math.sin(angle)*ball.speed;}}
function startGame(){{if(leftScore>=maxScore||rightScore>=maxScore){{leftScore=0;rightScore=0;updateScore();resetPaddles();}}overlay.hidden=true;running=true;resetBall(Math.random()>.5?1:-1);}}
function endGame(playerWon){{running=false;setOverlay(playerWon?copy.winLabel:copy.loseLabel,"Press the button or R to play again.",copy.restartLabel);}}
function scorePoint(playerWon){{if(playerWon)leftScore++;else rightScore++;updateScore();if(leftScore>=maxScore||rightScore>=maxScore){{endGame(leftScore>rightScore);return;}}resetPaddles();resetBall(playerWon?-1:1);}}
function hitPaddle(paddle,isLeft){{const relative=(ball.y-paddle.y)/(paddle.h||1)-.5;ball.speed=Math.min(ball.speed+.35,10.5);ball.vx=(isLeft?1:-1)*ball.speed;ball.vy=relative*10;ball.x=isLeft?paddle.x+paddle.w+ball.r:paddle.x-ball.r;}}
function update(){{if(!running)return;if(keys.up)player.y-=paddleSpeed;if(keys.down)player.y+=paddleSpeed;player.y=clamp(player.y,0,height-player.h);const aiCenter=ai.y+ai.h/2;if(aiCenter<ball.y-10)ai.y+=aiSpeed;else if(aiCenter>ball.y+10)ai.y-=aiSpeed;ai.y=clamp(ai.y,0,height-ai.h);ball.x+=ball.vx;ball.y+=ball.vy;if(ball.y-ball.r<=0){{ball.y=ball.r;ball.vy*=-1;}}else if(ball.y+ball.r>=height){{ball.y=height-ball.r;ball.vy*=-1;}}if(ball.vx<0&&ball.x-ball.r<=player.x+player.w&&ball.y>=player.y&&ball.y<=player.y+player.h)hitPaddle(player,true);else if(ball.vx>0&&ball.x+ball.r>=ai.x&&ball.y>=ai.y&&ball.y<=ai.y+ai.h)hitPaddle(ai,false);if(ball.x+ball.r<0)scorePoint(false);else if(ball.x-ball.r>width)scorePoint(true);}}
function draw(){{ctx.fillStyle="#081018";ctx.fillRect(0,0,width,height);ctx.strokeStyle="#213348";ctx.lineWidth=2;ctx.setLineDash([10,12]);ctx.beginPath();ctx.moveTo(width/2,12);ctx.lineTo(width/2,height-12);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=copy.accent;ctx.fillRect(player.x,player.y,player.w,player.h);ctx.fillRect(ai.x,ai.y,ai.w,ai.h);ctx.beginPath();ctx.arc(ball.x,ball.y,ball.r,0,Math.PI*2);ctx.fill();}}
function frame(){{update();draw();requestAnimationFrame(frame);}}
document.addEventListener("keydown",event=>{{const key=event.key.toLowerCase();if(key==="arrowup"||key==="w")keys.up=true;if(key==="arrowdown"||key==="s")keys.down=true;if(key==="r")startGame();}});
document.addEventListener("keyup",event=>{{const key=event.key.toLowerCase();if(key==="arrowup"||key==="w")keys.up=false;if(key==="arrowdown"||key==="s")keys.down=false;}});
actionBtn.addEventListener("click",startGame);
updateScore();
draw();
frame();
</script>
</body>
</html>
"""
    return rendered, safe_spec


def main() -> int:
    args = parse_args()
    request_path = Path(args.request_file)
    request_obj = json.loads(request_path.read_text())
    request_meta = {}
    for key in list(request_obj.keys()):
        if key.startswith("_"):
            request_meta[key] = request_obj.pop(key)
    endpoint = args.endpoint or request_meta.get("_endpoint") or "chat"

    run_id = f"{stamp_utc()}-{slugify(args.run_name)}"
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if endpoint == "chat":
        api_path = "/v1/chat/completions"
        result_name = f"01-{args.custom_label}-chat.json"
    else:
        api_path = "/v1/completions"
        result_name = f"01-{args.custom_label}-completion.json"

    url = args.base_url.rstrip("/") + api_path
    payload = json.dumps(request_obj).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
        method="POST",
    )

    context = ssl._create_unverified_context() if args.insecure else None
    started = time.time()
    http_code = 0
    raw_body = ""
    response_obj = None
    parse_error = False

    request_timeout = normalize_timeout_seconds(args.timeout_seconds)

    try:
        with urllib.request.urlopen(
            request,
            timeout=request_timeout,
            context=context,
        ) as response:
            http_code = response.status
            raw_body = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        http_code = error.code
        raw_body = error.read().decode("utf-8", "replace")
    except Exception as error:
        elapsed = time.time() - started
        result = {
            "run_id": run_id,
            "created_at": timestamp_utc(),
            "case": args.custom_label,
            "endpoint": endpoint,
            "base_url": args.base_url,
            "model_name": request_obj.get("model"),
            "prompt_file": None,
            "request_file": request_path.name,
            "request": request_obj,
            "response": None,
            "response_raw": f"{type(error).__name__}: {error}",
            "diagnostics": {
                "http_code": http_code,
                "time_total_seconds": elapsed,
                "request_timeout_seconds": args.timeout_seconds,
                "parse_error": True,
            },
        }
        (run_dir / result_name).write_text(json.dumps(result, indent=2) + "\n")
        summary = {
            "run_id": run_id,
            "created_at": timestamp_utc(),
            "results": [
                {
                    "case": args.custom_label,
                    "endpoint": endpoint,
                    "http_code": http_code,
                    "time_total_seconds": elapsed,
                    "finish_reason": None,
                    "bang_count": None,
                    "has_repeated_bangs": False,
                    "content_preview": None,
                }
            ],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(run_dir)
        return 1

    elapsed = time.time() - started
    try:
        response_obj = json.loads(raw_body)
    except json.JSONDecodeError:
        parse_error = True

    if parse_error:
        result = {
            "run_id": run_id,
            "created_at": timestamp_utc(),
            "case": args.custom_label,
            "endpoint": endpoint,
            "base_url": args.base_url,
            "model_name": request_obj.get("model"),
            "prompt_file": None,
            "request_file": request_path.name,
            "request": request_obj,
            "response": None,
            "response_raw": raw_body,
            "diagnostics": {
                "http_code": http_code,
                "time_total_seconds": elapsed,
                "request_timeout_seconds": args.timeout_seconds,
                "parse_error": True,
            },
        }
        summary_row = {
            "case": args.custom_label,
            "endpoint": endpoint,
            "http_code": http_code,
            "time_total_seconds": elapsed,
            "finish_reason": None,
            "bang_count": None,
            "has_repeated_bangs": False,
            "content_preview": None,
        }
    else:
        choice0 = (response_obj.get("choices") or [{}])[0]
        message = choice0.get("message") or {}
        text_content = message.get("content") or choice0.get("text") or ""
        reasoning_text = message.get("reasoning") or choice0.get("reasoning")
        bang_count = text_content.count("!")
        result = {
            "run_id": run_id,
            "created_at": timestamp_utc(),
            "case": args.custom_label,
            "endpoint": endpoint,
            "base_url": args.base_url,
            "model_name": request_obj.get("model"),
            "prompt_file": None,
            "request_file": request_path.name,
            "request": request_obj,
            "response": response_obj,
            "extracted": {
                "reasoning": reasoning_text,
                "content": text_content,
                "content_preview": text_content[:240],
                "finish_reason": choice0.get("finish_reason"),
                "usage": response_obj.get("usage"),
            },
            "diagnostics": {
                "http_code": http_code,
                "time_total_seconds": elapsed,
                "request_timeout_seconds": args.timeout_seconds,
                "bang_count": bang_count,
                "has_repeated_bangs": "!!!!!!!!" in text_content,
                "content_length": len(text_content),
            },
        }
        summary_row = {
            "case": args.custom_label,
            "endpoint": endpoint,
            "http_code": http_code,
            "time_total_seconds": elapsed,
            "finish_reason": choice0.get("finish_reason"),
            "bang_count": bang_count,
            "has_repeated_bangs": "!!!!!!!!" in text_content,
            "content_preview": text_content[:240],
        }

        renderer_name = request_meta.get("_renderer")
        artifact_extension = normalize_artifact_extension(
            request_meta.get("_artifact_extension")
        )
        artifact_prefix = request_meta.get("_artifact_prefix")
        artifact_suffix = request_meta.get("_artifact_suffix", "")
        rendered_artifact = None
        if renderer_name == "pong_html":
            rendered_artifact, rendered_spec = render_pong_html(
                extract_json_object(text_content)
            )
            artifact_extension = ".html"
            result["rendered_spec"] = rendered_spec
        if artifact_extension and artifact_prefix is None and endpoint == "completion":
            artifact_prefix = request_obj.get("prompt", "")
        if not artifact_extension and endpoint == "chat" and text_content.lstrip().startswith("<!DOCTYPE html>"):
            artifact_extension = ".html"
            artifact_prefix = ""
            artifact_suffix = ""
        if artifact_extension and (text_content or rendered_artifact is not None):
            artifact_name = f"02-{args.custom_label}{artifact_extension}"
            artifact_text = rendered_artifact
            if artifact_text is None:
                artifact_text = f"{artifact_prefix or ''}{text_content}"
                if artifact_suffix and not artifact_text.rstrip().endswith(artifact_suffix.strip()):
                    artifact_text += artifact_suffix
            (run_dir / artifact_name).write_text(artifact_text)
            result["artifact_file"] = artifact_name
            summary_row["artifact_file"] = artifact_name

    (run_dir / result_name).write_text(json.dumps(result, indent=2) + "\n")
    summary = {
        "run_id": run_id,
        "created_at": timestamp_utc(),
        "results": [summary_row],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

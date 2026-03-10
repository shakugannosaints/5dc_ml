# ml/ai_server.py
"""
AI Server for 5D Chess: provides HTTP API for the frontend UI to play against
the trained neural network.

Architecture:
  - Flask HTTP server with CORS
  - Loads latest checkpoint automatically
  - Receives game state (PGN), returns AI move
  - Runs alongside the WASM-based UI

API Endpoints:
  GET  /api/status         → server status + model info
  POST /api/move           → {pgn: "..."} → {moves: [...], value: float}
  POST /api/new_game       → {variant: "..."} → {pgn: "...", boards: [...]}
"""

import sys
import os
import json
import time
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine

from ml.config import TrainingConfig, SMALL_CONFIG, STANDARD_CONFIG
from ml.models.agent import Agent
from ml.trainer import Trainer


class AIPlayer:
    """Wraps the Agent model for interactive play."""

    def __init__(self, checkpoint_dir: str = "checkpoints", device: str = "auto"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.agent = None
        self.cfg = None
        self.device = None
        self.model_info = {"loaded": False, "checkpoint": None, "epoch": 0}

        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._load_latest()

    def _find_latest_checkpoint(self) -> Path | None:
        """Find the latest checkpoint."""
        if not self.checkpoint_dir.exists():
            return None
        # Epoch checkpoints
        ckpts = sorted(self.checkpoint_dir.glob("agent_epoch_*.pt"))
        if ckpts:
            return ckpts[-1]
        # Final checkpoint
        final = self.checkpoint_dir / "agent_final.pt"
        if final.exists():
            return final
        return None

    def _load_latest(self):
        """Load the latest checkpoint."""
        ckpt_path = self._find_latest_checkpoint()
        if ckpt_path is None:
            print("[AIPlayer] No checkpoint found. AI will play randomly.")
            self.model_info = {"loaded": False, "checkpoint": None, "epoch": 0}
            return

        print(f"[AIPlayer] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Reconstruct config from checkpoint
        self.cfg = ckpt.get('config', STANDARD_CONFIG)
        self.agent = Agent(self.cfg).to(self.device)
        self.agent.load_state_dict(ckpt['model_state_dict'])
        self.agent.eval()

        epoch = ckpt.get('epoch', 0)
        self.model_info = {
            "loaded": True,
            "checkpoint": str(ckpt_path),
            "epoch": epoch,
            "total_games": ckpt.get('total_games', 0),
            "device": str(self.device),
            "variant": self.cfg.variant,
        }
        print(f"[AIPlayer] Model loaded: epoch {epoch}, device {self.device}")

    def reload(self):
        """Reload the latest checkpoint (hot reload)."""
        self._load_latest()

    def get_ai_move(self, pgn: str, temperature: float = 0.5) -> dict:
        """
        Given a PGN string representing current game state, return AI's move.
        
        Returns:
            {
                'success': bool,
                'moves': [{'from': {x,y,t,l}, 'to': {x,y,t,l}}, ...],
                'value': float,
                'message': str
            }
        """
        try:
            # Strip custom FEN lines from PGN: the WASM engine exports lines
            # like "[nbrk/3p*/P*3/KRBN:0:1:w]" which the Python engine cannot
            # parse back.  The "[Board ...]" tag is sufficient for the engine
            # to reconstruct the initial position, so we simply remove any
            # header line whose value contains '/' (a FEN separator).
            import re
            cleaned_lines = []
            for line in pgn.splitlines():
                # Match PGN header lines like [TagName "value"]
                m = re.match(r'^\[(\w+)\s+"(.*)"\]\s*$', line)
                if m:
                    tag_name, tag_value = m.group(1), m.group(2)
                    # Skip lines that look like inline FEN (contain '/')
                    if '/' in tag_value and tag_name not in ('Board', 'Mode', 'Site', 'Event', 'Date', 'White', 'Black', 'Result'):
                        print(f"[AIPlayer] Stripping FEN header: {line}")
                        continue
                # Also strip bare FEN lines (no tag, just "[fen/stuff]")
                if re.match(r'^\[[\w\d\*\./ :]+\]\s*$', line) and '/' in line:
                    print(f"[AIPlayer] Stripping bare FEN line: {line}")
                    continue
                cleaned_lines.append(line)
            pgn = '\n'.join(cleaned_lines)

            # Create state from PGN
            print(f"[AIPlayer] get_ai_move called, PGN ({len(pgn)} chars):")
            print(pgn[:500])
            print("---")
            state = engine.create_state_from_pgn(pgn)
            if state is None:
                return {'success': False, 'moves': [], 'value': 0.0,
                        'message': 'Failed to parse PGN'}

            # Check game status
            match_status = state.get_match_status()
            if match_status != engine.match_status_t.PLAYING:
                return {'success': False, 'moves': [], 'value': 0.0,
                        'message': f'Game already ended: {match_status}'}

            if self.agent is not None and self.model_info['loaded']:
                # Use neural network
                with torch.no_grad():
                    action, log_prob, entropy, value = self.agent.select_action(
                        state, engine, temperature=temperature
                    )
                value_est = value.item() if isinstance(value, torch.Tensor) else float(value)
            else:
                # Fallback: random move
                action = engine.random_action(state)
                value_est = 0.0

            if action is None:
                return {'success': False, 'moves': [], 'value': value_est,
                        'message': 'No legal action found'}

            # Convert action to move list for frontend
            moves = []
            for mv in action.get_moves():
                from_v4 = mv.get_from()
                to_v4 = mv.get_to()
                print(f"[AIPlayer] Move: from=({from_v4.x()},{from_v4.y()},{from_v4.t()},{from_v4.l()}) "
                      f"to=({to_v4.x()},{to_v4.y()},{to_v4.t()},{to_v4.l()}) raw={mv.to_string()}")
                moves.append({
                    'from': {
                        'x': from_v4.x(),
                        'y': from_v4.y(),
                        't': from_v4.t(),
                        'l': from_v4.l(),
                    },
                    'to': {
                        'x': to_v4.x(),
                        'y': to_v4.y(),
                        't': to_v4.t(),
                        'l': to_v4.l(),
                    }
                })

            return {
                'success': True,
                'moves': moves,
                'value': round(value_est, 4),
                'message': f'{len(moves)} move(s) selected'
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[AIPlayer] ERROR processing PGN ({len(pgn)} chars):")
            print(pgn[:1000])
            return {'success': False, 'moves': [], 'value': 0.0,
                    'message': f'Error: {str(e)}'}


# ---- Flask App ----

def create_app(checkpoint_dir: str = "checkpoints", ui_dir: str | None = None):
    app = Flask(__name__)
    CORS(app)

    ai_player = AIPlayer(checkpoint_dir=checkpoint_dir)

    # Serve UI static files if ui_dir is provided
    if ui_dir:
        @app.route('/')
        def serve_index():
            return send_from_directory(ui_dir, 'index.html')

        @app.route('/<path:path>')
        def serve_static(path):
            return send_from_directory(ui_dir, path)

    @app.route('/api/status', methods=['GET'])
    def status():
        return jsonify({
            'status': 'online',
            'model': ai_player.model_info,
        })

    @app.route('/api/move', methods=['POST'])
    def get_move():
        data = request.get_json()
        if not data or 'pgn' not in data:
            return jsonify({'success': False, 'message': 'Missing pgn field'}), 400

        pgn = data['pgn']
        temperature = data.get('temperature', 0.5)
        result = ai_player.get_ai_move(pgn, temperature)
        return jsonify(result)

    @app.route('/api/reload', methods=['POST'])
    def reload_model():
        ai_player.reload()
        return jsonify({'status': 'reloaded', 'model': ai_player.model_info})

    @app.after_request
    def add_headers(response):
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
        return response

    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="5D Chess AI Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--ui-dir", type=str, default=None,
                        help="Path to UI directory to serve (optional)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    app = create_app(
        checkpoint_dir=args.checkpoint_dir,
        ui_dir=args.ui_dir,
    )
    print(f"\n[AI Server] Starting on http://{args.host}:{args.port}")
    print(f"[AI Server] API endpoints:")
    print(f"  GET  /api/status  - Server status")
    print(f"  POST /api/move    - Get AI move (body: {{pgn: '...'}})")
    print(f"  POST /api/reload  - Reload latest model\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

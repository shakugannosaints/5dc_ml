// main.js

import { UI } from 'ui';
import ChessBoardCanvasBase from 'draw';


let worker = new Worker(window.worker_path, { type: 'module' });

// State management for move selection
let clicking = false;
let clickedPos = null;
let generatedMoves = [];
let presentC;
let nextOptions = [];
let showPhantom = true;

// ================= AI Opponent State =================
const AI_API_BASE = '';  // same origin, proxied by serve.py
let aiConnected = false;
let aiThinking = false;
let aiWaitingForExport = false;

class ChessBoardCanvas extends ChessBoardCanvasBase {
    onClickSquare(l, t, c, x, y) {
        handle_click({l: l, t: t, c: c, x: x, y: y});
    }
    
    onRightClickSquare() {
        deselect();
    }
}

window.chessBoardCanvas = new ChessBoardCanvas('canvas');

function handle_click(pos) {
    if (clicking) {
        return;
    }
    if(pos.c !== presentC) {
        if (clickedPos !== null) {
            deselect();
        }
        return; // Ignore clicks on other colors
    }
    if (clickedPos !== null) {
        let from = clickedPos;
        let to = pos;
        clickedPos = null;
        if (generatedMoves.some(mv => mv.l === to.l && mv.t === to.t && mv.x === to.x && mv.y === to.y)) {
            worker.postMessage({type: 'apply_move', from: from, to: to});
            generatedMoves = [];
        } else {
            deselect();
        }
    } else {
        clicking = true;
        worker.postMessage({type: 'gen_move_if_playable', pos: pos});
        clickedPos = pos;
    }
}

function deselect() {
    clickedPos = null;
    generatedMoves = [];
    worker.postMessage({type: 'view'});
}

function disablePhantom() {
    showPhantom = false;
    UI.setHudLight(false);
}

function addHighlight(data, color, field, values) {
  if (!Array.isArray(data.highlights)) {
    data.highlights = [];
  }
  let colorBlock = data.highlights.find(h => h.color === color);
  if (!colorBlock) {
    data.highlights.push({
      color,
      [field]: [...values],
    });
  } else {
    if (!Array.isArray(colorBlock[field])) {
      colorBlock[field] = [];
    }
    colorBlock[field].push(...values);
  }
}


worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'ready') {
        // Send current export options to worker on startup
        worker.postMessage({
            type: 'update_settings', 
            settings: UI.getSettings()
        });
        worker.postMessage({type: 'load', pgn: '[Board "Very Small - Open"]\n[Mode "5D"]'});
    }
    else if (msg.type === 'engine_version') {
        // Update the version in the Information popup and welcome popup
        UI.setVersionNumber(msg.version);
        console.log('Version', msg.version);
        // Show info popup on startup if enabled
        UI.showInfoPage();
    }
    else if (msg.type === 'alert') {
        alert('[WORKER] ' + msg.message);
    }
    else if (msg.type === 'data') {
        let data = msg.data;
        presentC = data.present.c;
        console.log('[AI-DBG] data msg received, presentC=', presentC, 'afterSubmit=', data.afterSubmit);
        addHighlight(data, '--highlight-generated-move', 'coordinates', generatedMoves.map(q => ({l: q.l, t: q.t, x: q.x, y: q.y, c: presentC})));
        if(data.phantom && data.phantom.length > 0) {
            UI.setHudLight(true);
            if (showPhantom) {
                addHighlight(data, '--highlight-phantom-board', 'boards', data.phantom.map((b) => ({l: b.l, t: b.t, c: !presentC})));
                data.boards = data.boards.concat(data.phantom);
                addHighlight(data, '--highlight-check', 'arrows', data.phantomChecks.map((c) => ({
                    from: {...c.from, c: !presentC},
                    to: {...c.to, c: !presentC}
                })));
                data.fade = 0.5;
            }
        } else {
            UI.setHudLight(false);
        }
        window.chessBoardCanvas.setData(data);
        clicking = false;

        // ---- AI turn check (reliable: only after submit, presentC is up-to-date) ----
        if (data.afterSubmit) {
            console.log('[AI-DBG] afterSubmit detected, calling aiTryPlay()');
            aiTryPlay();
        }
    }
    else if (msg.type === 'moves') {
        generatedMoves = msg.moves;
        if (generatedMoves.length === 0){
            clickedPos = null;
            clicking = false;
        }
        else
        {
            worker.postMessage({type: 'view'});
        }
    }
    else if (msg.type === 'update_buttons')
    {
        for(let key of ['undo', 'redo', 'prev', 'next', 'submit'])
        {
            if(msg[key])
            {
                UI.buttons.enable(key);
            }
            else
            {
                UI.buttons.disable(key);
            }
        }
    }
    else if (msg.type === 'update_select')
    {
        let options = msg.options.map(obj => obj.pgn);
        nextOptions = msg.options.map(obj => obj.action);
        UI.select.setOptions(options);
    }
    else if (msg.type === 'update_pgn')
    {
        UI.setExportData(msg.pgn);
        console.log('[AI-DBG] update_pgn received, aiWaitingForExport=', aiWaitingForExport, 'pgn length=', msg.pgn.length);
        // If AI is waiting for export to get PGN for its move
        if (aiWaitingForExport) {
            aiWaitingForExport = false;
            console.log('[AI-DBG] Calling aiRequestMove with PGN');
            aiRequestMove(msg.pgn);
        }
    }
    else if (msg.type === 'update_hud_status')
    {
        UI.setHudTitle(msg.hudTitle);
        if (msg.hudText) {
            UI.setHudText(msg.hudText);
        }
        else {
            UI.setHudText('');
        }
        // If AI just submitted, check if game continues and it's AI's turn again
        // (This handles multi-move scenarios)
    }
}

UI.buttons.setCallbacks({
    prev: () => {
        deselect();
        worker.postMessage({type: 'prev'});
        disablePhantom();
    },
    next: () => {
        deselect();
        let index = UI.select.getSelectedIndex();
        let action = nextOptions[index];
        worker.postMessage({type: 'next', action: action});
        disablePhantom();
    },
    undo: () => {
        deselect();
        worker.postMessage({type: 'undo'});
    },
    redo: () => {
        deselect();
        worker.postMessage({type: 'redo'});
    },
    submit: () => {
        console.log('[AI-DBG] Submit button clicked');
        deselect();
        worker.postMessage({type: 'submit'});
        disablePhantom();
        // AI turn check happens automatically via data.afterSubmit from worker
    },
});

UI.setHintCallback(() => {
    worker.postMessage({type: 'hint'});
});

UI.setFocusCallback(() => {
    window.chessBoardCanvas.goToNextFocus();
});

UI.setImportCallback((data) => {
    worker.postMessage({type: 'load', pgn: data});
});

UI.setExportCallback(() => {
    worker.postMessage({type: 'export'});
});

UI.setCommentsEditCallback((text) => {
    worker.postMessage({ type: 'update_comment', comment: text });
});

// Unified settings handler: forward to worker and handle local UI changes
UI.setSettingsChangeCallback((settings) => {
    // Forward all settings updates to the worker
    worker.postMessage({ type: 'update_settings', settings: settings });

    // Handle local UI changes based on settings keys
    if (settings.theme !== undefined) {
        // Theme changed: reload canvas colors
        if (window.chessBoardCanvas && window.chessBoardCanvas.reloadColors) {
            window.chessBoardCanvas.reloadColors();
        }
    } else if (settings.debugWindow !== undefined) {
        // Toggle debug window visibility
        const dbg = document.querySelector('.debug-window');
        if (dbg) dbg.style.display = settings.debugWindow ? '' : 'none';
    } else if (settings.allowSubmitWithChecks !== undefined) {
        // No local action needed here; worker will use this
    } else if (settings.showMovablePieces !== undefined) {
        worker.postMessage({ type: 'view' });
    } else if (settings.autoToggleComments !== undefined) {
        // Auto toggle of the HUD comments area is handled entirely within UI.js
    }
});

// Testing HUD light functions
UI.setHudLightCallback((isOn) => {
    showPhantom = isOn;
    worker.postMessage({ type: 'view' });
});

UI.setHudLight(false); // Initialize HUD light to off

// ================= AI Opponent Logic =================

async function aiCheckStatus() {
    console.log('[AI-DBG] aiCheckStatus() called, fetching', AI_API_BASE + '/api/status');
    try {
        const resp = await fetch(AI_API_BASE + '/api/status');
        if (!resp.ok) throw new Error('Server error');
        const data = await resp.json();
        aiConnected = true;
        console.log('[AI-DBG] aiCheckStatus: connected! model=', data.model);
        UI.setAIStatus('connected', 'Connected');
        UI.setAIModelInfo(data.model || {});
    } catch (e) {
        aiConnected = false;
        console.log('[AI-DBG] aiCheckStatus FAILED:', e.message);
        UI.setAIStatus('error', 'Disconnected: ' + e.message);
        UI.setAIModelInfo({ loaded: false });
    }
}

async function aiReloadModel() {
    try {
        UI.setAIStatus('thinking', 'Reloading model...');
        const resp = await fetch(AI_API_BASE + '/api/reload', { method: 'POST' });
        if (!resp.ok) throw new Error('Reload failed');
        const data = await resp.json();
        UI.setAIStatus('connected', 'Model reloaded');
        UI.setAIModelInfo(data.model || {});
    } catch (e) {
        UI.setAIStatus('error', 'Reload failed: ' + e.message);
    }
}

async function aiRequestMove(pgn) {
    console.log('[AI-DBG] aiRequestMove() called, aiConnected=', aiConnected, 'aiThinking=', aiThinking);
    if (!aiConnected || aiThinking) {
        console.log('[AI-DBG] aiRequestMove EARLY RETURN: connected=', aiConnected, 'thinking=', aiThinking);
        return;
    }
    
    const settings = UI.getAISettings();
    console.log('[AI-DBG] aiRequestMove settings=', JSON.stringify(settings));
    if (!settings.enabled) {
        console.log('[AI-DBG] aiRequestMove EARLY RETURN: AI not enabled');
        return;
    }
    
    aiThinking = true;
    UI.setAIStatus('thinking', 'AI is thinking...');
    
    try {
        console.log('[AI-DBG] Sending PGN to AI server, length=', pgn.length);
        console.log('[AI-DBG] PGN content:', pgn);
        const resp = await fetch(AI_API_BASE + '/api/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pgn: pgn,
                temperature: settings.temperature,
            })
        });
        
        if (!resp.ok) throw new Error('Server error ' + resp.status);
        const data = await resp.json();
        console.log('[AI-DBG] AI server response:', JSON.stringify(data));
        
        if (data.success && data.moves && data.moves.length > 0) {
            UI.setAIStatus('connected', `AI move (value: ${data.value})`);
            // Apply each move from the AI
            for (const mv of data.moves) {
                const from = { l: mv.from.l, t: mv.from.t, x: mv.from.x, y: mv.from.y };
                const to = { l: mv.to.l, t: mv.to.t, x: mv.to.x, y: mv.to.y };
                worker.postMessage({ type: 'apply_move', from: from, to: to });
            }
            // Submit the AI's move; the worker will respond with
            // data.afterSubmit=true, which triggers aiTryPlay() in the handler.
            aiThinking = false;
            worker.postMessage({ type: 'submit' });
        } else {
            UI.setAIStatus('connected', data.message || 'No move returned');
            aiThinking = false;
        }
    } catch (e) {
        console.log('[AI-DBG] aiRequestMove FETCH ERROR:', e);
        UI.setAIStatus('error', 'Move failed: ' + e.message);
        aiThinking = false;
    }
}

/**
 * Check if it's the AI's turn and trigger a move if so.
 * This is called:
 *   - After human submits (from the 'data' handler)
 *   - After AI submits (from the 'data' handler)
 *   - When AI is first enabled (initial trigger)
 */
function aiTryPlay() {
    const settings = UI.getAISettings();
    console.log('[AI-DBG] aiTryPlay() called, enabled=', settings.enabled, 'aiConnected=', aiConnected, 'aiThinking=', aiThinking, 'presentC=', presentC, 'aiColor=', settings.color);
    if (!settings.enabled || !aiConnected || aiThinking) {
        console.log('[AI-DBG] aiTryPlay EARLY RETURN: enabled=', settings.enabled, 'connected=', aiConnected, 'thinking=', aiThinking);
        return;
    }
    
    // presentC: false = white's turn, true = black's turn
    const isAITurn = (settings.color === 'black' && presentC === true) ||
                     (settings.color === 'white' && presentC === false);
    
    console.log('[AI-DBG] aiTryPlay isAITurn=', isAITurn, '(color=', settings.color, 'presentC=', presentC, ')');
    if (isAITurn) {
        // Request PGN export, then send to AI
        console.log('[AI-DBG] AI turn! Requesting PGN export...');
        aiWaitingForExport = true;
        worker.postMessage({ type: 'export' });
    } else {
        console.log('[AI-DBG] Not AI turn, waiting for human');
    }
}

// Wire AI callbacks
UI.setAIStatusCallback(aiCheckStatus);
UI.setAIReloadCallback(aiReloadModel);
UI.setAIEnableCallback(async (enabled) => {
    console.log('[AI-DBG] AI enable callback, enabled=', enabled);
    if (enabled) {
        await aiCheckStatus();
        console.log('[AI-DBG] After aiCheckStatus, aiConnected=', aiConnected, 'presentC=', presentC);
        // Initial trigger: if it's already AI's turn, start playing
        aiTryPlay();
    }
});

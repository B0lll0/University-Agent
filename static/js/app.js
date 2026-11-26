/* Main client controller for the Offline Edge AI Campus Assistant & Dataset Portal.
   Runs seamlessly on both `/` (full landing + Avatar + Dataset Studio) and `/kiosk` (standalone touch kiosk). */

(() => {
  const MAX_FILE_BYTES = 20 * 1024 * 1024;
  const isStandaloneKiosk = document.body.getAttribute('data-kiosk-standalone') === 'true';

  const state = {
    lang: localStorage.getItem('campus_lang') || 'en',
    theme: localStorage.getItem('campus_theme') || 'dark',
    queuedFiles: [],
    sources: [],
    chunks: [],
    qaPairs: [],
    activeTab: 'chunks',
    editingId: null,
    lastAnswer: '',
    pendingTarget: '',
    muted: localStorage.getItem('campus_muted') === 'true',
    avatarState: 'idle',
    listening: false,
    recognition: null,
    activeAudio: null,
  };

  const els = {
    html: document.documentElement,
    langToggle: document.getElementById('lang-toggle'),
    langToggleLabel: document.getElementById('lang-toggle-label'),
    themeToggle: document.getElementById('theme-toggle'),
    themeToggleLabel: document.getElementById('theme-toggle-label'),
    themeSun: document.getElementById('theme-icon-sun'),
    themeMoon: document.getElementById('theme-icon-moon'),
    menuToggle: document.getElementById('mobile-menu-toggle'),
    mobileDrawer: document.getElementById('mobile-drawer'),
    taglineReveal: document.getElementById('tagline-reveal'),
    faqContainer: document.getElementById('faq-container'),

    /* Avatar & Multimodal Voice controls */
    avatarStage: document.getElementById('avatar-stage'),
    avatarStateLabel: document.getElementById('avatar-state-label'),
    avatarLiveTranscript: document.getElementById('avatar-live-transcript'),
    avatarMicBtn: document.getElementById('avatar-mic-btn'),
    avatarMicLabel: document.getElementById('avatar-mic-label'),
    entryGate: document.getElementById('entry-gate'),
    sttLang: document.getElementById('stt-lang'),
    ttsEngine: document.getElementById('tts-engine'),
    ttsRate: document.getElementById('tts-rate'),
    ttsAutoSpeak: document.getElementById('tts-auto-speak'),
    avatarSpeakBtn: document.getElementById('avatar-speak-btn'),
    avatarStopBtn: document.getElementById('avatar-stop-btn'),
    avatarMuteBtn: document.getElementById('avatar-mute-btn'),
    avatarMuteLabel: document.getElementById('avatar-mute-label'),
    muteIconOn: document.getElementById('mute-icon-on'),
    muteIconOff: document.getElementById('mute-icon-off'),
    avatarClearBtn: document.getElementById('avatar-clear-btn'),
    chatMiniAvatar: document.getElementById('chat-mini-avatar'),
    avatarLatencyBadge: document.getElementById('avatar-latency-badge'),
    avatarChatFeed: document.getElementById('avatar-chat-feed'),
    avatarChatForm: document.getElementById('avatar-chat-form'),
    avatarChatInput: document.getElementById('avatar-chat-input'),
    quickPrompts: document.getElementById('quick-prompts'),

    /* Health & Workspace elements (present on /) */
    healthFlaskDot: document.getElementById('health-flask-dot'),
    healthOllamaDot: document.getElementById('health-ollama-dot'),
    healthOllamaLabel: document.getElementById('health-ollama-label'),
    healthChromaLabel: document.getElementById('health-chroma-label'),

    statSources: document.getElementById('stat-sources'),
    statChunks: document.getElementById('stat-chunks'),
    statQa: document.getElementById('stat-qa'),
    statTokens: document.getElementById('stat-tokens'),

    statusBanner: document.getElementById('status-banner'),
    statusBannerText: document.getElementById('status-banner-text'),
    statusBannerClose: document.getElementById('status-banner-close'),

    uploadForm: document.getElementById('upload-form'),
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('file-input'),
    fileQueue: document.getElementById('file-queue'),
    courseName: document.getElementById('course-name'),
    department: document.getElementById('department'),
    chunkSize: document.getElementById('chunk-size'),
    chunkOverlap: document.getElementById('chunk-overlap'),
    useOllama: document.getElementById('use-ollama'),
    ollamaModelWrap: document.getElementById('ollama-model-wrap'),
    ollamaModel: document.getElementById('ollama-model'),
    btnProcess: document.getElementById('btn-process'),
    btnLoadSample: document.getElementById('btn-load-sample'),
    btnClearAll: document.getElementById('btn-clear-all'),
    sourcesContainer: document.getElementById('sources-container'),
    sourcesList: document.getElementById('sources-list'),

    tabChunks: document.getElementById('tab-chunks'),
    tabQa: document.getElementById('tab-qa'),
    tabKiosk: document.getElementById('tab-kiosk'),
    panelChunks: document.getElementById('panel-chunks'),
    panelQa: document.getElementById('panel-qa'),
    panelKiosk: document.getElementById('panel-kiosk'),
    inspectorFilters: document.getElementById('inspector-filters'),
    searchInput: document.getElementById('search-input'),
    langFilter: document.getElementById('lang-filter'),

    workspaceSkeleton: document.getElementById('workspace-skeleton'),
    workspaceEmpty: document.getElementById('workspace-empty'),
    chunksList: document.getElementById('chunks-list'),
    qaList: document.getElementById('qa-list'),
    manualQaForm: document.getElementById('manual-qa-form'),
    manualQ: document.getElementById('manual-q'),
    manualA: document.getElementById('manual-a'),

    kioskForm: document.getElementById('kiosk-form'),
    kioskInput: document.getElementById('kiosk-input'),
    btnVoiceIn: document.getElementById('btn-voice-in'),
    btnVoiceOut: document.getElementById('btn-voice-out'),
    kioskLatency: document.getElementById('kiosk-latency'),
    kioskAnswerText: document.getElementById('kiosk-answer-text'),
    kioskSources: document.getElementById('kiosk-sources'),

    exportBtns: document.querySelectorAll('.export-btn'),
    btnChromaSync: document.getElementById('btn-chroma-sync'),
  };

  function dict() {
    return window.CAMPUS_I18N[state.lang] || window.CAMPUS_I18N.en;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /* ---------- Theme Management (Dark / Light) ---------- */
  function applyTheme(theme) {
    state.theme = theme === 'light' ? 'light' : 'dark';
    els.html.setAttribute('data-theme', state.theme);
    localStorage.setItem('campus_theme', state.theme);
    const t = dict();
    if (els.themeSun && els.themeMoon && els.themeToggleLabel) {
      if (state.theme === 'dark') {
        els.themeSun.classList.remove('is-hidden');
        els.themeMoon.classList.add('is-hidden');
        els.themeToggleLabel.textContent = t.themeLight;
      } else {
        els.themeSun.classList.add('is-hidden');
        els.themeMoon.classList.remove('is-hidden');
        els.themeToggleLabel.textContent = t.themeDark;
      }
    }
  }

  /* ---------- Language & RTL Management (EN / AR) ---------- */
  function applyLanguage(lang) {
    state.lang = lang === 'ar' ? 'ar' : 'en';
    els.html.setAttribute('lang', state.lang);
    els.html.setAttribute('dir', state.lang === 'ar' ? 'rtl' : 'ltr');
    localStorage.setItem('campus_lang', state.lang);

    const t = dict();
    document.querySelectorAll('[data-i18n]').forEach((node) => {
      const key = node.getAttribute('data-i18n');
      if (t[key]) node.textContent = t[key];
    });
    document.querySelectorAll('[data-i18n-html]').forEach((node) => {
      const key = node.getAttribute('data-i18n-html');
      if (t[key]) node.innerHTML = t[key];
    });

    if (els.langToggleLabel) els.langToggleLabel.textContent = t.langSwitchLabel;
    if (els.avatarChatInput) els.avatarChatInput.placeholder = t.placeholderAvatarChat;
    if (els.searchInput) els.searchInput.placeholder = t.placeholderSearch;
    if (els.courseName) els.courseName.placeholder = t.placeholderCourse;
    if (els.manualQ) els.manualQ.placeholder = t.placeholderManualQ;
    if (els.manualA) els.manualA.placeholder = t.placeholderManualA;
    if (els.kioskInput) els.kioskInput.placeholder = t.placeholderKiosk;

    applyTheme(state.theme);
    applyMuteState(state.muted);
    setAvatarState(state.avatarState);
    if (els.taglineReveal) renderTaglineWords();
    if (els.faqContainer) renderFaqList();
    if (!isStandaloneKiosk) {
      renderFileQueue();
      renderDatasetViews();
    }
  }

  /* ---------- Avatar Visual State & Mute Management ---------- */
  function applyMuteState(isMuted) {
    state.muted = Boolean(isMuted);
    localStorage.setItem('campus_muted', state.muted ? 'true' : 'false');
    if (state.muted) stopSpeaking();
    const t = dict();
    if (els.avatarMuteBtn) {
      els.avatarMuteBtn.setAttribute('aria-pressed', state.muted ? 'true' : 'false');
      els.avatarMuteBtn.classList.toggle('btn-muted-active', state.muted);
    }
    if (els.avatarMuteLabel) {
      els.avatarMuteLabel.textContent = state.muted ? t.btnUnmute : t.btnMute;
    }
    if (els.muteIconOn && els.muteIconOff) {
      els.muteIconOn.classList.toggle('is-hidden', state.muted);
      els.muteIconOff.classList.toggle('is-hidden', !state.muted);
    }
    if (els.ttsAutoSpeak) {
      els.ttsAutoSpeak.checked = !state.muted;
    }
  }

  function clearChatFeed() {
    stopSpeaking();
    state.pendingTarget = '';
    state.lastAnswer = '';
    if (els.avatarSpeakBtn) els.avatarSpeakBtn.disabled = true;
    if (els.avatarLatencyBadge) els.avatarLatencyBadge.textContent = '0 ms · local';
    if (els.avatarLiveTranscript) els.avatarLiveTranscript.textContent = dict().avatarMicHint;
    if (!els.avatarChatFeed) return;
    const t = dict();
    els.avatarChatFeed.innerHTML = '';
    appendChatBubble('avatar', t.avatarWelcomeMsg);
  }

  function setAvatarState(nextState) {
    state.avatarState = nextState;
    if (els.avatarStage) els.avatarStage.setAttribute('data-avatar-state', nextState);
    if (els.chatMiniAvatar) els.chatMiniAvatar.setAttribute('data-avatar-state', nextState);
    const t = dict();
    const labelMap = {
      idle: t.avatarStateIdle,
      listening: t.avatarStateListening,
      thinking: t.avatarStateThinking,
      speaking: t.avatarStateSpeaking,
    };
    if (els.avatarStateLabel) {
      els.avatarStateLabel.textContent = labelMap[nextState] || t.avatarStateIdle;
    }
    if (els.avatarStopBtn) {
      els.avatarStopBtn.disabled = nextState !== 'speaking';
    }
  }

  /* ---------- Dual-Layer Text-to-Speech (Browser Voices + Offline Flask WAV) ---------- */
  function stopSpeaking() {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    if (state.activeAudio) {
      state.activeAudio.pause();
      state.activeAudio = null;
    }
    if (state.avatarState === 'speaking') {
      setAvatarState('idle');
    }
  }

  function pickBrowserVoice(isArabic) {
    if (!('speechSynthesis' in window)) return null;
    const voices = window.speechSynthesis.getVoices() || [];
    if (!voices.length) return null;
    const prefix = isArabic ? 'ar' : 'en';
    const preferred = voices.find((v) => v.lang && v.lang.toLowerCase().startsWith(isArabic ? 'ar-eg' : 'en-us'));
    return preferred || voices.find((v) => v.lang && v.lang.toLowerCase().startsWith(prefix)) || null;
  }

  async function playServerTtsWav(text, isArabic, rate) {
    setAvatarState('speaking');
    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, lang: isArabic ? 'ar' : 'en', rate }),
      });
      if (!res.ok) {
        setAvatarState('idle');
        return false;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      state.activeAudio = audio;
      audio.onended = () => {
        URL.revokeObjectURL(url);
        state.activeAudio = null;
        setAvatarState('idle');
      };
      audio.onerror = () => {
        URL.revokeObjectURL(url);
        state.activeAudio = null;
        setAvatarState('idle');
      };
      await audio.play();
      return true;
    } catch (_err) {
      setAvatarState('idle');
      return false;
    }
  }

  async function speakText(text, force = false) {
    if (!text) return;
    if (state.muted && !force) return;
    stopSpeaking();
    const isArabic = /[\u0600-\u06FF]/.test(text);
    const rate = els.ttsRate ? Number(els.ttsRate.value) || 1.0 : 1.0;
    const engine = els.ttsEngine ? els.ttsEngine.value : 'browser';

    if (engine === 'server') {
      const ok = await playServerTtsWav(text, isArabic, rate);
      if (ok) return;
    }

    if ('speechSynthesis' in window) {
      const utter = new SpeechSynthesisUtterance(text);
      utter.lang = isArabic ? 'ar-EG' : 'en-US';
      utter.rate = rate;
      const chosen = pickBrowserVoice(isArabic);
      if (chosen) utter.voice = chosen;

      utter.onstart = () => setAvatarState('speaking');
      utter.onend = () => setAvatarState('idle');
      utter.onerror = () => setAvatarState('idle');
      setAvatarState('speaking');
      window.speechSynthesis.speak(utter);
    } else {
      await playServerTtsWav(text, isArabic, rate);
    }
  }

  /* ---------- Avatar Conversation & Local RAG Query ---------- */
  function appendChatBubble(role, text, matches = [], gateOptions = [], ragTrace = null, suggestedQuestions = []) {
    if (!els.avatarChatFeed) return;
    const t = dict();
    const sender = role === 'student' ? t.studentSenderName : t.avatarSenderName;
    const gateButtonsHtml = gateOptions.length
      ? `<div class="action-row gate-choice-row" style="margin-top: var(--space-50);">${gateOptions
          .map(
            (opt) =>
              `<button type="button" class="btn btn-primary button-sm gate-choice-btn" data-gate="${escapeHtml(opt.gate)}" data-target="${escapeHtml(opt.target)}" data-query="${escapeHtml(opt.query)}">${escapeHtml(opt.label)}</button>`
          )
          .join('')}</div>`
      : '';

    let followupsHtml = '';
    if (role === 'avatar' && suggestedQuestions && suggestedQuestions.length) {
      const hasArFollowup = /[\u0600-\u06FF]/.test(suggestedQuestions[0]);
      const followLabel = hasArFollowup ? 'أسئلة مقترحة تالية:' : 'Suggested next questions:';
      followupsHtml = `
        <div class="followup-wrap">
          <span class="caption-4 text-muted">${escapeHtml(followLabel)}</span>
          <div class="followup-list">
            ${suggestedQuestions
              .map(
                (q, idx) =>
                  `<button type="button" class="followup-btn caption-2" data-followup-query="${escapeHtml(q)}">${idx + 1}. ${escapeHtml(q)}</button>`
              )
              .join('')}
          </div>
        </div>
      `;
    }

    const sourcesHtml = matches.length
      ? `<div class="chat-sources">${matches
          .map(
            (m) =>
              `<span class="badge caption-5 font-mono">${escapeHtml(m.source_file)} · ${Math.round((m.confidence || 0) * 100)}%</span>`
          )
          .join('')}</div>`
      : '';

    let traceHtml = '';
    if (ragTrace && role === 'avatar') {
      const traceSummary = state.lang === 'ar'
        ? `تفاصيل RAG (${ragTrace.vector_db} · ${ragTrace.indexed_vectors} متجه)`
        : `RAG trace (${ragTrace.vector_db} · ${ragTrace.indexed_vectors} vectors)`;
      const chunkRows = (ragTrace.retrieved_chunks || [])
        .map(
          (c) =>
            `<div class="caption-5 text-secondary" style="margin-top: var(--space-50);"><span class="font-mono text-primary">${Math.round((c.similarity || 0) * 100)}%</span> · <strong>${escapeHtml(c.source_file)}</strong> — ${escapeHtml(c.excerpt)}</div>`
        )
        .join('');
      traceHtml = `
        <details style="margin-top: var(--space-50);">
          <summary class="caption-5 font-mono text-muted" style="cursor: pointer;">${escapeHtml(traceSummary)}</summary>
          <div style="margin-top: var(--space-50); padding: var(--space-75); border-radius: var(--radius-md); background-color: var(--bg-nested); border: 1px solid var(--border-subtle);">
            <div class="caption-5 font-mono text-muted">${escapeHtml(ragTrace.embedding_model)} → ${escapeHtml(ragTrace.generator)}</div>
            ${chunkRows}
          </div>
        </details>
      `;
    }

    const row = document.createElement('div');
    row.className = `chat-bubble-row ${role === 'student' ? 'is-student' : ''}`;
    const avatarIconHtml =
      role === 'avatar'
        ? `<span class="chat-bubble-avatar" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 220 220" fill="none">
              <rect x="34" y="44" width="152" height="132" rx="40" fill="var(--bg-elevated)" stroke="currentColor" stroke-width="12"/>
              <circle cx="86" cy="100" r="13" fill="currentColor"/>
              <circle cx="134" cy="100" r="13" fill="currentColor"/>
              <path d="M86 136C96 146 124 146 134 136" stroke="currentColor" stroke-width="12" stroke-linecap="round"/>
            </svg>
          </span>`
        : '';

    row.innerHTML = `
      ${avatarIconHtml}
      <div class="chat-bubble ${role === 'student' ? 'is-student' : 'is-avatar'}">
        <span class="caption-3 text-muted">${escapeHtml(sender)}</span>
        <p class="body-5" style="white-space: pre-wrap;">${escapeHtml(text)}</p>
        ${gateButtonsHtml}
        ${followupsHtml}
        ${sourcesHtml}
        ${traceHtml}
      </div>
    `;
    els.avatarChatFeed.appendChild(row);
    els.avatarChatFeed.scrollTop = els.avatarChatFeed.scrollHeight;
  }

  async function askAvatar(question, explicitGate = '', explicitTarget = '') {
    const cleanQ = (question || '').trim();
    if (!cleanQ) return;
    stopSpeaking();
    appendChatBubble('student', cleanQ);
    setAvatarState('thinking');
    if (els.avatarLiveTranscript) {
      els.avatarLiveTranscript.textContent = cleanQ;
    }

    const targetToSend = explicitTarget || state.pendingTarget || '';

    try {
      const edgeModel = document.body.getAttribute('data-edge-model') || '';
      const selectedModel = els.ollamaModel ? els.ollamaModel.value : edgeModel;
      const useOllama = selectedModel !== 'rag_only';
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: cleanQ,
          use_ollama: useOllama,
          ollama_model: useOllama ? selectedModel : '',
          entry_gate: explicitGate || '',
          pending_target: targetToSend,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setAvatarState('idle');
        appendChatBubble('avatar', data.error || 'Could not process this question.');
        return;
      }

      const answer = data.answer || '';
      state.lastAnswer = answer;
      state.pendingTarget = (data.needs_gate || (data.pending_target || '').startsWith('MSA_TOP_FAQ:'))
        ? (data.pending_target || '')
        : '';
      let followups = Array.isArray(data.suggested_questions) ? data.suggested_questions : [];
      const hasInteractiveOptions = Array.isArray(data.gate_options) && data.gate_options.length > 0;
      if (!followups.length && !data.needs_gate && !hasInteractiveOptions) {
        const isArAns = /[\u0600-\u06FF]/.test(answer || cleanQ);
        followups = isArAns
          ? [
              'كيف تُدفع المصروفات الدراسية ورسوم الاعتماد البريطاني في MSA؟',
              'ما هي المنح الدراسية والخصومات المتاحة في جامعة MSA؟',
              'ما هي كليات جامعة MSA؟',
            ]
          : [
              'How do tuition fees, British validation fees, and annual increases work at MSA University?',
              'What scholarships and tuition discounts does MSA University offer?',
              'What are the faculties MSA University has?',
            ];
      }
      appendChatBubble('avatar', answer, data.matches || [], data.gate_options || [], data.rag_trace || null, followups);

      if (els.avatarLatencyBadge) {
        els.avatarLatencyBadge.textContent = `${data.latency_ms} ms · ${data.mode}`;
      }
      if (els.avatarSpeakBtn) {
        els.avatarSpeakBtn.disabled = !answer;
      }
      if (els.kioskAnswerText) {
        els.kioskAnswerText.textContent = answer;
      }
      if (els.kioskLatency) {
        els.kioskLatency.textContent = `${data.latency_ms} ms · ${data.mode}`;
      }
      if (els.btnVoiceOut) {
        els.btnVoiceOut.disabled = !answer;
      }

      const shouldAutoSpeak = !state.muted && (!els.ttsAutoSpeak || els.ttsAutoSpeak.checked);
      if (shouldAutoSpeak && answer) {
        await speakText(answer);
      } else {
        setAvatarState('idle');
      }
    } catch (_err) {
      setAvatarState('idle');
      appendChatBubble('avatar', 'Could not reach the local Flask server. Verify app.py is running.');
    }
  }

  /* ---------- Voice-to-Text (Speech Recognition) ---------- */
  function resolveSpeechLang() {
    const selected = els.sttLang ? els.sttLang.value : 'auto';
    if (selected && selected !== 'auto') return selected;
    return state.lang === 'ar' ? 'ar-EG' : 'en-US';
  }

  function toggleAvatarMic() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      if (els.avatarLiveTranscript) els.avatarLiveTranscript.textContent = dict().errNoSpeech;
      if (!isStandaloneKiosk) showBanner(dict().errNoSpeech, 'error');
      return;
    }

    if (state.listening && state.recognition) {
      state.recognition.stop();
      return;
    }

    stopSpeaking();
    const rec = new SpeechRec();
    rec.lang = resolveSpeechLang();
    rec.interimResults = true;
    rec.maxAlternatives = 1;
    state.recognition = rec;
    state.listening = true;
    setAvatarState('listening');
    if (els.avatarMicBtn) els.avatarMicBtn.classList.add('is-active');
    if (els.avatarMicLabel) els.avatarMicLabel.textContent = dict().avatarMicStop;

    let finalTranscript = '';
    rec.onresult = (evt) => {
      let interim = '';
      for (let i = evt.resultIndex; i < evt.results.length; i += 1) {
        const piece = evt.results[i][0].transcript;
        if (evt.results[i].isFinal) {
          finalTranscript += piece;
        } else {
          interim += piece;
        }
      }
      const preview = (finalTranscript || interim).trim();
      if (preview && els.avatarLiveTranscript) {
        els.avatarLiveTranscript.textContent = preview;
      }
      if (preview && els.avatarChatInput) {
        els.avatarChatInput.value = preview;
      }
    };

    rec.onerror = () => {
      state.listening = false;
      if (els.avatarMicBtn) els.avatarMicBtn.classList.remove('is-active');
      if (els.avatarMicLabel) els.avatarMicLabel.textContent = dict().avatarMicStart;
      setAvatarState('idle');
    };

    rec.onend = () => {
      state.listening = false;
      if (els.avatarMicBtn) els.avatarMicBtn.classList.remove('is-active');
      if (els.avatarMicLabel) els.avatarMicLabel.textContent = dict().avatarMicStart;
      const query = (finalTranscript || (els.avatarChatInput ? els.avatarChatInput.value : '')).trim();
      if (query) {
        if (els.avatarChatInput) els.avatarChatInput.value = '';
        askAvatar(query);
      } else {
        setAvatarState('idle');
      }
    };

    rec.start();
  }

  /* ---------- B11. Word-by-Word Scroll Tagline Reveal ---------- */
  function renderTaglineWords() {
    if (!els.taglineReveal) return;
    const words = dict().taglineText.split(/\s+/);
    els.taglineReveal.innerHTML = words
      .map((w, idx) => `<span class="tagline-word" data-word-idx="${idx}">${escapeHtml(w)}</span>`)
      .join(' ');
    updateTaglineProgress();
  }

  let taglineRafPending = false;
  function updateTaglineProgress() {
    taglineRafPending = false;
    const section = document.getElementById('tagline');
    if (!section || !els.taglineReveal) return;
    const words = els.taglineReveal.querySelectorAll('.tagline-word');
    if (!words.length) return;

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      words.forEach((w) => w.classList.add('is-lit'));
      return;
    }

    const rect = section.getBoundingClientRect();
    const vh = window.innerHeight || 800;
    const startY = vh * 0.88;
    const endY = vh * 0.24;
    const rawProgress = (startY - rect.top) / (startY - endY + rect.height * 0.45);
    const clamped = Math.max(0, Math.min(1, rawProgress));
    const litCount = Math.floor(clamped * (words.length + 1));

    words.forEach((word, index) => {
      word.classList.toggle('is-lit', index < litCount);
    });
  }

  function scheduleTaglineUpdate() {
    if (!taglineRafPending) {
      taglineRafPending = true;
      window.requestAnimationFrame(updateTaglineProgress);
    }
  }

  /* ---------- Render FAQs ---------- */
  function renderFaqList() {
    if (!els.faqContainer) return;
    const faqs = dict().faqs || [];
    els.faqContainer.innerHTML = faqs
      .map(
        (item, idx) => `
        <details class="faq-item"${idx === 0 ? ' open' : ''}>
          <summary class="title-2">
            <span>${escapeHtml(item.q)}</span>
            <svg class="faq-chevron" width="18" height="18" viewBox="0 0 256 256" fill="none" aria-hidden="true">
              <polyline points="208 96 128 176 48 96" stroke="currentColor" stroke-width="20" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </summary>
          <p class="body-3 faq-answer">${escapeHtml(item.a)}</p>
        </details>
      `
      )
      .join('');
  }

  /* ---------- B7. Mobile Drawer & Section Observers ---------- */
  function setMobileMenu(open) {
    if (!els.menuToggle || !els.mobileDrawer) return;
    els.menuToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    els.mobileDrawer.setAttribute('aria-hidden', open ? 'false' : 'true');
    els.mobileDrawer.classList.toggle('is-open', open);
    document.body.style.overflow = open ? 'hidden' : '';
  }

  function initObservers() {
    const revealEls = document.querySelectorAll('.reveal');
    if ('IntersectionObserver' in window) {
      const revealObserver = new IntersectionObserver(
        (entries, obs) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              entry.target.classList.add('is-visible');
              obs.unobserve(entry.target);
            }
          });
        },
        { threshold: 0.12 }
      );
      revealEls.forEach((el) => revealObserver.observe(el));

      const navLinks = document.querySelectorAll('[data-nav-target]');
      const sections = ['avatar', 'workspace', 'benefits', 'how-it-works', 'faq']
        .map((id) => document.getElementById(id))
        .filter(Boolean);

      const navObserver = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              navLinks.forEach((link) => {
                const match = link.getAttribute('data-nav-target') === entry.target.id;
                link.setAttribute('aria-current', match ? 'true' : 'false');
              });
            }
          });
        },
        { rootMargin: '-35% 0px -55% 0px' }
      );
      sections.forEach((sec) => navObserver.observe(sec));
    } else {
      revealEls.forEach((el) => el.classList.add('is-visible'));
    }
  }

  /* ---------- Status Banner & Skeleton Loading ---------- */
  function showBanner(message, type = 'info') {
    if (!els.statusBanner || !els.statusBannerText) return;
    els.statusBannerText.textContent = message;
    els.statusBanner.classList.remove('is-error', 'is-success');
    if (type === 'error') els.statusBanner.classList.add('is-error');
    if (type === 'success') els.statusBanner.classList.add('is-success');
    els.statusBanner.classList.add('is-visible');
  }

  function hideBanner() {
    if (!els.statusBanner) return;
    els.statusBanner.classList.remove('is-visible', 'is-error', 'is-success');
  }

  function setLoading(isLoading) {
    if (!els.workspaceSkeleton) return;
    els.workspaceSkeleton.classList.toggle('is-hidden', !isLoading);
    if (els.btnProcess) els.btnProcess.disabled = isLoading;
    if (els.btnLoadSample) els.btnLoadSample.disabled = isLoading;
    if (isLoading) {
      els.workspaceEmpty.classList.add('is-hidden');
      els.panelChunks.classList.add('is-hidden');
      els.panelQa.classList.add('is-hidden');
      els.panelKiosk.classList.add('is-hidden');
    } else {
      renderDatasetViews();
    }
  }

  /* ---------- File Queue & Dataset Studio Rendering ---------- */
  function addFilesToQueue(fileList) {
    let skippedSize = false;
    Array.from(fileList).forEach((file) => {
      if (file.size > MAX_FILE_BYTES) {
        skippedSize = true;
        return;
      }
      const already = state.queuedFiles.some((f) => f.name === file.name && f.size === file.size);
      if (!already) state.queuedFiles.push(file);
    });
    if (skippedSize) showBanner(dict().errFileTooLarge, 'error');
    renderFileQueue();
  }

  function renderFileQueue() {
    if (!els.fileQueue) return;
    if (!state.queuedFiles.length) {
      els.fileQueue.innerHTML = '';
      return;
    }
    const removeLabel = dict().removeBtn;
    els.fileQueue.innerHTML = state.queuedFiles
      .map((file, idx) => {
        const kb = Math.max(1, Math.round(file.size / 1024));
        return `
          <div class="file-chip">
            <div style="min-width: 0;">
              <div class="body-5" style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(file.name)}</div>
              <div class="caption-5 font-mono text-muted">${kb} KB</div>
            </div>
            <button type="button" class="btn btn-ghost button-xsm remove-queued-btn" data-queue-idx="${idx}">${escapeHtml(removeLabel)}</button>
          </div>
        `;
      })
      .join('');
  }

  function updateStats(stats) {
    if (!els.statSources) return;
    els.statSources.textContent = String(stats.sources_count || 0);
    els.statChunks.textContent = String(stats.chunks_count || 0);
    els.statQa.textContent = String(stats.qa_count || 0);
    els.statTokens.textContent = String(stats.total_tokens || 0);

    const hasRecords = (stats.chunks_count || 0) > 0 || (stats.qa_count || 0) > 0;
    els.exportBtns.forEach((btn) => {
      btn.disabled = !hasRecords;
    });
    if (els.btnChromaSync) els.btnChromaSync.disabled = (stats.chunks_count || 0) === 0;
  }

  function renderSources() {
    if (!els.sourcesContainer || !els.sourcesList) return;
    if (!state.sources.length) {
      els.sourcesContainer.classList.add('is-hidden');
      return;
    }
    els.sourcesContainer.classList.remove('is-hidden');
    els.sourcesList.innerHTML = state.sources
      .map(
        (src) => `
        <div class="file-chip">
          <div style="min-width: 0;">
            <div class="body-5" style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(src.filename)}</div>
            <div class="caption-5 text-muted">${escapeHtml(src.course)} · ${escapeHtml(src.department)}</div>
          </div>
          <span class="badge caption-4 font-mono">${src.chunks_count} / ${src.qa_count}</span>
        </div>
      `
      )
      .join('');
  }

  function getFilteredItems(items, fields) {
    const query = (els.searchInput ? els.searchInput.value : '').trim().toLowerCase();
    const lang = els.langFilter ? els.langFilter.value : 'all';
    return items.filter((item) => {
      if (lang !== 'all' && item.language !== lang) return false;
      if (!query) return true;
      return fields.some((f) => String(item[f] || '').toLowerCase().includes(query));
    });
  }

  function renderChunksList() {
    if (!els.chunksList) return;
    const t = dict();
    const filtered = getFilteredItems(state.chunks, ['heading', 'text', 'source_file', 'course']);
    if (!filtered.length) {
      els.chunksList.innerHTML = `<div class="record-card"><p class="body-6 text-muted">No chunks match the current filter.</p></div>`;
      return;
    }
    els.chunksList.innerHTML = filtered
      .map((chunk) => {
        if (state.editingId === chunk.id) {
          return `
            <article class="record-card" data-chunk-id="${escapeHtml(chunk.id)}">
              <input type="text" class="input-control body-5 edit-chunk-heading" value="${escapeHtml(chunk.heading)}">
              <textarea class="input-control body-6 edit-chunk-text" rows="4">${escapeHtml(chunk.text)}</textarea>
              <div class="action-row">
                <button type="button" class="btn btn-primary button-xsm save-chunk-btn" data-id="${escapeHtml(chunk.id)}">${escapeHtml(t.saveBtn)}</button>
                <button type="button" class="btn btn-ghost button-xsm cancel-edit-btn">${escapeHtml(t.cancelBtn)}</button>
              </div>
            </article>
          `;
        }
        return `
          <article class="record-card" data-chunk-id="${escapeHtml(chunk.id)}">
            <div class="record-meta">
              <div class="badge-group">
                <span class="badge caption-4 font-mono">${escapeHtml(chunk.language.toUpperCase())}</span>
                <span class="badge caption-4">${escapeHtml(chunk.source_file)}</span>
                <span class="badge caption-4 font-mono">${chunk.token_estimate} tok</span>
              </div>
              <div class="action-row">
                <button type="button" class="btn btn-ghost button-xsm edit-chunk-btn" data-id="${escapeHtml(chunk.id)}">${escapeHtml(t.editBtn)}</button>
                <button type="button" class="btn btn-ghost button-xsm delete-chunk-btn" data-id="${escapeHtml(chunk.id)}">${escapeHtml(t.deleteBtn)}</button>
              </div>
            </div>
            <h4 class="title-4">${escapeHtml(chunk.heading)}</h4>
            <p class="body-6 record-text">${escapeHtml(chunk.text)}</p>
          </article>
        `;
      })
      .join('');
  }

  function renderQaList() {
    if (!els.qaList) return;
    const t = dict();
    const filtered = getFilteredItems(state.qaPairs, ['question', 'answer', 'source_file', 'course']);
    if (!filtered.length) {
      els.qaList.innerHTML = `<div class="record-card"><p class="body-6 text-muted">No Q&amp;A pairs match the current filter.</p></div>`;
      return;
    }
    els.qaList.innerHTML = filtered
      .map((item) => {
        if (state.editingId === item.id) {
          return `
            <article class="record-card" data-qa-id="${escapeHtml(item.id)}">
              <input type="text" class="input-control body-5 edit-qa-q" value="${escapeHtml(item.question)}">
              <textarea class="input-control body-6 edit-qa-a" rows="3">${escapeHtml(item.answer)}</textarea>
              <div class="action-row">
                <button type="button" class="btn btn-primary button-xsm save-qa-btn" data-id="${escapeHtml(item.id)}">${escapeHtml(t.saveBtn)}</button>
                <button type="button" class="btn btn-ghost button-xsm cancel-edit-btn">${escapeHtml(t.cancelBtn)}</button>
              </div>
            </article>
          `;
        }
        return `
          <article class="record-card" data-qa-id="${escapeHtml(item.id)}">
            <div class="record-meta">
              <div class="badge-group">
                <span class="badge caption-4 font-mono">${escapeHtml(item.language.toUpperCase())}</span>
                <span class="badge caption-4">${escapeHtml(item.source_file)}</span>
              </div>
              <div class="action-row">
                <button type="button" class="btn btn-ghost button-xsm edit-qa-btn" data-id="${escapeHtml(item.id)}">${escapeHtml(t.editBtn)}</button>
                <button type="button" class="btn btn-ghost button-xsm delete-qa-btn" data-id="${escapeHtml(item.id)}">${escapeHtml(t.deleteBtn)}</button>
              </div>
            </div>
            <h4 class="title-3">${escapeHtml(item.question)}</h4>
            <p class="body-6 record-text">${escapeHtml(item.answer)}</p>
          </article>
        `;
      })
      .join('');
  }

  function renderDatasetViews() {
    if (!els.workspaceEmpty) return;
    renderSources();
    const hasData = state.chunks.length > 0 || state.qaPairs.length > 0;
    const isKiosk = state.activeTab === 'kiosk';

    els.inspectorFilters.classList.toggle('is-hidden', isKiosk || !hasData);
    els.workspaceEmpty.classList.toggle('is-hidden', hasData || isKiosk);
    els.panelChunks.classList.toggle('is-hidden', !hasData || state.activeTab !== 'chunks');
    els.panelQa.classList.toggle('is-hidden', !hasData || state.activeTab !== 'qa');
    els.panelKiosk.classList.toggle('is-hidden', state.activeTab !== 'kiosk');

    if (hasData) {
      if (state.activeTab === 'chunks') renderChunksList();
      if (state.activeTab === 'qa') renderQaList();
    }
  }

  function switchTab(tabName) {
    state.activeTab = tabName;
    state.editingId = null;
    if (els.tabChunks) els.tabChunks.setAttribute('aria-selected', tabName === 'chunks' ? 'true' : 'false');
    if (els.tabQa) els.tabQa.setAttribute('aria-selected', tabName === 'qa' ? 'true' : 'false');
    if (els.tabKiosk) els.tabKiosk.setAttribute('aria-selected', tabName === 'kiosk' ? 'true' : 'false');
    renderDatasetViews();
  }

  /* ---------- API Calls ---------- */
  async function fetchHealthAndDataset() {
    try {
      const [healthRes, dataRes] = await Promise.all([fetch('/api/health'), fetch('/api/dataset')]);
      if (healthRes.ok) {
        const health = await healthRes.json();
        const t = dict();
        if (health.ollama && health.ollama.available && els.healthOllamaLabel) {
          els.healthOllamaLabel.textContent = t.statusOllamaOnline;
          if (els.healthOllamaDot) els.healthOllamaDot.classList.remove('is-warn');
        }
        if (health.ollama && health.ollama.available && els.ollamaModel) {
          const chatModels = (health.ollama.models || []).filter((m) => !m.toLowerCase().includes('embed'));
          if (chatModels.length) {
            const currentVal = els.ollamaModel.value;
            els.ollamaModel.innerHTML =
              chatModels
                .map((m) => `<option value="${escapeHtml(m)}">Ollama · ${escapeHtml(m)}</option>`)
                .join('') + `<option value="rag_only">ChromaDB RAG only (instant)</option>`;
            if (chatModels.includes(currentVal) || currentVal === 'rag_only') {
              els.ollamaModel.value = currentVal;
            } else {
              const preferred = chatModels.find((m) => m.toLowerCase().includes('phi3')) || chatModels[0];
              els.ollamaModel.value = preferred;
            }
          }
        }
        if (health.rag && els.healthChromaLabel) {
          const vecCount = health.rag.indexed_vectors || 0;
          const hasOllama = health.ollama && health.ollama.available;
          els.healthChromaLabel.textContent = hasOllama
            ? `Ollama + ChromaDB (${vecCount} vec)`
            : health.rag.chromadb_active
              ? `ChromaDB live (${vecCount} vec)`
              : `Local RAG (${vecCount} vec)`;
        }
      }
      if (dataRes.ok) {
        const data = await dataRes.json();
        state.sources = data.sources || [];
        state.chunks = data.chunks || [];
        state.qaPairs = data.qa_pairs || [];
        updateStats(data.stats || {});
        renderDatasetViews();
      }
    } catch (_err) {
      if (els.healthFlaskDot) els.healthFlaskDot.classList.add('is-warn');
    }
  }

  async function handleUploadSubmit(event) {
    event.preventDefault();
    hideBanner();
    const t = dict();
    if (!state.queuedFiles.length) {
      showBanner(t.errNoFiles, 'error');
      return;
    }
    const courseVal = els.courseName.value.trim();
    if (!courseVal) {
      els.courseName.focus();
      showBanner(t.errBadCourse, 'error');
      return;
    }

    const formData = new FormData();
    state.queuedFiles.forEach((f) => formData.append('files', f));
    formData.append('course_name', courseVal);
    formData.append('department', els.department.value);
    formData.append('chunk_size', els.chunkSize.value);
    formData.append('chunk_overlap', els.chunkOverlap.value);
    formData.append('use_ollama', els.useOllama.checked ? 'true' : 'false');
    formData.append('ollama_model', els.ollamaModel.value);

    setLoading(true);
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) {
        showBanner(data.error || 'Upload failed. Check the file format and try again.', 'error');
        return;
      }
      state.queuedFiles = [];
      els.fileInput.value = '';
      renderFileQueue();
      state.sources = data.sources || [];
      state.chunks = data.chunks || [];
      state.qaPairs = data.qa_pairs || [];
      updateStats(data.stats || {});
      const note = data.warnings && data.warnings.length ? `${data.message} ${data.warnings.join(' ')}` : data.message;
      showBanner(note, 'success');
    } catch (_err) {
      showBanner('Could not reach the local Flask server. Verify app.py is running.', 'error');
    } finally {
      setLoading(false);
    }
  }

  async function handleLoadSample() {
    hideBanner();
    setLoading(true);
    try {
      const res = await fetch('/api/load-sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chunk_size: Number(els.chunkSize.value),
          chunk_overlap: Number(els.chunkOverlap.value),
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        showBanner(data.error || 'Failed to load sample dataset.', 'error');
        return;
      }
      state.sources = data.sources || [];
      state.chunks = data.chunks || [];
      state.qaPairs = data.qa_pairs || [];
      updateStats(data.stats || {});
      showBanner(data.message, 'success');
    } catch (_err) {
      showBanner('Could not reach the local Flask server.', 'error');
    } finally {
      setLoading(false);
    }
  }

  async function handleClearAll() {
    hideBanner();
    try {
      const res = await fetch('/api/dataset/clear', { method: 'DELETE' });
      const data = await res.json();
      state.queuedFiles = [];
      state.sources = [];
      state.chunks = [];
      state.qaPairs = [];
      renderFileQueue();
      updateStats(data.stats || {});
      renderDatasetViews();
      showBanner(data.message, 'success');
    } catch (_err) {
      showBanner('Could not clear dataset.', 'error');
    }
  }

  async function handleAddManualQa(event) {
    event.preventDefault();
    hideBanner();
    const question = els.manualQ.value.trim();
    const answer = els.manualA.value.trim();
    if (question.length < 4 || answer.length < 4) {
      showBanner('Enter both a question and a verified faculty answer.', 'error');
      return;
    }
    try {
      const res = await fetch('/api/qa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          answer,
          course: els.courseName.value.trim() || 'Faculty FAQ',
          department: els.department.value,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        showBanner(data.error || 'Could not save Q&A pair.', 'error');
        return;
      }
      state.qaPairs.unshift(data.qa);
      els.manualQ.value = '';
      els.manualA.value = '';
      updateStats(data.stats || {});
      renderDatasetViews();
      showBanner(data.message, 'success');
    } catch (_err) {
      showBanner('Could not save Q&A pair.', 'error');
    }
  }

  async function handleChromaSync() {
    hideBanner();
    els.btnChromaSync.disabled = true;
    try {
      const res = await fetch('/api/chroma/ingest', { method: 'POST' });
      const data = await res.json();
      showBanner(data.message, res.ok ? 'success' : 'error');
    } catch (_err) {
      showBanner('Could not connect to local ChromaDB endpoint.', 'error');
    } finally {
      els.btnChromaSync.disabled = state.chunks.length === 0;
    }
  }

  async function handleInlineKioskAsk(event) {
    event.preventDefault();
    const question = els.kioskInput.value.trim();
    if (!question) {
      els.kioskInput.focus();
      return;
    }
    els.kioskAnswerText.textContent = state.lang === 'ar' ? 'جاري البحث في قاعدة المعرفة المحلية...' : 'Searching local knowledge base...';
    els.kioskSources.innerHTML = '';

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          use_ollama: els.useOllama.checked,
          ollama_model: els.ollamaModel.value,
          entry_gate: els.entryGate ? els.entryGate.value : '',
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        els.kioskAnswerText.textContent = data.error || 'Query failed.';
        return;
      }
      state.lastAnswer = data.answer || '';
      els.kioskAnswerText.textContent = state.lastAnswer;
      els.kioskLatency.textContent = `${data.latency_ms} ms · ${data.mode}`;
      els.btnVoiceOut.disabled = !state.lastAnswer;
      if (els.avatarSpeakBtn) els.avatarSpeakBtn.disabled = !state.lastAnswer;

      els.kioskSources.innerHTML = (data.matches || [])
        .map(
          (m) => `
          <div class="file-chip">
            <div style="min-width: 0;">
              <div class="body-5">${escapeHtml(m.title)}</div>
              <div class="caption-5 text-muted">${escapeHtml(m.source_file)} · ${escapeHtml(m.course)}</div>
            </div>
            <span class="badge caption-4 font-mono">${Math.round((m.confidence || 0) * 100)}%</span>
          </div>
        `
        )
        .join('');
    } catch (_err) {
      els.kioskAnswerText.textContent = 'Could not reach the local RAG endpoint.';
    }
  }

  /* ---------- Event Binding ---------- */
  function bindEvents() {
    if (els.themeToggle) {
      els.themeToggle.addEventListener('click', () => applyTheme(state.theme === 'dark' ? 'light' : 'dark'));
    }
    if (els.langToggle) {
      els.langToggle.addEventListener('click', () => applyLanguage(state.lang === 'en' ? 'ar' : 'en'));
    }
    if (els.menuToggle) {
      els.menuToggle.addEventListener('click', () => {
        const isOpen = els.menuToggle.getAttribute('aria-expanded') === 'true';
        setMobileMenu(!isOpen);
      });
    }
    document.querySelectorAll('.drawer-link').forEach((link) => {
      link.addEventListener('click', () => setMobileMenu(false));
    });

    /* Avatar & Multimodal Voice bindings */
    if (els.avatarMicBtn) els.avatarMicBtn.addEventListener('click', toggleAvatarMic);
    if (els.avatarSpeakBtn) els.avatarSpeakBtn.addEventListener('click', () => speakText(state.lastAnswer, true));
    if (els.avatarStopBtn) els.avatarStopBtn.addEventListener('click', stopSpeaking);
    if (els.avatarMuteBtn) els.avatarMuteBtn.addEventListener('click', () => applyMuteState(!state.muted));
    if (els.avatarClearBtn) els.avatarClearBtn.addEventListener('click', clearChatFeed);
    if (els.ttsAutoSpeak) {
      els.ttsAutoSpeak.addEventListener('change', () => applyMuteState(!els.ttsAutoSpeak.checked));
    }
    if (els.avatarChatForm) {
      els.avatarChatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const q = els.avatarChatInput.value.trim();
        if (!q) {
          els.avatarChatInput.focus();
          return;
        }
        els.avatarChatInput.value = '';
        askAvatar(q);
      });
    }
    if (els.quickPrompts) {
      els.quickPrompts.addEventListener('click', (e) => {
        const btn = e.target.closest('.quick-prompt-btn');
        if (!btn) return;
        const prompt = btn.getAttribute('data-prompt');
        if (prompt) askAvatar(prompt);
      });
    }
    if (els.avatarChatFeed) {
      els.avatarChatFeed.addEventListener('click', (e) => {
        const followupBtn = e.target.closest('.followup-btn');
        if (followupBtn) {
          const query = followupBtn.getAttribute('data-followup-query') || '';
          const wrap = followupBtn.closest('.followup-wrap');
          if (wrap) {
            wrap.querySelectorAll('.followup-btn').forEach((b) => {
              b.disabled = true;
            });
          }
          if (query) askAvatar(query);
          return;
        }
        const btn = e.target.closest('.gate-choice-btn');
        if (!btn) return;
        const gate = btn.getAttribute('data-gate') || '';
        const target = btn.getAttribute('data-target') || '';
        const query = btn.getAttribute('data-query') || `Gate ${gate}`;
        const row = btn.closest('.gate-choice-row');
        if (row) {
          row.querySelectorAll('.gate-choice-btn').forEach((b) => {
            b.disabled = true;
          });
        }
        askAvatar(query, gate, target);
      });
    }

    if (isStandaloneKiosk) return;

    if (els.statusBannerClose) els.statusBannerClose.addEventListener('click', hideBanner);

    if (els.dropzone && els.fileInput) {
      els.dropzone.addEventListener('click', () => els.fileInput.click());
      els.dropzone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          els.fileInput.click();
        }
      });
      els.dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        els.dropzone.classList.add('is-dragover');
      });
      els.dropzone.addEventListener('dragleave', () => els.dropzone.classList.remove('is-dragover'));
      els.dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        els.dropzone.classList.remove('is-dragover');
        if (e.dataTransfer && e.dataTransfer.files) addFilesToQueue(e.dataTransfer.files);
      });
      els.fileInput.addEventListener('change', (e) => {
        if (e.target.files) addFilesToQueue(e.target.files);
      });
    }

    if (els.fileQueue) {
      els.fileQueue.addEventListener('click', (e) => {
        const btn = e.target.closest('.remove-queued-btn');
        if (!btn) return;
        state.queuedFiles.splice(Number(btn.getAttribute('data-queue-idx')), 1);
        renderFileQueue();
      });
    }

    if (els.useOllama) {
      els.useOllama.addEventListener('change', () => {
        els.ollamaModelWrap.classList.toggle('is-hidden', !els.useOllama.checked);
      });
    }

    if (els.uploadForm) els.uploadForm.addEventListener('submit', handleUploadSubmit);
    if (els.btnLoadSample) els.btnLoadSample.addEventListener('click', handleLoadSample);
    if (els.btnClearAll) els.btnClearAll.addEventListener('click', handleClearAll);
    if (els.manualQaForm) els.manualQaForm.addEventListener('submit', handleAddManualQa);
    if (els.kioskForm) els.kioskForm.addEventListener('submit', handleInlineKioskAsk);
    if (els.btnVoiceIn) els.btnVoiceIn.addEventListener('click', toggleAvatarMic);
    if (els.btnVoiceOut) els.btnVoiceOut.addEventListener('click', () => speakText(state.lastAnswer));
    if (els.btnChromaSync) els.btnChromaSync.addEventListener('click', handleChromaSync);

    if (els.tabChunks) els.tabChunks.addEventListener('click', () => switchTab('chunks'));
    if (els.tabQa) els.tabQa.addEventListener('click', () => switchTab('qa'));
    if (els.tabKiosk) els.tabKiosk.addEventListener('click', () => switchTab('kiosk'));
    if (els.searchInput) els.searchInput.addEventListener('input', renderDatasetViews);
    if (els.langFilter) els.langFilter.addEventListener('change', renderDatasetViews);

    els.exportBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const fmt = btn.getAttribute('data-format');
        window.location.href = `/api/export/${fmt}`;
        showBanner(dict().msgExportStarted, 'success');
      });
    });

    if (els.chunksList) {
      els.chunksList.addEventListener('click', async (e) => {
        const editBtn = e.target.closest('.edit-chunk-btn');
        if (editBtn) {
          state.editingId = editBtn.getAttribute('data-id');
          renderChunksList();
          return;
        }
        if (e.target.closest('.cancel-edit-btn')) {
          state.editingId = null;
          renderChunksList();
          return;
        }
        const saveBtn = e.target.closest('.save-chunk-btn');
        if (saveBtn) {
          const id = saveBtn.getAttribute('data-id');
          const card = saveBtn.closest('.record-card');
          const heading = card.querySelector('.edit-chunk-heading').value.trim();
          const text = card.querySelector('.edit-chunk-text').value.trim();
          const res = await fetch(`/api/chunks/${id}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ heading, text }),
          });
          const data = await res.json();
          if (res.ok) {
            state.chunks = state.chunks.map((c) => (c.id === id ? data.chunk : c));
            state.editingId = null;
            updateStats(data.stats);
            renderChunksList();
            showBanner(data.message, 'success');
          } else {
            showBanner(data.error, 'error');
          }
          return;
        }
        const delBtn = e.target.closest('.delete-chunk-btn');
        if (delBtn) {
          const id = delBtn.getAttribute('data-id');
          const res = await fetch(`/api/chunks/${id}`, { method: 'DELETE' });
          const data = await res.json();
          if (res.ok) {
            state.chunks = state.chunks.filter((c) => c.id !== id);
            updateStats(data.stats);
            renderDatasetViews();
            showBanner(data.message, 'success');
          }
        }
      });
    }

    if (els.qaList) {
      els.qaList.addEventListener('click', async (e) => {
        const editBtn = e.target.closest('.edit-qa-btn');
        if (editBtn) {
          state.editingId = editBtn.getAttribute('data-id');
          renderQaList();
          return;
        }
        if (e.target.closest('.cancel-edit-btn')) {
          state.editingId = null;
          renderQaList();
          return;
        }
        const saveBtn = e.target.closest('.save-qa-btn');
        if (saveBtn) {
          const id = saveBtn.getAttribute('data-id');
          const card = saveBtn.closest('.record-card');
          const question = card.querySelector('.edit-qa-q').value.trim();
          const answer = card.querySelector('.edit-qa-a').value.trim();
          const res = await fetch(`/api/qa/${id}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, answer }),
          });
          const data = await res.json();
          if (res.ok) {
            state.qaPairs = state.qaPairs.map((q) => (q.id === id ? data.qa : q));
            state.editingId = null;
            updateStats(data.stats);
            renderQaList();
            showBanner(data.message, 'success');
          } else {
            showBanner(data.error, 'error');
          }
          return;
        }
        const delBtn = e.target.closest('.delete-qa-btn');
        if (delBtn) {
          const id = delBtn.getAttribute('data-id');
          const res = await fetch(`/api/qa/${id}`, { method: 'DELETE' });
          const data = await res.json();
          if (res.ok) {
            state.qaPairs = state.qaPairs.filter((q) => q.id !== id);
            updateStats(data.stats);
            renderDatasetViews();
            showBanner(data.message, 'success');
          }
        }
      });
    }

    window.addEventListener('scroll', scheduleTaglineUpdate, { passive: true });
    window.addEventListener('resize', scheduleTaglineUpdate, { passive: true });
  }

  /* Preload browser TTS voices */
  if ('speechSynthesis' in window) {
    window.speechSynthesis.getVoices();
  }

  /* ---------- Boot ---------- */
  applyLanguage(state.lang);
  bindEvents();
  fetchHealthAndDataset();
  if (!isStandaloneKiosk) {
    initObservers();
  }
})();

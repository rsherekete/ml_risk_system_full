/* Shared UI behaviour: navigation progress, panel refresh, and auto-update.
 *
 * Three problems this solves, all of which read as "the app is broken" when
 * unhandled:
 *
 *  1. A page that takes a second to render looks like a dead click. A top
 *     progress bar starts on navigation and completes on load, so there is
 *     always visible evidence the app heard you.
 *  2. Live panels went stale silently. Each now carries its own refresh
 *     control, a timestamp, and an auto-update toggle -- and the toggle is
 *     REMEMBERED, because a risk desk that wants live updates wants them on
 *     every visit, not once.
 *  3. Polling continued in hidden tabs, burning queries against five servers
 *     for nobody. Updates pause when the tab is hidden and resume on return.
 */

(function () {
  'use strict';

  // ---- navigation progress ------------------------------------------------
  const bar = document.createElement('div');
  bar.className = 'nav-progress';
  bar.innerHTML = '<div class="nav-progress-fill"></div>';
  document.addEventListener('DOMContentLoaded', () => document.body.appendChild(bar));

  let progressTimer = null;
  function startProgress() {
    const fill = bar.querySelector('.nav-progress-fill');
    bar.classList.add('on');
    let width = 0;
    clearInterval(progressTimer);
    // Ease toward 90% and wait: the true duration is unknown, and a bar that
    // reaches 100% before the page arrives is worse than one that pauses.
    progressTimer = setInterval(() => {
      width += (90 - width) * 0.12;
      fill.style.width = width + '%';
    }, 90);
  }
  function endProgress() {
    clearInterval(progressTimer);
    const fill = bar.querySelector('.nav-progress-fill');
    fill.style.width = '100%';
    setTimeout(() => { bar.classList.remove('on'); fill.style.width = '0'; }, 260);
  }
  window.addEventListener('beforeunload', startProgress);
  window.addEventListener('load', endProgress);
  window.addEventListener('pageshow', endProgress);

  document.addEventListener('click', (event) => {
    const link = event.target.closest('a[href]');
    if (!link) return;
    const url = link.getAttribute('href');
    if (!url || url.startsWith('#') || url.startsWith('javascript:')
        || link.target === '_blank' || url.endsWith('.csv')) return;
    startProgress();
  });
  document.addEventListener('submit', startProgress);

  // ---- live panel registry ------------------------------------------------
  // Panels register a loader; the registry owns timing, visibility and state so
  // each panel does not reimplement it.
  const panels = new Map();

  window.zfxLive = {
    register(name, loader, intervalMs) {
      panels.set(name, {
        loader, intervalMs: intervalMs || 15000,
        timer: null, lastRun: 0, running: false,
      });
      const auto = window.zfxLive.autoEnabled(name);
      if (auto) window.zfxLive.start(name);
      window.zfxLive.run(name);
    },

    autoEnabled(name) {
      try {
        const stored = localStorage.getItem('zfx.auto.' + name);
        return stored === null ? true : stored === '1';
      } catch (error) {
        return true;   // private browsing: default to live rather than frozen
      }
    },

    setAuto(name, enabled) {
      try { localStorage.setItem('zfx.auto.' + name, enabled ? '1' : '0'); } catch (error) {}
      enabled ? window.zfxLive.start(name) : window.zfxLive.stop(name);
      window.zfxLive.paint(name);
    },

    start(name) {
      const panel = panels.get(name);
      if (!panel || panel.timer) return;
      panel.timer = setInterval(() => {
        // Hidden tabs poll nobody: five servers were being queried for a
        // window nobody was looking at.
        if (document.hidden) return;
        window.zfxLive.run(name);
      }, panel.intervalMs);
    },

    stop(name) {
      const panel = panels.get(name);
      if (panel && panel.timer) { clearInterval(panel.timer); panel.timer = null; }
    },

    async run(name) {
      const panel = panels.get(name);
      if (!panel || panel.running) return;
      panel.running = true;
      window.zfxLive.paint(name, 'loading');
      try {
        await panel.loader();
        panel.lastRun = Date.now();
        window.zfxLive.paint(name);
      } catch (error) {
        window.zfxLive.paint(name, 'error');
      } finally {
        panel.running = false;
      }
    },

    paint(name, state) {
      const host = document.querySelector(`[data-live="${name}"]`);
      if (!host) return;
      const panel = panels.get(name);
      const auto = window.zfxLive.autoEnabled(name);
      const stamp = panel && panel.lastRun
        ? new Date(panel.lastRun).toLocaleTimeString() : '--';
      const status = state === 'loading' ? 'refreshing…'
                   : state === 'error' ? 'refresh failed'
                   : 'updated ' + stamp;
      host.innerHTML = `
        <button class="icon-btn ${state === 'loading' ? 'spinning' : ''}"
                title="Refresh now" onclick="zfxLive.run('${name}')">${ICONS.refresh}</button>
        <label class="auto-toggle" title="Update automatically every ${Math.round((panel?.intervalMs || 15000) / 1000)}s">
          <input type="checkbox" ${auto ? 'checked' : ''}
                 onchange="zfxLive.setAuto('${name}', this.checked)">
          <span>auto</span>
        </label>
        <span class="live-stamp ${state === 'error' ? 'neg' : ''}">${status}</span>`;
    },
  };

  document.addEventListener('visibilitychange', () => {
    // Catch up immediately on return rather than waiting a full interval.
    if (!document.hidden) {
      panels.forEach((panel, name) => {
        if (window.zfxLive.autoEnabled(name) && Date.now() - panel.lastRun > panel.intervalMs) {
          window.zfxLive.run(name);
        }
      });
    }
  });

  // ---- sortable, filterable tables ---------------------------------------
  // Every data table becomes sortable by clicking a header and filterable by a
  // box above it. A risk screen where the biggest loss is buried in row 400 and
  // cannot be sorted to the top is not usable, however correct the numbers are.
  function parseCell(text) {
    // Strip currency, thousands separators and percent so numeric columns sort
    // numerically rather than lexically ("$9" must not sort above "$1,000").
    const cleaned = String(text).replace(/[$,%\s]/g, '').replace(/[()]/g, '-');
    const value = parseFloat(cleaned);
    return Number.isFinite(value) && /[\d]/.test(cleaned) ? value : null;
  }

  function makeSortable(table) {
    if (table.dataset.enhanced) return;
    table.dataset.enhanced = '1';
    const head = table.tHead;
    const body = table.tBodies[0];
    if (!head || !body || body.rows.length < 2) return;

    Array.from(head.rows[0].cells).forEach((cell, index) => {
      cell.classList.add('sortable');
      cell.addEventListener('click', () => {
        const descending = cell.dataset.dir !== 'desc';
        Array.from(head.rows[0].cells).forEach(other => {
          other.dataset.dir = ''; other.classList.remove('sorted-asc', 'sorted-desc');
        });
        cell.dataset.dir = descending ? 'desc' : 'asc';
        cell.classList.add(descending ? 'sorted-desc' : 'sorted-asc');

        const rows = Array.from(body.rows);
        rows.sort((left, right) => {
          const a = left.cells[index]?.textContent.trim() ?? '';
          const b = right.cells[index]?.textContent.trim() ?? '';
          const na = parseCell(a), nb = parseCell(b);
          // Blanks always sort last, whichever direction -- an empty cell is
          // absence of data, not the smallest value.
          if (a === '' || a === '--') return 1;
          if (b === '' || b === '--') return -1;
          const cmp = (na !== null && nb !== null) ? na - nb : a.localeCompare(b);
          return descending ? -cmp : cmp;
        });
        rows.forEach(row => body.appendChild(row));
      });
    });
  }

  function addFilter(table) {
    if (table.dataset.filtered) return;
    const body = table.tBodies[0];
    if (!body || body.rows.length < 8) return;   // not worth it on short tables
    table.dataset.filtered = '1';

    const bar = document.createElement('div');
    bar.className = 'table-filter';
    bar.innerHTML = `<input type="text" placeholder="Filter rows…">
                     <span class="table-count"></span>`;
    const input = bar.querySelector('input');
    const count = bar.querySelector('.table-count');
    const total = body.rows.length;
    count.textContent = `${total} rows`;

    input.addEventListener('input', () => {
      const needle = input.value.toLowerCase();
      let shown = 0;
      Array.from(body.rows).forEach(row => {
        const match = !needle || row.textContent.toLowerCase().includes(needle);
        row.style.display = match ? '' : 'none';
        if (match) shown++;
      });
      count.textContent = needle ? `${shown} of ${total} rows` : `${total} rows`;
    });

    const host = table.closest('.table-scroll') || table;
    host.parentNode.insertBefore(bar, host);
  }

  function enhanceTables(root) {
    (root || document).querySelectorAll('table').forEach(table => {
      makeSortable(table);
      addFilter(table);
    });
  }
  window.zfxEnhanceTables = enhanceTables;
  document.addEventListener('DOMContentLoaded', () => enhanceTables());
  // Tables rendered by fetch land after DOMContentLoaded, so re-scan on change.
  const observer = new MutationObserver((records) => {
    records.forEach(record => {
      record.addedNodes.forEach(node => {
        if (node.nodeType === 1 && node.querySelector) enhanceTables(node);
      });
    });
  });
  document.addEventListener('DOMContentLoaded',
    () => observer.observe(document.body, { childList: true, subtree: true }));

  // ---- heavy async loads: themed loading bar + cancel + cache restore -----
  // A slow calculation (markout grid, forecast, universe) must (a) show a
  // themed bar so it never looks dead, (b) be cancellable so it stops eating
  // resources, and (c) on cancel OR error, RESTORE the last good result rather
  // than blanking the tab. On success the new result becomes that tab's default.
  const _lastGood = new Map();      // key -> last successfully rendered HTML/data

  function overlayFor(target) {
    let ov = target.querySelector(':scope > .zfx-busy');
    if (!ov) {
      ov = document.createElement('div');
      ov.className = 'zfx-busy';
      ov.innerHTML =
        '<div class="zfx-busy-bar"><i></i></div>'
        + '<div class="zfx-busy-row"><span class="zfx-busy-label">Working…</span>'
        + '<button type="button" class="zfx-busy-cancel">Cancel</button></div>';
      const cs = getComputedStyle(target);
      if (cs.position === 'static') target.style.position = 'relative';
      target.appendChild(ov);
    }
    return ov;
  }

  /**
   * zfxRun(key, target, work, opts)
   *  key    – cache identity for this tab/panel (string)
   *  target – element to overlay and to consider "the result area"
   *  work   – async (signal) => data ; does the fetch/calc, honouring `signal`
   *  opts   – { label, render(data), restore() }
   * Returns { promise, cancel }.
   */
  window.zfxRun = function (key, target, work, opts) {
    opts = opts || {};
    const controller = new AbortController();
    const ov = overlayFor(target);
    ov.querySelector('.zfx-busy-label').textContent = opts.label || 'Working…';
    ov.classList.add('on');
    const scrollY = window.scrollY;               // never jump the page on load
    let cancelled = false;

    function cleanup() { ov.classList.remove('on'); window.scrollTo(0, scrollY); }
    ov.querySelector('.zfx-busy-cancel').onclick = () => {
      cancelled = true;
      try { controller.abort(); } catch (e) {}
      cleanup();
      // restore the last good view for this tab, if any
      if (opts.restore) opts.restore(_lastGood.get(key));
    };

    const promise = Promise.resolve()
      .then(() => work(controller.signal))
      .then(data => {
        if (cancelled) return null;
        if (opts.render) opts.render(data);
        _lastGood.set(key, data);                 // new result = new default
        cleanup();
        return data;
      })
      .catch(err => {
        cleanup();
        if (cancelled || err.name === 'AbortError') return null;
        // real failure: keep the last good result on screen, note the error
        const prev = _lastGood.get(key);
        if (opts.restore && prev !== undefined) opts.restore(prev);
        if (opts.onError) opts.onError(err);
        return null;
      });
    return { promise, cancel: () => ov.querySelector('.zfx-busy-cancel').click() };
  };
  window.zfxLastGood = (key) => _lastGood.get(key);

  // ---- global scroll-jump guard ------------------------------------------
  // A stray href="#" (used as a JS hook) scrolls to the top on click, which
  // reads as the page "jumping". Neutralise it app-wide unless a link opts in
  // with data-allow-jump. Real navigation (href with a path) is untouched.
  document.addEventListener('click', (event) => {
    const a = event.target.closest('a[href="#"], a[href=""]');
    if (a && !a.dataset.allowJump) event.preventDefault();
  });

  // ---- inline icons -------------------------------------------------------
  // Inline SVG rather than an icon font or CDN: no extra request, no flash of
  // missing glyphs, and they inherit colour from the surrounding text.
  const ICONS = window.ICONS = {
    refresh: '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M14 8a6 6 0 1 1-1.8-4.3"/><path d="M14 2v4h-4"/></svg>',
    live: '<svg viewBox="0 0 16 16" width="10" height="10"><circle cx="8" cy="8" r="4" fill="currentColor"/></svg>',
    download: '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M8 2v8m0 0 3-3m-3 3L5 7"/><path d="M2.5 11v2A1.5 1.5 0 0 0 4 14.5h8a1.5 1.5 0 0 0 1.5-1.5v-2"/></svg>',
    warning: '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M8 2 1.5 13.5h13z"/><path d="M8 6.5v3.2M8 11.8v.1"/></svg>',
    chart: '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M2 13V3"/><path d="M2 13h12"/><path d="m4.5 10 3-3.5 2.5 2 3-4"/></svg>',
    shield: '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M8 1.8 3 3.6v4.1c0 3 2.1 5.6 5 6.5 2.9-.9 5-3.5 5-6.5V3.6z"/></svg>',
    search: '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="7" cy="7" r="4.5"/><path d="m10.5 10.5 3 3"/></svg>',
  };
})();

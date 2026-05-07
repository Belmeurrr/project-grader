// pg-nav.js — global navigation, used on every Project Grader page.
// Usage:
//   <link rel="stylesheet" href="shared.css" />
//   <script src="pg-nav.js"></script>
//   <script>PGNav.mount('capture.html', { subtitle: 'Capture wizard', crumbs: [['Workbench','workbench.html'], ['Capture','#']] });</script>

window.PGNav = {
  routes: [
    { path: 'workbench.html', label: 'Workbench' },
    { path: 'capture.html', label: 'Capture' },
    { path: 'cert.html', label: 'Cert', disabled: true },
    { path: 'card.html', label: 'Card', disabled: true },
    { path: 'queue.html', label: 'Queue', disabled: true },
    { path: 'insights.html', label: 'Insights', disabled: true },
    { path: 'system.html', label: 'System', disabled: true },
  ],
  mount(currentPath, opts = {}) {
    const nav = document.createElement('nav');
    nav.className = 'pg-nav';
    const linksHtml = this.routes.map(r => {
      const active = r.path === currentPath;
      const cls = 'pg-nav-link' + (active ? ' active' : '');
      const style = r.disabled ? 'opacity:0.4;cursor:not-allowed;' : '';
      const href = r.disabled ? 'javascript:void(0)' : r.path;
      const soon = r.disabled ? ' <span class="mono" style="font-size:8px;color:var(--ink-4);margin-left:4px;letter-spacing:0.1em;">SOON</span>' : '';
      const title = r.disabled ? 'title="Mock pending"' : '';
      return `<a class="${cls}" style="${style}" href="${href}" ${title}>${r.label}${soon}</a>`;
    }).join('');
    nav.innerHTML = `
      <div class="pg-nav-brand">
        <a href="workbench.html" style="display:flex;align-items:center;gap:10px;">
          <span class="pg-nav-mark">◰</span>
          <span style="display:flex;flex-direction:column;line-height:1.1;">
            <span class="pg-nav-title">Project Grader</span>
            <span class="pg-nav-sub">${opts.subtitle || 'The Workbench'}</span>
          </span>
        </a>
      </div>
      <div class="pg-nav-links">${linksHtml}</div>
      <div class="pg-nav-actions">
        <span class="mono" style="font-size:10px;color:var(--ink-3);">${opts.user || 'belme · personal'}</span>
        <span style="width:6px;height:6px;border-radius:3px;background:var(--lime);box-shadow:0 0 6px var(--lime);"></span>
      </div>
    `;
    document.body.insertBefore(nav, document.body.firstChild);

    if (opts.crumbs && opts.crumbs.length) {
      const cr = document.createElement('div');
      cr.className = 'pg-crumbs';
      cr.style.cssText = 'padding:10px 24px;border-bottom:1px solid var(--line);background:var(--bg-1);';
      cr.innerHTML = opts.crumbs.map((c, i) => {
        const last = i === opts.crumbs.length - 1;
        const sep = last ? '' : '<span class="sep">/</span>';
        if (last) return `<span class="here">${c[0]}</span>`;
        return `<a href="${c[1]}">${c[0]}</a>${sep}`;
      }).join(' ');
      nav.parentNode.insertBefore(cr, nav.nextSibling);
    }
  },
};

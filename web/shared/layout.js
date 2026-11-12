// Page frame shared by the portal, admin and client pages: footer and navbar highlighting.
export function initLayout() {
  const f = document.createElement('footer');
  f.className = 'site-footer';
  f.innerHTML = `<div class="accent-line"></div><div class="in">
    <div><h4>Incremental Learning in a Federated Setup for Health IoT Devices</h4>
      <p>B.Tech project, Department of Computer Science and Engineering, The LNM Institute of Information Technology, Jaipur.</p>
      <p style="margin-top:8px">Pose from MediaPipe in the browser · Bi-LSTM with attention · replay + EWC · FedProx, FedAdam and Ditto · Shapley contributions.</p></div>
    <div><h4>Team</h4><ul><li>Aditya Sharma</li><li>Hussain Haidary</li><li>Dhruv Sangal</li><li>Sarrah Sadriwala</li></ul></div>
    <div><h4>Guides</h4><ul><li>Dr. Anubhav Shivhare</li><li>Dr. Lal Upendra Pratap Singh</li></ul><p style="margin-top:8px">Department of CSE</p></div>
  </div><div class="bottom">© 2026 The LNM Institute of Information Technology, Jaipur · BTP project</div>`;
  document.body.appendChild(f);

  // highlight the navbar link of the section in view
  const links = [...document.querySelectorAll('.navbar a[href^="#"]')];
  const byId = new Map(links.map(a => [a.getAttribute('href').slice(1), a]));
  const secs = [...byId.keys()].map(id => document.getElementById(id)).filter(Boolean);
  const mark = id => links.forEach(a => a.classList.toggle('active', a === byId.get(id)));
  if ('IntersectionObserver' in window && secs.length) {
    const seen = new Map();
    const io = new IntersectionObserver(es => {
      es.forEach(e => seen.set(e.target.id, e.isIntersecting ? e.intersectionRatio : 0));
      const best = secs.find(s => (seen.get(s.id) || 0) > 0);
      if (best) mark(best.id);
    }, { rootMargin: '-130px 0px -55% 0px', threshold: [0, .01, .2] });
    secs.forEach(s => io.observe(s));
  }
  links.forEach(a => a.addEventListener('click', () => mark(a.getAttribute('href').slice(1))));
}

const e = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
export function userBox(el, me, onLogout) {
  el.innerHTML = `<span class="avatar">${e((me.username || '?')[0])}</span><span>${e(me.username)}<br><span style="color:#8fb0d6">${e(me.role)}${me.node_id ? ' · node ' + me.node_id : ''}</span></span><button id="logout">Sign out</button>`;
  el.querySelector('#logout').onclick = onLogout;
}

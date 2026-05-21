/**
 * Centralized Toast Notifications
 */
export function toast(msg, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) {
    console.warn("Toast container not found in DOM");
    return;
  }
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

/**
 *Centralized Upload Log helper
 */
export function log(msg, type = 'info') {
  const box = document.getElementById('upload-log');
  if (!box) return; // Silent return if upload log container isn't rendered or active
  const span = document.createElement('span');
  span.className = `log-${type}`;
  span.textContent = msg + '\n';
  box.appendChild(span);
  box.scrollTop = box.scrollHeight;
}

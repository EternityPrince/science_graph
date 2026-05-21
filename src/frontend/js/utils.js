/**
 * Escape HTML special characters.
 */
export function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Escape single quotes for embedding in inline JS attributes (like focusAndDetails('ID')).
 */
export function escapeSingleQuotes(str) {
  if (str === null || str === undefined) return '';
  return String(str).replace(/'/g, "\\'");
}

/**
 * Converts text into a safe alphanumeric lowercase slug matching Python's backend implementation.
 */
export function slugify(text) {
  if (!text) return '';
  return text.toString().toLowerCase().trim()
    .replace(/[^a-z0-9\s-]/g, '')
    .replace(/[\s-]+/g, '_');
}

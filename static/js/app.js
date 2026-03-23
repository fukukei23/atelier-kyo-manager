// app.js - Atelier Kyo Manager カスタムJS

/* =============================================
   テーマ切り替え（ライト/ダークモード）
   ============================================= */
(function initTheme() {
  const stored = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (stored === 'dark' || (!stored && prefersDark)) {
    document.documentElement.classList.add('dark');
  }
  updateThemeIcons();
})();

function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.classList.toggle('dark');
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
  updateThemeIcons();
}

function updateThemeIcons() {
  const isDark = document.documentElement.classList.contains('dark');
  const sun = document.getElementById('icon-sun');
  const moon = document.getElementById('icon-moon');
  if (sun) sun.classList.toggle('hidden', !isDark);
  if (moon) moon.classList.toggle('hidden', isDark);
}

document.addEventListener('DOMContentLoaded', function() {
  const btn = document.getElementById('theme-toggle');
  if (btn) btn.addEventListener('click', toggleTheme);
});

/* =============================================
   削除確認モーダル
   ============================================= */
/**
 * 削除確認モーダルを表示
 * @param {string} productId - 削除する商品的ID
 * @param {string} productName - 商品名（表示用）
 */
function showDeleteModal(productId, productName) {
  const modal = document.getElementById('delete-modal');
  const nameSpan = document.getElementById('delete-product-name');
  const form = document.getElementById('delete-form');

  if (!modal) {
    // モーダルがない場合はブラウザのconfirmを使用（フォールバック）
    if (confirm(`「${productName}」を削除しますか？`)) {
      const form = document.createElement('form');
      form.method = 'POST';
      form.action = `/products/${productId}/delete`;
      document.body.appendChild(form);
      form.submit();
    }
    return;
  }

  if (nameSpan) nameSpan.textContent = productName;
  if (form) form.action = `/product/delete/${productId}`;

  modal.showModal();
}

/**
 * モーダルを閉じる
 */
function closeDeleteModal() {
  const modal = document.getElementById('delete-modal');
  if (modal) modal.close();
}

/**
 * 削除フォームを送信
 */
function submitDelete() {
  const form = document.getElementById('delete-form');
  if (form) form.submit();
}

/**
 * モーダル外クリックで閉じる
 */
document.addEventListener('DOMContentLoaded', function() {
  const modal = document.getElementById('delete-modal');
  if (modal) {
    modal.addEventListener('click', function(e) {
      if (e.target === modal) closeDeleteModal();
    });
  }
});

// app.js - Atelier Kyo Manager カスタムJS

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

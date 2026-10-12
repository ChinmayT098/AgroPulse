/**
 * AgroPulse - Agricultural Data Analysis & Crop Yield Prediction
 * Frontend JavaScript interactions
 */

document.addEventListener('DOMContentLoaded', () => {
  // Add subtle ripple / click effect to buttons
  const buttons = document.querySelectorAll('.btn');
  buttons.forEach(btn => {
    btn.addEventListener('mousedown', () => {
      btn.style.transform = 'scale(0.98)';
    });
    btn.addEventListener('mouseup', () => {
      btn.style.transform = '';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.transform = '';
    });
  });

  // Table row click highlighting for easy reading
  const tableRows = document.querySelectorAll('.data-table tbody tr');
  tableRows.forEach(row => {
    row.addEventListener('click', () => {
      tableRows.forEach(r => r.style.backgroundColor = '');
      row.style.backgroundColor = '#ecfdf5';
    });
  });
});

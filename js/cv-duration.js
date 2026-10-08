(function () {
  'use strict';

  // Dates use month precision; both the start and end month count.
  function monthIndex(value) {
    var match = /^(\d{4})-(0[1-9]|1[0-2])$/.exec(value || '');
    return match ? Number(match[1]) * 12 + Number(match[2]) - 1 : null;
  }

  function updateDurations() {
    var now = new Date();
    var currentMonth = now.getFullYear() * 12 + now.getMonth();

    document.querySelectorAll('.cv-grid dt[data-start]').forEach(function (entry) {
      var start = monthIndex(entry.dataset.start);
      var end = entry.hasAttribute('data-end') ? monthIndex(entry.dataset.end) : currentMonth;
      var duration = entry.querySelector('.cv-duration');
      if (start === null || end === null || end < start) {
        if (duration) duration.remove();
        return;
      }
      if (!duration) {
        duration = document.createElement('span');
        duration.className = 'cv-duration';
        entry.appendChild(duration);
      }
      var totalMonths = end - start + 1;
      var years = Math.floor(totalMonths / 12);
      var months = totalMonths % 12;
      var parts = [];
      if (years) parts.push(years + (years === 1 ? ' year' : ' years'));
      if (months) parts.push(months + (months === 1 ? ' month' : ' months'));
      duration.textContent = parts.join(' ');
    });
  }

  updateDurations();
  // Refresh when returning to a tab that may have crossed into a new month.
  document.addEventListener('visibilitychange', function () {
    if (!document.hidden) updateDurations();
  });
}());

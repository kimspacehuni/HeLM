/* Keep only the recording the visitor is watching active. */
(function () {
  'use strict';
  const videos = Array.from(document.querySelectorAll('video'));
  videos.forEach((video) => {
    video.addEventListener('play', () => {
      videos.forEach((other) => { if (other !== video && !other.paused) other.pause(); });
    });
  });
  document.querySelectorAll('details').forEach((details) => {
    if (!details.querySelector('video')) return;
    details.addEventListener('toggle', () => {
      if (!details.open) details.querySelectorAll('video').forEach((video) => video.pause());
    });
  });
})();

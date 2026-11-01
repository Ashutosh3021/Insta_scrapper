async function getReels(username, mode = "both") {
  // normalize: "r" = reels, "p" = posts/pics, anything else = both
  mode = (mode || "both").toLowerCase();
  if (mode !== "r" && mode !== "p") mode = "both";

  // Reels tab for "r", main profile grid for "p" and both
  const path = mode === "r" ? `/${username}/reels/` : `/${username}/`;
  const onRightPage = mode === "r"
    ? location.pathname.startsWith(path)
    : location.pathname === path;

  if (!onRightPage) {
    location.href = `https://www.instagram.com${path}`;
    console.log(`Navigated. Run getReels("${username}"${mode === "both" ? "" : `, "${mode}"`}) again once the page loads.`);
    return;
  }

  const found = new Map(); // key: "type/code" -> clean URL
  let lastCount = 0, stale = 0;

  while (stale < 5) {
    document.querySelectorAll('main a[href*="/reel/"], main a[href*="/p/"]').forEach(a => {
      const m = a.getAttribute('href').match(/\/(reel|p)\/([A-Za-z0-9_-]+)/);
      if (!m) return;
      const [, type, code] = m;
      if (mode === "r" && type !== "reel") return;
      if (mode === "p" && type !== "p") return;
      found.set(`${type}/${code}`, `https://www.instagram.com/${type}/${code}/`);
    });

    window.scrollTo(0, document.body.scrollHeight);
    await new Promise(r => setTimeout(r, 1500));

    if (found.size === lastCount) stale++;
    else { stale = 0; lastCount = found.size; }
  }

  const links = [...found.values()];
  const label = mode === "r" ? "reels" : mode === "p" ? "posts" : "all";
  const blob = new Blob([links.join('\n')], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${username}_${label}.txt`;
  a.click();
  console.log(`Saved ${links.length} unique ${label} links.`);
}

// getReels("userid")        -> reels + posts
// getReels("userid", "r")   -> reels only
// getReels("userid", "p")   -> posts/pics only
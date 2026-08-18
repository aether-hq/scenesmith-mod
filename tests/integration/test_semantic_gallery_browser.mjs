import assert from 'node:assert/strict';

const playwrightModule = process.env.PLAYWRIGHT_MODULE ?? 'playwright';
const { chromium } = await import(playwrightModule);
const galleryUrl = process.env.SCENESMITH_GALLERY_URL
  ?? 'http://127.0.0.1:8766/viewer.html';

async function waitForServer(url, timeoutMilliseconds = 60_000) {
  const deadline = Date.now() + timeoutMilliseconds;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // The gallery compiler/server may still be starting.
    }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  throw new Error(`gallery server did not become ready: ${url}`);
}

await waitForServer(galleryUrl);
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errors = [];
  page.on('console', message => {
    if (message.type() === 'error') errors.push(`console: ${message.text()}`);
  });
  page.on('pageerror', error => errors.push(`page: ${error.message}`));
  await page.goto(galleryUrl, { waitUntil: 'networkidle' });
  await page.locator('#loading.hidden').waitFor({ timeout: 60_000 });

  const buttons = page.locator('.scene-button');
  const sceneCount = await buttons.count();
  assert.ok(sceneCount >= 3, `expected trials and a control; got ${sceneCount}`);
  const rendered = [];
  for (let index = 0; index < sceneCount; index += 1) {
    const button = buttons.nth(index);
    await button.click();
    await page.locator('#loading.hidden').waitFor({ timeout: 60_000 });
    await page.waitForTimeout(250);
    const sceneId = await button.getAttribute('data-scene-id');
    const title = await page.locator('#scene-title').textContent();
    const renderStats = await page.locator('#scene-meta').evaluate(() => (
      window.__SCENESMITH_GALLERY_ACTIVE_RENDER_STATS__
    ));
    const pixels = await page.locator('canvas').evaluate(canvas => {
      const probe = document.createElement('canvas');
      probe.width = 128;
      probe.height = 80;
      const context = probe.getContext('2d', { willReadFrequently: true });
      context.drawImage(canvas, 0, 0, probe.width, probe.height);
      const data = context.getImageData(0, 0, probe.width, probe.height).data;
      let luminance = 0;
      let dark = 0;
      for (let offset = 0; offset < data.length; offset += 4) {
        const value = (
          0.2126 * data[offset]
          + 0.7152 * data[offset + 1]
          + 0.0722 * data[offset + 2]
        );
        luminance += value;
        if (value < 8) dark += 1;
      }
      const count = data.length / 4;
      return { mean: luminance / count, darkFraction: dark / count };
    });
    assert.ok(pixels.mean > 2, `${sceneId} initial view is effectively black`);
    assert.ok(pixels.mean < 245, `${sceneId} initial view is effectively white`);
    assert.ok(
      pixels.darkFraction < 0.98,
      `${sceneId} has no meaningfully illuminated architecture`,
    );
    if (sceneId === 'original_scenesmith_bar') {
      assert.equal(renderStats.representation, 'full_fidelity_gltf');
      assert.ok(renderStats.meshes >= 280, `full bar loaded only ${renderStats.meshes} meshes`);
      assert.ok(renderStats.triangles >= 187_086, `full bar loaded only ${renderStats.triangles} triangles`);
      assert.ok(renderStats.materials >= 50, `full bar loaded only ${renderStats.materials} materials`);
      assert.ok(renderStats.textures >= 50, `full bar loaded only ${renderStats.textures} textures`);
    }
    rendered.push({ sceneId, title, ...pixels, ...renderStats });
  }
  assert.deepEqual(errors, []);
  process.stdout.write(`${JSON.stringify({ sceneCount, rendered })}\n`);
} finally {
  await browser.close();
}

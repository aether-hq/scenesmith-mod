#!/usr/bin/env node

import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';

const galleryUrl = process.env.SCENESMITH_GALLERY_URL
  ?? 'http://127.0.0.1:8766/viewer.html';
const prisonUrl = process.env.SCENESMITH_PRISON_URL
  ?? 'http://127.0.0.1:8765/viewer.html';
const outputDirectory = path.resolve(
  process.env.SCENESMITH_RENDER_OUTPUT ?? 'examples/rendered_samples',
);
const reportPath = path.resolve(
  process.env.SCENESMITH_RENDER_REPORT
    ?? 'examples/sample-render-report.json',
);
const playwrightModule = process.env.PLAYWRIGHT_MODULE ?? 'playwright';
const { chromium } = await import(playwrightModule);

async function waitForServer(url, timeoutMilliseconds = 60_000) {
  const deadline = Date.now() + timeoutMilliseconds;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // The clean compiler/server may still be starting.
    }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  throw new Error(`sample server did not become ready: ${url}`);
}

async function canvasMetrics(canvas) {
  return canvas.evaluate(element => {
    const probe = document.createElement('canvas');
    probe.width = 160;
    probe.height = 100;
    const context = probe.getContext('2d', { willReadFrequently: true });
    context.drawImage(element, 0, 0, probe.width, probe.height);
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
    return { meanLuminance: luminance / count, darkFraction: dark / count };
  });
}

function assertVisibleArchitecture(id, metrics) {
  const detail = JSON.stringify(metrics);
  assert.ok(
    metrics.meanLuminance > 2,
    `${id} render is effectively black: ${detail}`,
  );
  assert.ok(
    metrics.meanLuminance < 245,
    `${id} render is effectively white: ${detail}`,
  );
  assert.ok(
    metrics.darkFraction < 0.98,
    `${id} has no meaningfully illuminated architecture: ${detail}`,
  );
}

function captureErrors(page) {
  const errors = [];
  page.on('console', message => {
    if (message.type() === 'error') errors.push(`console: ${message.text()}`);
  });
  page.on('pageerror', error => errors.push(`page: ${error.message}`));
  return errors;
}

async function renderGallery(browser) {
  await waitForServer(galleryUrl);
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = captureErrors(page);
  const rendered = [];
  try {
    await page.goto(galleryUrl, { waitUntil: 'networkidle' });
    await page.locator('#loading.hidden').waitFor({ timeout: 60_000 });
    const buttons = page.locator('.scene-button');
    const count = await buttons.count();
    assert.ok(count >= 4, `expected four gallery entries; got ${count}`);
    for (let index = 0; index < count; index += 1) {
      const button = buttons.nth(index);
      await button.click();
      await page.locator('#loading.hidden').waitFor({ timeout: 60_000 });
      await page.waitForTimeout(300);
      const sceneId = await button.getAttribute('data-scene-id');
      const canvas = page.locator('canvas');
      const metrics = await canvasMetrics(canvas);
      assertVisibleArchitecture(sceneId, metrics);
      const renderStats = await page.locator('#scene-meta').evaluate(() => (
        window.__SCENESMITH_GALLERY_ACTIVE_RENDER_STATS__
      ));
      const screenshot = path.join(outputDirectory, `gallery-${sceneId}.png`);
      await canvas.screenshot({ path: screenshot });
      rendered.push({
        id: sceneId,
        target: 'semantic_gallery',
        status: 'passed',
        screenshot,
        metrics,
        renderStats,
      });
    }
    assert.deepEqual(errors, []);
    return rendered;
  } finally {
    await page.close();
  }
}

async function renderPrison(browser) {
  await waitForServer(prisonUrl);
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = captureErrors(page);
  try {
    await page.goto(prisonUrl, { waitUntil: 'networkidle' });
    await page.locator('#loading.done').waitFor({ timeout: 60_000 });
    await page.waitForTimeout(500);
    const canvas = page.locator('canvas');
    const views = [];
    for (const place of ['prison', 'breach', 'tunnel', 'cavern']) {
      await page.locator(`[data-place="${place}"]`).click({ force: true });
      await page.waitForTimeout(250);
      const metrics = await canvasMetrics(canvas);
      const screenshot = path.join(outputDirectory, `prison-${place}.png`);
      await canvas.screenshot({ path: screenshot });
      assertVisibleArchitecture(`prison_escape_${place}`, metrics);
      views.push({ place, screenshot, metrics });
      await page.keyboard.press('Escape');
    }
    assert.deepEqual(errors, []);
    return [{
      id: 'prison_escape_long_way_out',
      target: 'prison_escape',
      status: 'passed',
      views,
    }];
  } finally {
    await page.close();
  }
}

await fs.mkdir(outputDirectory, { recursive: true });
const browser = await chromium.launch({ headless: true });
let renders;
try {
  renders = [
    ...await renderGallery(browser),
    ...await renderPrison(browser),
  ];
} finally {
  await browser.close();
}

const report = {
  schema_version: 1,
  provider: 'chromium/webgl2',
  renders,
};
const temporaryPath = `${reportPath}.${process.pid}.tmp`;
await fs.mkdir(path.dirname(reportPath), { recursive: true });
await fs.writeFile(temporaryPath, `${JSON.stringify(report, null, 2)}\n`);
await fs.rename(temporaryPath, reportPath);
process.stdout.write(`${JSON.stringify(report)}\n`);

"""Browser stealth / anti-fingerprint init scripts."""

from __future__ import annotations

from playwright.async_api import BrowserContext


async def setup_stealth_init_scripts(context: BrowserContext) -> None:
    # --- Baseline init script ---
    await context.add_init_script(
        """try { localStorage.setItem('a11y-contrast','off'); localStorage.setItem('high-contrast','off'); } catch(e){} Object.defineProperty(navigator, 'language', {get: () => 'en-GB'}); Object.defineProperty(navigator, 'languages', {get: () => ['en-GB','en']}); (function(){ const _rz=Intl.DateTimeFormat.prototype.resolvedOptions; Intl.DateTimeFormat.prototype.resolvedOptions=function(){const o=_rz.call(this); o.timeZone='UTC'; return o;}; })();"""
    )

    # --- Stealthish patches to reduce headless fingerprinting ---
    await context.add_init_script(r"""
      (() => {
        try {
          const rand = (min, max) => Math.random() * (max - min) + min;
          const jitter = (base, span) => base + rand(-span, span);

          // navigator.* tweaks
          const nav = navigator;
          if (nav) {
            const lang = (nav.language || 'en-GB');
            const langs = Array.isArray(nav.languages) && nav.languages.length ? nav.languages : ['en-GB','en'];
            Object.defineProperty(nav, 'webdriver', { get: () => undefined });
            Object.defineProperty(nav, 'hardwareConcurrency', { get: () => 8 });
            Object.defineProperty(nav, 'deviceMemory', { get: () => 8 });
            Object.defineProperty(nav, 'language', { get: () => lang });
            Object.defineProperty(nav, 'languages', { get: () => langs });
            Object.defineProperty(nav, 'maxTouchPoints', { get: () => 0 });
            Object.defineProperty(nav, 'platform', { get: () => 'Win32' });
          }

          // Canvas noise
          const patchCanvas = (proto) => {
            if (!proto) return;
            const toDataURL = proto.toDataURL;
            proto.toDataURL = function(...args) {
              const ctx = this.getContext && this.getContext('2d');
              if (ctx) {
                const shift = () => (Math.random() - 0.5) * 2;
                ctx.fillStyle = `rgba(${128+shift()},${128+shift()},${128+shift()},0.01)`;
                ctx.fillRect(0, 0, 2, 2);
              }
              return toDataURL.apply(this, args);
            };
          };
          if (typeof HTMLCanvasElement !== 'undefined' && HTMLCanvasElement.prototype) {
            patchCanvas(HTMLCanvasElement.prototype);
          }
          if (typeof OffscreenCanvas !== 'undefined' && OffscreenCanvas.prototype) {
            patchCanvas(OffscreenCanvas.prototype);
          }

          // WebGL noise
          const patchWebGL = (proto) => {
            if (!proto) return;
            const getParameter = proto.getParameter;
            proto.getParameter = function(param) {
              // Vendor/renderer slightly jittered
              const VENDOR = 0x1F00, RENDERER = 0x1F01;
              if (param === VENDOR) {
                const v = getParameter.call(this, param);
                return typeof v === 'string' ? v.replace(/Google Inc\./, 'Google LLC') : v;
              }
              if (param === RENDERER) {
                const r = getParameter.call(this, param);
                return typeof r === 'string' ? r.replace(/ANGLE \(|\)/g, '') : r;
              }
              return getParameter.call(this, param);
            };
          };
          if (typeof WebGLRenderingContext !== 'undefined' && WebGLRenderingContext.prototype) {
            patchWebGL(WebGLRenderingContext.prototype);
          }
          if (typeof WebGL2RenderingContext !== 'undefined' && WebGL2RenderingContext.prototype) {
            patchWebGL(WebGL2RenderingContext.prototype);
          }

          // AudioContext fingerprint jitter
          const patchAudio = (Cls) => {
            if (!Cls || !Cls.prototype) return;
            const getFloatFrequencyData = Cls.prototype.getFloatFrequencyData;
            if (getFloatFrequencyData) {
              Cls.prototype.getFloatFrequencyData = function(arr) {
                const res = getFloatFrequencyData.call(this, arr);
                for (let i = 0; i < arr.length; i += Math.floor(arr.length / 8) || 1) {
                  arr[i] = arr[i] * (0.99 + Math.random() * 0.02);
                }
                return res;
              };
            }
          };
          if (typeof AnalyserNode !== 'undefined') {
            patchAudio(AnalyserNode);
          }

          // Fonts enumeration shield
          if (typeof Navigator !== 'undefined' && Navigator.prototype) {
            const origFonts = Navigator.prototype.fonts;
            if (origFonts) {
              Navigator.prototype.fonts = function() {
                const it = origFonts.apply(this, arguments);
                if (it && typeof it.status === 'string') return it;
                return {
                  status: 'loaded',
                  check: () => true,
                  load: () => Promise.resolve(),
                  values: () => [].values()
                };
              };
            }
          }

        } catch (e) { /* swallow */ }
      })();
    """)

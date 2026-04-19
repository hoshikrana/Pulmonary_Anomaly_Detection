/**
 * app/static/js/utils.js
 * ──────────────────────
 * Shared utilities used by upload.js and result.js.
 * No DOM manipulation here — pure functions only.
 */

"use strict";

const Utils = (() => {

  /**
   * POST a FormData object to a URL and return parsed JSON.
   * Throws an Error with a user-facing message on failure.
   *
   * @param {string}   url
   * @param {FormData} formData
   * @returns {Promise<object>}
   */
  async function postFormData(url, formData) {
    const response = await fetch(url, {
      method: "POST",
      body:   formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || `Request failed (${response.status})`);
    }
    return data;
  }

  /**
   * Format a decimal score (0–1) as a percentage string.
   * e.g. 0.734 → "73.4%"
   */
  function formatScore(score) {
    return (score * 100).toFixed(1) + "%";
  }

  /**
   * Format raw MSE to 6 decimal places for display.
   */
  function formatMSE(mse) {
    return parseFloat(mse).toFixed(6);
  }

  /**
   * Show or hide an element by toggling the "hidden" class.
   */
  function show(el) { el.classList.remove("hidden"); }
  function hide(el) { el.classList.add("hidden");    }

  /**
   * Store result data in sessionStorage so result.html can read it
   * after the page navigation.
   */
  function storeResult(data, originalDataUrl) {
    sessionStorage.setItem("pad_result",   JSON.stringify(data));
    sessionStorage.setItem("pad_original", originalDataUrl);
  }

  function loadResult() {
    const result   = sessionStorage.getItem("pad_result");
    const original = sessionStorage.getItem("pad_original");
    if (!result) return null;
    return { data: JSON.parse(result), originalDataUrl: original };
  }

  function clearResult() {
    sessionStorage.removeItem("pad_result");
    sessionStorage.removeItem("pad_original");
  }

  return { postFormData, formatScore, formatMSE, show, hide,
           storeResult, loadResult, clearResult };
})();
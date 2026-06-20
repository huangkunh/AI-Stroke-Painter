/**
 * stroke-processor.js — Web Worker for compute-intensive stroke preprocessing.
 *
 * Runs in a dedicated worker thread so the main thread stays responsive
 * during painting playback. Supports three jobs:
 *
 *   1. "smooth"   — Catmull-Rom path smoothing on a flat point array.
 *   2. "resample" — Resample a polyline to a uniform point count.
 *   3. "preprocess" — Batch-preprocess a list of stroke instructions:
 *                     extracts line strokes, smooths + resamples each, and
 *                     returns a map { instructionIndex -> processedPoints }.
 *
 * Message protocol (main thread -> worker):
 *   { id: <number>, job: "smooth"|"resample"|"preprocess", payload: {...} }
 *
 * Message protocol (worker -> main thread):
 *   { id: <number>, result: <any>, error: <string|null> }
 *
 * The worker is intentionally dependency-free (no imports) so it can be
 * loaded via `new Worker('stroke-processor.js')` without a bundler.
 */
'use strict';

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/**
 * Catmull-Rom spline interpolation.
 * @param {number[]} points  Flat array [x0,y0,x1,y1,...]
 * @param {number}   segments  Subdivisions per segment (default 8)
 * @returns {number[]} Smoothed flat point array.
 */
function catmullRomSmooth(points, segments) {
  segments = segments || 8;
  if (points.length < 4) return points.slice();

  // Pad endpoints by duplicating them so the curve passes through all points.
  var pts = points.slice();
  pts.unshift(points[1], points[0]);
  pts.push(points[points.length - 2], points[points.length - 1]);

  var out = [];
  for (var i = 2; i < pts.length - 4; i += 2) {
    var p0x = pts[i - 2], p0y = pts[i - 1];
    var p1x = pts[i],     p1y = pts[i + 1];
    var p2x = pts[i + 2], p2y = pts[i + 3];
    var p3x = pts[i + 4], p3y = pts[i + 5];

    for (var s = 0; s < segments; s++) {
      var t = s / segments;
      var t2 = t * t, t3 = t2 * t;
      var x = 0.5 * ((2 * p1x) +
                     (-p0x + p2x) * t +
                     (2 * p0x - 5 * p1x + 4 * p2x - p3x) * t2 +
                     (-p0x + 3 * p1x - 3 * p2x + p3x) * t3);
      var y = 0.5 * ((2 * p1y) +
                     (-p0y + p2y) * t +
                     (2 * p0y - 5 * p1y + 4 * p2y - p3y) * t2 +
                     (-p0y + 3 * p1y - 3 * p2y + p3y) * t3);
      out.push(x, y);
    }
  }
  // Append the final point
  out.push(points[points.length - 2], points[points.length - 1]);
  return out;
}

/**
 * Resample a polyline so it has exactly `targetCount` points, uniformly
 * spaced along the arc length.
 * @param {number[]} points  Flat [x0,y0,x1,y1,...]
 * @param {number} targetCount  Desired number of output points
 * @returns {number[]} Resampled flat array.
 */
function resamplePolyline(points, targetCount) {
  if (points.length < 4 || targetCount < 2) return points.slice();

  // Compute cumulative arc lengths
  var coords = [];
  var cumLen = [0];
  for (var i = 0; i < points.length; i += 2) {
    coords.push([points[i], points[i + 1]]);
  }
  for (var j = 1; j < coords.length; j++) {
    var dx = coords[j][0] - coords[j - 1][0];
    var dy = coords[j][1] - coords[j - 1][1];
    cumLen.push(cumLen[j - 1] + Math.sqrt(dx * dx + dy * dy));
  }
  var totalLen = cumLen[cumLen.length - 1];
  if (totalLen === 0) return points.slice();

  var out = [];
  var segIdx = 0;
  for (var k = 0; k < targetCount; k++) {
    var targetLen = (k / (targetCount - 1)) * totalLen;
    // Advance segIdx until targetLen falls in [cumLen[segIdx], cumLen[segIdx+1]]
    while (segIdx < cumLen.length - 2 && cumLen[segIdx + 1] < targetLen) {
      segIdx++;
    }
    var segStart = cumLen[segIdx];
    var segEnd = cumLen[segIdx + 1];
    var segLen = segEnd - segStart;
    var t = segLen === 0 ? 0 : (targetLen - segStart) / segLen;
    var x = coords[segIdx][0] + t * (coords[segIdx + 1][0] - coords[segIdx][0]);
    var y = coords[segIdx][1] + t * (coords[segIdx + 1][1] - coords[segIdx][1]);
    out.push(x, y);
  }
  return out;
}

/**
 * Extract the flat point array from a "line" instruction.
 * Handles both brush 0 (2 values/point: x,y) and brush 5 (3 values/point:
 * x,y,pressure). Returns only the x,y pairs.
 * @param {Array} inst  e.g. ["line", brushId, x1,y1,p1, x2,y2,p2, ...]
 * @returns {number[]} Flat [x0,y0,x1,y1,...]
 */
function extractPoints(inst) {
  var brushId = inst[1];
  var rest = inst.slice(2);
  var stride = brushId === 5 ? 3 : 2;
  var pts = [];
  for (var i = 0; i < rest.length; i += stride) {
    pts.push(rest[i], rest[i + 1]);
  }
  return pts;
}

// ---------------------------------------------------------------------------
// Job dispatcher
// ---------------------------------------------------------------------------

self.onmessage = function (e) {
  var msg = e.data;
  var id = msg.id;
  var job = msg.job;
  var payload = msg.payload || {};
  var result, err = null;

  try {
    if (job === 'smooth') {
      result = catmullRomSmooth(payload.points, payload.segments || 8);
    } else if (job === 'resample') {
      result = resamplePolyline(payload.points, payload.targetCount || 16);
    } else if (job === 'preprocess') {
      // payload.instructions: array of engine instructions
      // Returns: { index: [smoothed points], ... }
      result = {};
      var instrs = payload.instructions;
      var smoothSegs = payload.segments || 8;
      var resampleCount = payload.resampleCount || 0; // 0 = no resample
      for (var i = 0; i < instrs.length; i++) {
        var inst = instrs[i];
        if (!Array.isArray(inst) || inst[0] !== 'line') continue;
        var pts = extractPoints(inst);
        if (pts.length < 4) continue;
        var smoothed = catmullRomSmooth(pts, smoothSegs);
        if (resampleCount > 0) {
          smoothed = resamplePolyline(smoothed, resampleCount);
        }
        result[i] = smoothed;
      }
    } else if (job === 'batch') {
      // Process a sub-range of instructions (for parallel processing).
      // payload: { instructions, startIdx, endIdx, segments, resampleCount }
      // Returns: { startIdx, results: { index: [smoothed points], ... } }
      result = { startIdx: payload.startIdx, results: {} };
      var batchInstrs = payload.instructions;
      var bStart = payload.startIdx || 0;
      var bEnd = payload.endIdx || batchInstrs.length;
      var bSegs = payload.segments || 8;
      var bResample = payload.resampleCount || 0;
      for (var bi = bStart; bi < bEnd; bi++) {
        var bInst = batchInstrs[bi];
        if (!Array.isArray(bInst) || bInst[0] !== 'line') continue;
        var bPts = extractPoints(bInst);
        if (bPts.length < 4) continue;
        var bSmoothed = catmullRomSmooth(bPts, bSegs);
        if (bResample > 0) bSmoothed = resamplePolyline(bSmoothed, bResample);
        result.results[bi] = bSmoothed;
      }
    } else {
      err = 'Unknown job: ' + job;
    }
  } catch (e2) {
    err = e2.message;
  }

  self.postMessage({ id: id, result: result, error: err });
};

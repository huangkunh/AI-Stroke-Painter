/**
 * test-stroke-processor.js — Node.js unit tests for stroke-processor.js
 *
 * Since stroke-processor.js is a Web Worker, we can't `require()` it directly.
 * Instead we extract the pure geometry functions and test them.
 *
 * Run with: node tests/test-stroke-processor.js
 */
'use strict';

// We test the geometry functions by re-implementing them here (they're
// dependency-free). In production, stroke-processor.js runs in a Worker.
// This test verifies the algorithm correctness.

function catmullRomSmooth(points, segments) {
  segments = segments || 8;
  if (points.length < 4) return points.slice();
  var result = [];
  var n = points.length / 2;
  for (var i = 0; i < n - 1; i++) {
    var p0x = i === 0 ? points[0] : points[(i - 1) * 2];
    var p0y = i === 0 ? points[1] : points[(i - 1) * 2 + 1];
    var p1x = points[i * 2], p1y = points[i * 2 + 1];
    var p2x = points[(i + 1) * 2], p2y = points[(i + 1) * 2 + 1];
    var p3x = (i + 2) < n ? points[(i + 2) * 2] : p2x;
    var p3y = (i + 2) < n ? points[(i + 2) * 2 + 1] : p2y;
    for (var t = 0; t < segments; t++) {
      var s = t / segments;
      var s2 = s * s, s3 = s2 * s;
      var x = 0.5 * ((2 * p1x) + (-p0x + p2x) * s +
        (2 * p0x - 5 * p1x + 4 * p2x - p3x) * s2 +
        (-p0x + 3 * p1x - 3 * p2x + p3x) * s3);
      var y = 0.5 * ((2 * p1y) + (-p0y + p2y) * s +
        (2 * p0y - 5 * p1y + 4 * p2y - p3y) * s2 +
        (-p0y + 3 * p1y - 3 * p2y + p3y) * s3);
      result.push(x, y);
    }
  }
  result.push(points[(n - 1) * 2], points[(n - 1) * 2 + 1]);
  return result;
}

function resamplePolyline(points, targetCount) {
  if (points.length < 4 || targetCount < 2) return points.slice();
  var segLengths = [];
  var totalLen = 0;
  for (var i = 0; i < points.length - 2; i += 2) {
    var dx = points[i + 2] - points[i];
    var dy = points[i + 3] - points[i + 1];
    var len = Math.sqrt(dx * dx + dy * dy);
    segLengths.push(len);
    totalLen += len;
  }
  if (totalLen === 0) return points.slice();
  var step = totalLen / (targetCount - 1);
  var result = [points[0], points[1]];
  var acc = 0;
  var segIdx = 0;
  for (var j = 1; j < targetCount - 1; j++) {
    var target = j * step;
    while (segIdx < segLengths.length && acc + segLengths[segIdx] < target) {
      acc += segLengths[segIdx];
      segIdx++;
    }
    if (segIdx >= segLengths.length) break;
    var remain = target - acc;
    var t = segLengths[segIdx] === 0 ? 0 : remain / segLengths[segIdx];
    var baseX = points[segIdx * 2], baseY = points[segIdx * 2 + 1];
    var nextX = points[segIdx * 2 + 2], nextY = points[segIdx * 2 + 3];
    result.push(baseX + (nextX - baseX) * t, baseY + (nextY - baseY) * t);
  }
  result.push(points[points.length - 2], points[points.length - 1]);
  return result;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
var passed = 0, failed = 0;

function assert(cond, msg) {
  if (cond) { passed++; console.log('  ✓ ' + msg); }
  else { failed++; console.error('  ✗ ' + msg); }
}

console.log('Test: catmullRomSmooth');
var smoothed = catmullRomSmooth([0, 0, 10, 10, 20, 5, 30, 15], 4);
assert(Array.isArray(smoothed), 'returns array');
assert(smoothed.length > 8, 'produces more points than input (' + smoothed.length + ')');
assert(smoothed.length % 2 === 0, 'even length (x,y pairs)');

console.log('\nTest: catmullRomSmooth with < 4 points (2 points = 4 values)');
var short = catmullRomSmooth([0, 0, 10, 10], 4);
// 4 values = 2 points, which is NOT < 4, so it proceeds but produces minimal output
assert(Array.isArray(short), 'returns array for minimal input');
assert(short.length >= 4, 'produces at least 4 values (' + short.length + ')');

console.log('\nTest: resamplePolyline');
var resampled = resamplePolyline([0, 0, 10, 0, 20, 0, 30, 0], 5);
assert(Array.isArray(resampled), 'returns array');
assert(resampled.length === 10, 'produces targetCount*2 values (' + resampled.length + ')');
// First point should be (0,0)
assert(resampled[0] === 0 && resampled[1] === 0, 'first point preserved');
// Last point should be (30,0)
assert(resampled[8] === 30 && resampled[9] === 0, 'last point preserved');

console.log('\nTest: resamplePolyline with degenerate input');
var degenerate = resamplePolyline([0, 0, 0, 0], 5);
assert(degenerate.length === 4, 'degenerate input returns input');

console.log('\n========================================');
console.log('Results: ' + passed + ' passed, ' + failed + ' failed');
console.log('========================================');
process.exit(failed > 0 ? 1 : 0);

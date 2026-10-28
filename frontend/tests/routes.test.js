import assert from "node:assert/strict";
import test from "node:test";
import { APP_ROUTES, getRouteTitle, resolveRoute } from "../src/lib/routes.js";

test("exposes all six application routes", () => {
  assert.deepEqual(
    APP_ROUTES.map(({ path }) => path),
    ["/overview", "/predict", "/analytics", "/history", "/sensors", "/status"],
  );
});

test("resolves direct paths and page titles", () => {
  for (const route of APP_ROUTES) {
    assert.equal(resolveRoute(route.path), route.path);
    assert.equal(getRouteTitle(route.path), route.title);
  }
});

test("keeps the previous dashboard hash links working", () => {
  assert.equal(resolveRoute("/", "#overview"), "/overview");
  assert.equal(resolveRoute("/", "#inference"), "/predict");
  assert.equal(resolveRoute("/", "#sensor-data"), "/sensors");
  assert.equal(resolveRoute("/", "#evaluation"), "/analytics");
  assert.equal(resolveRoute("/", "#training"), "/analytics");
});

test("routes unknown paths to the overview", () => {
  assert.equal(resolveRoute("/missing"), "/overview");
});

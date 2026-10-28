export const APP_ROUTES = [
  {
    path: "/overview",
    label: "Overview",
    title: "Activity Classification",
    icon: "overview",
  },
  {
    path: "/predict",
    label: "Live Prediction",
    title: "Live Prediction",
    icon: "predict",
  },
  {
    path: "/analytics",
    label: "Model Analytics",
    title: "Model Analytics",
    icon: "analytics",
  },
  {
    path: "/history",
    label: "Prediction History",
    title: "Prediction History",
    icon: "history",
  },
  {
    path: "/sensors",
    label: "Sensor Explorer",
    title: "Sensor Explorer",
    icon: "sensors",
  },
  {
    path: "/status",
    label: "System Status",
    title: "System Status",
    icon: "status",
  },
];

const LEGACY_HASH_ROUTES = {
  "#overview": "/overview",
  "#inference": "/predict",
  "#sensor-data": "/sensors",
  "#evaluation": "/analytics",
  "#training": "/analytics",
  "#history": "/history",
};

const ROUTE_PATHS = new Set(APP_ROUTES.map(({ path }) => path));

export function resolveRoute(pathname, hash = "") {
  if (ROUTE_PATHS.has(pathname)) {
    return pathname;
  }

  if (pathname === "/" && LEGACY_HASH_ROUTES[hash]) {
    return LEGACY_HASH_ROUTES[hash];
  }

  return "/overview";
}

export function getRouteTitle(path) {
  return APP_ROUTES.find((route) => route.path === path)?.title ?? "Overview";
}

// load.mjs — read vehicle.json and track files from disk for Node.js tests and the parity harness.
// Global context: the browser fetches the same files over HTTP; Node reads them from web/assets.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..'); // repository root
export const loadJson = (p) => JSON.parse(readFileSync(p, 'utf8'));
export const loadVehicle = () => loadJson(join(ROOT, 'web', 'assets', 'vehicle.json'));
export const loadTrack = (name) => loadJson(join(ROOT, 'web', 'assets', 'tracks', `${name}.json`));

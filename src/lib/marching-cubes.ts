import * as THREE from 'three';
import * as mc from 'marching-cubes-fast';

const marchingCubes = (mc as any).marchingCubes || (mc as any).default?.marchingCubes;

export function generateMesh(
  sdfValues: Float32Array,
  res: number,
  boxMin: number,
  boxMax: number,
  level = 0.0
): THREE.BufferGeometry {
  if (typeof marchingCubes !== 'function') {
    console.error('marchingCubes is not a function. Module content:', mc);
    return new THREE.BufferGeometry();
  }

  try {
    const bounds = [[boxMin, boxMin, boxMin], [boxMax, boxMax, boxMax]];
    const inv = (res - 1) / (boxMax - boxMin);
    const maxIdx = res - 1;
    const flatIdx = (i: number, j: number, k: number) => i * res * res + j * res + k;

    const sample = (x: number, y: number, z: number) => {
      const gx = Math.max(0, Math.min(maxIdx, (x - boxMin) * inv));
      const gy = Math.max(0, Math.min(maxIdx, (y - boxMin) * inv));
      const gz = Math.max(0, Math.min(maxIdx, (z - boxMin) * inv));

      const x0 = Math.floor(gx);
      const y0 = Math.floor(gy);
      const z0 = Math.floor(gz);
      const x1 = Math.min(maxIdx, x0 + 1);
      const y1 = Math.min(maxIdx, y0 + 1);
      const z1 = Math.min(maxIdx, z0 + 1);

      const tx = gx - x0;
      const ty = gy - y0;
      const tz = gz - z0;

      const c000 = sdfValues[flatIdx(x0, y0, z0)];
      const c100 = sdfValues[flatIdx(x1, y0, z0)];
      const c010 = sdfValues[flatIdx(x0, y1, z0)];
      const c110 = sdfValues[flatIdx(x1, y1, z0)];
      const c001 = sdfValues[flatIdx(x0, y0, z1)];
      const c101 = sdfValues[flatIdx(x1, y0, z1)];
      const c011 = sdfValues[flatIdx(x0, y1, z1)];
      const c111 = sdfValues[flatIdx(x1, y1, z1)];

      const c00 = c000 * (1 - tx) + c100 * tx;
      const c10 = c010 * (1 - tx) + c110 * tx;
      const c01 = c001 * (1 - tx) + c101 * tx;
      const c11 = c011 * (1 - tx) + c111 * tx;

      const c0 = c00 * (1 - ty) + c10 * ty;
      const c1 = c01 * (1 - ty) + c11 * ty;
      const v = c0 * (1 - tz) + c1 * tz;
      return v - level;
    };

    const mesh = marchingCubes(res, sample, bounds, false);

    if (!mesh?.positions || !mesh?.cells || mesh.positions.length === 0 || mesh.cells.length === 0) {
      return new THREE.BufferGeometry();
    }

    // Expand indexed triangles to non-indexed geometry.
    // This avoids renderer/index-type incompatibilities that can create torn meshes.
    const triPositions = new Float32Array(mesh.cells.length * 9);
    let out = 0;
    for (let i = 0; i < mesh.cells.length; i++) {
      const c = mesh.cells[i];
      for (let t = 0; t < 3; t++) {
        const vi = c[t];
        const p = mesh.positions[vi];
        triPositions[out++] = p[0];
        triPositions[out++] = p[1];
        triPositions[out++] = p[2];
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(triPositions, 3));
    geometry.computeVertexNormals();
    return geometry;
  } catch (err) {
    console.error('Error in marchingCubes:', err);
    return new THREE.BufferGeometry();
  }
}

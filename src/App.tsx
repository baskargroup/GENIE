import React, { useState, useMemo, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { OBJExporter } from 'three/examples/jsm/exporters/OBJExporter.js';
import { generateMesh } from './lib/marching-cubes';
import {
  Settings2,
  RefreshCw,
  Cpu,
  Layers,
  ChevronRight,
  ChevronDown,
  Download,
  Activity,
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

function withBaseUrl(path: string) {
  const base = import.meta.env.BASE_URL;
  return `${base}${path.replace(/^\/+/, '')}`;
}

type ModelSpec = {
  id: string;
  name: string;
  desc: string;
  color: string;
  imagePath?: string;
  precomputedMetaPath: string;
};

type PrecomputedMeta = {
  source_head: number;
  target_head: number;
  k: number;
  grid_res: number;
  box_min: number;
  box_max: number;
  n_vox: number;
  binary: string;
  cs: number[];
  ct: number[];
  bs: number;
  bt: number;
  expl_source_topk: number;
  expl_target_topk: number;
};

const MODEL_SPECS: ModelSpec[] = [
  {
    id: 'dfaust-multi55',
    name: 'DFAUST Multi-Head (0 -> 50)',
    desc: '55-head DFAUST INR loaded from your shared checkpoint.',
    color: 'from-violet-500 to-indigo-600',
    precomputedMetaPath: 'precomputed/dfaust_multi55_0_50/meta.json',
  },
  {
    id: 'human-dfaust',
    name: 'Human DFAUST (0 -> 50)',
    desc: 'Two-head INR from human sequence, using head 0 to head 1 interpolation.',
    color: 'from-indigo-500 to-blue-600',
    imagePath: 'content_teaser_dfaust.jpg',
    precomputedMetaPath: 'precomputed/human_00000_00050/meta.json',
  },
  {
    id: 'bunny-ear',
    name: 'Bunny Ear Movement',
    desc: 'Two-head INR ear movement model, using head 0 to head 1 interpolation.',
    color: 'from-emerald-500 to-teal-600',
    imagePath: 'bunny.jpg',
    precomputedMetaPath: 'precomputed/bunny_ear/meta.json',
  },
];

export default function App() {
  const [view, setView] = useState<'landing' | 'editor'>('landing');
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const [k] = useState(20);
  const [gridRes] = useState(64);
  const [isComputing, setIsComputing] = useState(false);

  const [t, setT] = useState(0);
  const [db, setDb] = useState(0);
  const [d, setD] = useState<number[]>(new Array(20).fill(0));
  const [showBasis, setShowBasis] = useState(false);

  const [modelData, setModelData] = useState<{
    modelName: string;
    sourceHead: number;
    targetHead: number;
    kUsed: number;
    cs: number[];
    ct: number[];
    bs: number;
    bt: number;
    explS: number;
    explT: number;
    gridRes: number;
    boxMin: number;
    boxMax: number;
    nVox: number;
    baseVolume: Float32Array;
    modeVolumes: Float32Array[];
  } | null>(null);

  const selectedSpec = useMemo(
    () => MODEL_SPECS.find((m) => m.id === selectedModel) ?? null,
    [selectedModel]
  );

  const canvasRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    mesh: THREE.Mesh | null;
  } | null>(null);

  const initializeModel = async () => {
    if (!selectedSpec) {
      return;
    }
    setIsComputing(true);
    try {
      const metaResp = await fetch(withBaseUrl(selectedSpec.precomputedMetaPath));
      if (!metaResp.ok) {
        throw new Error(`Failed to load precomputed meta: ${selectedSpec.precomputedMetaPath}`);
      }
      const meta = (await metaResp.json()) as PrecomputedMeta;

      const metaDir = selectedSpec.precomputedMetaPath.slice(
        0,
        selectedSpec.precomputedMetaPath.lastIndexOf('/') + 1
      );
      const binPath = `${metaDir}${meta.binary}`;
      const binResp = await fetch(binPath);
      if (!binResp.ok) {
        throw new Error(`Failed to load precomputed binary: ${binPath}`);
      }
      const all = new Float32Array(await binResp.arrayBuffer());

      const nVox = Number(meta.n_vox);
      const kAvail = Number(meta.k);
      const required = (kAvail + 1) * nVox;
      if (all.length < required) {
        throw new Error(`Precomputed cache too small: got ${all.length}, need ${required}`);
      }

      const kUsed = Math.min(k, kAvail);
      const baseVolume = all.subarray(0, nVox);
      const modeVolumes = Array.from(
        { length: kUsed },
        (_, i) => all.subarray((i + 1) * nVox, (i + 2) * nVox)
      );
      const cs = meta.cs.slice(0, kUsed);
      const ct = meta.ct.slice(0, kUsed);

      setModelData({
        modelName: selectedSpec.name,
        sourceHead: Number(meta.source_head),
        targetHead: Number(meta.target_head),
        kUsed,
        cs,
        ct,
        bs: Number(meta.bs),
        bt: Number(meta.bt),
        explS: Number(meta.expl_source_topk),
        explT: Number(meta.expl_target_topk),
        gridRes: Number(meta.grid_res),
        boxMin: Number(meta.box_min),
        boxMax: Number(meta.box_max),
        nVox,
        baseVolume,
        modeVolumes,
      });
      setD(new Array(kUsed).fill(0));
      setT(0);
      setDb(0);
    } catch (err) {
      console.error('Initialization failed:', err);
    } finally {
      setIsComputing(false);
    }
  };

  useEffect(() => {
    if (view === 'editor' && selectedSpec) {
      initializeModel();
    }
  }, [view, selectedSpec]);

  useEffect(() => {
    if (view !== 'editor' || !canvasRef.current) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fafc);

    const camera = new THREE.PerspectiveCamera(
      45,
      canvasRef.current.clientWidth / canvasRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(2.5, 2.5, 2.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0xf8fafc, 1.0);
    canvasRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const pointLight1 = new THREE.PointLight(0xffffff, 0.9);
    pointLight1.position.set(5, 5, 5);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0xdbeafe, 0.7);
    pointLight2.position.set(-5, -5, 5);
    scene.add(pointLight2);

    const grid = new THREE.GridHelper(2, 20, 0xcbd5e1, 0xe2e8f0);
    grid.position.y = -1;
    scene.add(grid);

    sceneRef.current = { scene, camera, renderer, controls, mesh: null };

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!canvasRef.current) return;
      camera.aspect = canvasRef.current.clientWidth / canvasRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      if (canvasRef.current) {
        canvasRef.current.removeChild(renderer.domElement);
      }
    };
  }, [view]);

  useEffect(() => {
    if (view !== 'editor' || !modelData || !sceneRef.current) return;

    const { cs, ct, bs, bt, kUsed, nVox, baseVolume, modeVolumes, gridRes, boxMin, boxMax } = modelData;
    const { scene } = sceneRef.current;

    const c = Array.from({ length: kUsed }, (_, i) => (1 - t) * cs[i] + t * ct[i] + (d[i] ?? 0));
    const b = (1 - t) * bs + t * bt + db;

    const sdfValues = new Float32Array(nVox);
    for (let i = 0; i < nVox; i++) {
      sdfValues[i] = baseVolume[i] + b;
    }
    for (let m = 0; m < kUsed; m++) {
      const coeff = c[m];
      if (Math.abs(coeff) < 1e-8) continue;
      const mode = modeVolumes[m];
      for (let i = 0; i < nVox; i++) {
        sdfValues[i] += coeff * mode[i];
      }
    }

    const geometry = generateMesh(sdfValues, gridRes, boxMin, boxMax, 0);

    if (sceneRef.current.mesh) {
      scene.remove(sceneRef.current.mesh);
      sceneRef.current.mesh.geometry.dispose();
    }

    const material = new THREE.MeshStandardMaterial({
      color: 0x4c78a8,
      roughness: 0.3,
      metalness: 0.2,
      side: THREE.DoubleSide,
    });

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
    sceneRef.current.mesh = mesh;
  }, [t, db, d, modelData, view]);

  const handleDownload = () => {
    if (!sceneRef.current?.mesh) return;
    const exporter = new OBJExporter();
    const result = exporter.parse(sceneRef.current.mesh);
    const blob = new Blob([result], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'siren_mesh.obj';
    link.click();
    URL.revokeObjectURL(url);
  };

  const activeK = modelData?.kUsed ?? k;
  const activeGridRes = modelData?.gridRes ?? gridRes;
  const demoModels = MODEL_SPECS.filter(
    (model) => model.id === 'human-dfaust' || model.id === 'bunny-ear'
  );

  if (view === 'landing') {
    return (
      <div className="min-h-screen bg-[#f8fafc] text-zinc-900 flex flex-col items-center justify-center p-6 relative overflow-hidden">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-200/60 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-sky-200/60 blur-[120px] rounded-full" />

        <div className="z-10 text-center space-y-8 max-w-4xl">
          <div className="space-y-4">
            <h1 className="text-8xl font-black tracking-tighter italic uppercase text-transparent bg-clip-text bg-gradient-to-b from-zinc-900 to-zinc-500">
              GENIE
            </h1>
            <div className="space-y-2 max-w-3xl mx-auto">
              <p className="text-zinc-700 text-lg md:text-xl font-semibold leading-snug">
                Gram-Eigenmode INR Editing with Closed-Form Geometry Updates
              </p>
              <p className="text-zinc-500 text-sm md:text-base">
                Samundra Karki, Adarsh Krishnamurthy, Baskar Ganapathysubramanian
              </p>
            </div>
            <div className="pt-2">
              <a
                href="https://arxiv.org/abs/2603.29860"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center rounded-full border border-zinc-300 bg-white/90 px-5 py-2.5 text-sm font-semibold text-zinc-800 shadow-sm transition-colors hover:border-indigo-300 hover:text-indigo-700 hover:bg-white"
              >
                Link to Paper
              </a>
            </div>
          </div>

          <div className="pt-6">
            <div className="mb-6 flex items-center justify-center gap-8 text-zinc-500 text-[10px] font-bold uppercase tracking-[0.2em]">
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                Real-time Editing
              </div>
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-pink-500" />
                500x Faster
              </div>
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                Structural Editing
              </div>
            </div>
            <div className="mx-auto max-w-3xl rounded-[2rem] border border-zinc-200 bg-white/85 p-6 md:p-8 shadow-sm backdrop-blur-sm">
              <div className="mb-5 text-left">
                <div className="inline-flex items-center rounded-full border border-zinc-300 bg-zinc-50 px-3 py-1 text-[11px] font-bold uppercase tracking-[0.24em] text-zinc-500">
                  Demo
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {demoModels.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => {
                      setSelectedModel(model.id);
                      setView('editor');
                    }}
                    className="group relative p-8 bg-white border border-zinc-200 rounded-3xl text-left transition-all hover:bg-white hover:border-zinc-300 hover:scale-[1.02] active:scale-[0.98] shadow-sm"
                  >
                    <div className="mb-6 overflow-hidden rounded-2xl border border-zinc-200 bg-zinc-50 shadow-sm">
                      {model.imagePath ? (
                        <img
                  src={withBaseUrl(model.imagePath)}
                          alt={model.name}
                          className="h-32 w-full object-cover transition-transform duration-300 group-hover:scale-[1.03]"
                        />
                      ) : (
                        <div className={cn('h-32 w-full bg-gradient-to-br', model.color)} />
                      )}
                    </div>
                    <h3 className="text-xl font-bold group-hover:text-indigo-600 transition-colors">{model.name}</h3>
                    <div className="absolute bottom-8 right-8 opacity-0 group-hover:opacity-100 transition-opacity">
                      <ChevronRight className="w-5 h-5 text-indigo-600" />
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="pt-12">
            <figure className="mx-auto max-w-3xl overflow-hidden rounded-[2rem] border border-zinc-200 bg-white/90 p-3 shadow-sm backdrop-blur-sm">
              <div className="overflow-hidden rounded-[1.4rem] bg-zinc-50">
                <img
                  src={withBaseUrl('RepresentativeImage.jpeg')}
                  alt="Representative GENIE deformation results and editing directions"
                  className="block h-auto w-full"
                />
              </div>
            </figure>
          </div>

          <div className="pt-12">
            <div className="mx-auto max-w-3xl rounded-3xl border border-zinc-200 bg-white/80 p-6 text-left shadow-sm backdrop-blur-sm">
              <div className="mb-3 text-[11px] font-bold uppercase tracking-[0.24em] text-zinc-500">
                Abstract
              </div>
              <p className="text-sm leading-7 text-zinc-700 md:text-base">
                Implicit Neural Representations (INRs) provide compact models of geometry, but it is unclear when
                their learned shapes can be edited without retraining. We show that the Gram operator induced by the
                INR&apos;s penultimate features admits deformation eigenmodes that parameterize a family of realizable
                edits of the SDF zero level set. A key finding is that these modes are not intrinsic to the geometry
                alone: they are reliably recoverable only when the Gram operator is estimated from sufficiently rich
                sampling distributions. We derive a single closed-form update that performs geometric edits to the INR
                without optimization by leveraging the deformation modes. We characterize theoretically the precise
                set of deformations that are feasible under this one-shot update, and show that editing is well-posed
                exactly within the span of these deformation modes.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-[#f8fafc] text-zinc-800 font-sans overflow-hidden selection:bg-indigo-200">
      <div className="w-80 border-r border-zinc-200 flex flex-col bg-white shadow-sm z-10">
        <div className="p-6 border-b border-zinc-200">
          <button
            onClick={() => setView('landing')}
            className="flex items-center gap-3 mb-6 group hover:text-zinc-900 transition-colors"
          >
            <div className="p-2 bg-indigo-100 rounded-lg group-hover:bg-indigo-200 transition-colors">
              <Activity className="w-5 h-5 text-indigo-600" />
            </div>
            <h1 className="text-lg font-semibold tracking-tight">GENIE Editor</h1>
          </button>
          <div className="flex items-center justify-between">
            <div className="px-2 py-1 bg-indigo-100 rounded text-[10px] font-bold text-indigo-700 uppercase tracking-wider">
              {modelData?.modelName ?? selectedSpec?.name ?? selectedModel}
            </div>
            <div className="text-[10px] text-zinc-500 font-mono">v1.0.5</div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-8 scrollbar-thin scrollbar-thumb-zinc-300">
          <section className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-bold uppercase tracking-widest text-indigo-400 flex items-center gap-2">
                <Settings2 className="w-3 h-3" />
                Global Parameters
              </h2>
            </div>

            <div className="space-y-5">
              <div className="space-y-2">
                <div className="flex justify-between text-[11px]">
                  <span className="text-zinc-600">Interpolation (t)</span>
                  <span className="text-indigo-600 font-mono font-medium">{t.toFixed(3)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.001"
                  value={t}
                  onChange={(e) => setT(parseFloat(e.target.value))}
                  className="w-full accent-indigo-600 bg-zinc-200 h-1 rounded-full appearance-none cursor-pointer hover:bg-zinc-300 transition-colors"
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-[11px]">
                  <span className="text-zinc-600">Bias Offset (db)</span>
                  <span className="text-indigo-600 font-mono font-medium">{db.toFixed(3)}</span>
                </div>
                <input
                  type="range"
                  min="-0.5"
                  max="0.5"
                  step="0.001"
                  value={db}
                  onChange={(e) => setDb(parseFloat(e.target.value))}
                  className="w-full accent-indigo-600 bg-zinc-200 h-1 rounded-full appearance-none cursor-pointer hover:bg-zinc-300 transition-colors"
                />
              </div>
            </div>
          </section>

          <section className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-bold uppercase tracking-widest text-indigo-400 flex items-center gap-2">
                <Layers className="w-3 h-3" />
                Eigen Coefficients (d)
              </h2>
              <button
                onClick={() => setD(new Array(activeK).fill(0))}
                className="text-[10px] text-zinc-500 hover:text-zinc-900 transition-colors flex items-center gap-1"
              >
                <RefreshCw className="w-2.5 h-2.5" />
                Reset
              </button>
            </div>

            <div className="space-y-4">
              {(showBasis ? d : d.slice(0, 8)).map((val, i) => (
                <div key={i} className="space-y-1.5 group">
                  <div className="flex justify-between text-[10px]">
                    <span className="text-zinc-500 font-mono group-hover:text-zinc-700 transition-colors">λ_{i + 1}</span>
                    <span className="text-indigo-700 font-mono">{val.toFixed(3)}</span>
                  </div>
                  <input
                    type="range"
                    min="-1"
                    max="1"
                    step="0.01"
                    value={val}
                    onChange={(e) => {
                      const newD = [...d];
                      newD[i] = parseFloat(e.target.value);
                      setD(newD);
                    }}
                    className="w-full accent-indigo-600 bg-zinc-200 h-1 rounded-full appearance-none cursor-pointer hover:bg-zinc-300 transition-colors"
                  />
                </div>
              ))}
              <div className="pt-2 text-center">
                <button
                  onClick={() => setShowBasis(!showBasis)}
                  className="text-[10px] text-zinc-500 hover:text-zinc-700 transition-colors flex items-center gap-1 mx-auto"
                >
                  {showBasis ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                  {showBasis ? 'Show less' : `Show all ${activeK} coefficients`}
                </button>
              </div>
            </div>
          </section>
        </div>

        <div className="p-6 border-t border-zinc-200 bg-zinc-50 space-y-3">
          <button
            onClick={initializeModel}
            disabled={isComputing || !selectedSpec}
            className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-semibold rounded-lg flex items-center justify-center gap-2 transition-all shadow-lg shadow-indigo-500/10 active:scale-[0.98]"
          >
            {isComputing ? <RefreshCw className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
            Reload Cache
          </button>
          <button
            onClick={handleDownload}
            className="w-full py-2.5 bg-zinc-200 hover:bg-zinc-300 text-zinc-800 text-xs font-semibold rounded-lg flex items-center justify-center gap-2 transition-all active:scale-[0.98]"
          >
            <Download className="w-3 h-3" />
            Export Mesh (.obj)
          </button>
        </div>
      </div>

      <div className="flex-1 relative flex flex-col">
        <div className="absolute top-0 left-0 right-0 p-6 flex justify-between items-start pointer-events-none z-20">
          <div className="bg-white/85 backdrop-blur-xl border border-zinc-200 p-4 rounded-2xl pointer-events-auto shadow-sm">
            <div className="flex items-center gap-6">
              <div className="space-y-1">
                <div className="text-[10px] uppercase tracking-wider text-zinc-500 font-bold">Source Expl.</div>
                <div className="text-lg font-mono text-zinc-900">{(modelData?.explS || 0).toFixed(4)}</div>
              </div>
              <div className="w-px h-8 bg-zinc-200" />
              <div className="space-y-1">
                <div className="text-[10px] uppercase tracking-wider text-zinc-500 font-bold">Target Expl.</div>
                <div className="text-lg font-mono text-zinc-900">{(modelData?.explT || 0).toFixed(4)}</div>
              </div>
              <div className="w-px h-8 bg-zinc-200" />
              <div className="space-y-1">
                <div className="text-[10px] uppercase tracking-wider text-zinc-500 font-bold">Resolution</div>
                <div className="text-lg font-mono text-zinc-900">{activeGridRes}³</div>
              </div>
            </div>
          </div>

          <div className="flex gap-2 pointer-events-auto">
            <div className="bg-white/85 backdrop-blur-xl border border-zinc-200 p-2 rounded-xl flex items-center gap-2">
              <div className="px-2 py-1 bg-zinc-100 rounded text-[10px] font-mono text-zinc-600">FPS: 60</div>
              <div className="px-2 py-1 bg-zinc-100 rounded text-[10px] font-mono text-zinc-600 uppercase">WebGL 2.0</div>
            </div>
          </div>
        </div>

        <div ref={canvasRef} className="flex-1 w-full h-full" />

        <div className="absolute bottom-6 left-6 right-6 pointer-events-none z-20 flex justify-center">
          <div className="bg-white/85 backdrop-blur-xl border border-zinc-200 p-3 rounded-2xl inline-flex items-center gap-4 pointer-events-auto shadow-sm">
            <div className="flex items-center gap-2 text-[10px] text-zinc-600 font-medium">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
              Real-time Inference
            </div>
            <div className="w-px h-3 bg-zinc-200" />
            <div className="text-[10px] text-zinc-600 font-mono tracking-tight">
              w(t) = w_perp + V_k[(1-t)c_s + t c_t + d]
            </div>
          </div>
        </div>
      </div>

      {isComputing && (
        <div className="absolute inset-0 bg-white/75 backdrop-blur-md z-50 flex flex-col items-center justify-center gap-6">
          <div className="relative">
            <div className="w-20 h-20 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin" />
            <Cpu className="w-8 h-8 text-indigo-600 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
          </div>
          <div className="text-center space-y-3">
            <h3 className="text-zinc-900 text-xl font-semibold tracking-tight">Initializing SIREN Model</h3>
            <p className="text-zinc-600 text-sm max-w-xs leading-relaxed">
              Loading precomputed mode cache for fast interactive editing...
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

import * as mc from 'marching-cubes-fast';
const mod:any = (mc as any);
const fn = mod.marchingCubes || mod.default?.marchingCubes || mod.default;
console.log('types', typeof mod, typeof mod.default, typeof mod.marchingCubes, typeof mod.default?.marchingCubes, typeof fn);
const res=16;
const sdf = new Float32Array(res*res*res);
for(let x=0;x<res;x++)for(let y=0;y<res;y++)for(let z=0;z<res;z++){
  const ix=x*res*res+y*res+z;
  const px = -1 + 2*x/(res-1); const py=-1+2*y/(res-1); const pz=-1+2*z/(res-1);
  sdf[ix] = Math.sqrt(px*px+py*py+pz*pz)-0.6;
}
const mesh = fn(sdf,[res,res,res],{isoLevel:0,bounds:[[-1,-1,-1],[1,1,1]]});
console.log(Object.keys(mesh));
console.log('pos len', mesh.positions?.length, 'cells', mesh.cells?.length);

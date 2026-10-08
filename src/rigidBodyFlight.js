// Preliminary 6-DoF rigid-body disc / annulus flight: position, velocity,
// attitude quaternion, and body-frame angular velocity. Forces and moments
// are coefficient-model approximations, NOT calibrated FRC game-piece models.
const DEG = Math.PI / 180;
const EPS = 1e-10;
const dot = (a,b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm = (a) => Math.hypot(...a);
const normalize = (a) => {const n=norm(a);return n<EPS?[0,0,1]:a.map(x=>x/n);};
export const quaternionNormalize = (q) => {
  const n = Math.hypot(...q);
  return n < EPS ? [1,0,0,0] : q.map(x => x/n);
};
export function quaternionMultiply(a,b) {
  const [w,x,y,z]=a, [W,X,Y,Z]=b;
  return [w*W-x*X-y*Y-z*Z, w*X+x*W+y*Z-z*Y,
    w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W];
}
export function rotateByQuaternion(q,v) {
  const [w,x,y,z]=quaternionNormalize(q);
  const t=cross([x,y,z],v).map(x=>2*x);
  const u=cross([x,y,z],t);
  return v.map((value,i)=>value+w*t[i]+u[i]);
}
function inverseRotate(q,v) {
  return rotateByQuaternion([q[0],-q[1],-q[2],-q[3]],v);
}
export function initialAttitude(pitchDeg=0,rollDeg=0) {
  const a=-pitchDeg*DEG/2, b=rollDeg*DEG/2;
  return quaternionNormalize(quaternionMultiply(
    [Math.cos(a),0,Math.sin(a),0],
    [Math.cos(b),Math.sin(b),0,0],
  ));
}
export function discPrincipalInertia(piece) {
  const R=piece.diameter/2, r=piece.shape==='ring'?(piece.innerDiameter??R)/2:0;
  const t=piece.thickness??piece.diameter*0.1, m=piece.mass;
  const Iz=0.5*m*(R*R+r*r);
  const Ixy=0.25*m*(R*R+r*r)+m*t*t/12;
  return [Ixy,Ixy,Iz];
}

function physicalModel(piece,params) {
  if (!piece || !['disc','ring'].includes(piece.shape)) throw new RangeError('Requires a disc or ring profile');
  const fields=[piece.mass,piece.diameter,piece.dragCoeff,piece.liftCoeff,
    piece.thickness??0.02,piece.clAlpha??1.5,piece.cdAlpha??1,piece.cmAlpha??0.05,
    piece.angularDamping??0.01,params.gravity,params.airDensity];
  if (!fields.every(Number.isFinite) || fields[0]<=0 || fields[1]<=0) throw new RangeError('Invalid rigid-body parameters');
  return {
    piece, gravity: params.gravity, rho: params.airDensity, wind: params.wind??[0,0,0],
    enableDrag:params.enableDrag!==false, enableLift:params.enableMagnus!==false,
    mass:piece.mass, diameter:piece.diameter, area:Math.PI*(piece.diameter/2)**2,
    inertia:discPrincipalInertia(piece),
  };
}

function derivative(y,p) {
  const velocity=y.slice(3,6), q=quaternionNormalize(y.slice(6,10)), omega=y.slice(10,13);
  const worldNormal=rotateByQuaternion(q,[0,0,1]);
  const rel=velocity.map((v,i)=>v-p.wind[i]);
  const speed=norm(rel), heading=speed<EPS?[1,0,0]:rel.map(v=>v/speed);
  const sinAlpha=Math.max(-1,Math.min(1,-dot(heading,worldNormal)));
  const alpha=Math.asin(sinAlpha);
  const cd=Math.max(0,p.piece.dragCoeff+(p.piece.cdAlpha??1)*alpha*alpha);
  const cl=Math.max(-3,Math.min(3,p.piece.liftCoeff+(p.piece.clAlpha??1.5)*alpha));
  const dynamic=0.5*p.rho*p.area*speed*speed;
  const accel=[0,0,-p.gravity];
  if (speed>EPS) {
    if(p.enableDrag) for(let i=0;i<3;i++) accel[i]-=dynamic*cd*heading[i]/p.mass;
    const perp=worldNormal.map((v,i)=>v-dot(worldNormal,heading)*heading[i]);
    const mag=norm(perp);
    if(p.enableLift&&mag>EPS) for(let i=0;i<3;i++) accel[i]+=dynamic*cl*perp[i]/(mag*p.mass);
  }
  // Thin disc/annulus volume displaced by air; no foam deformation.
  const radius=p.diameter/2, inner=p.piece.shape==='ring'?(p.piece.innerDiameter??0)/2:0;
  const volume=Math.PI*(radius*radius-inner*inner)*(p.piece.thickness??0.02);
  accel[2]+=p.rho*volume*p.gravity/p.mass;

  const pitchAxis=normalize(cross(heading,worldNormal));
  const cm=p.piece.cmAlpha??0.05;
  const worldTorque=pitchAxis.map(v=>-dynamic*p.diameter/2*cm*alpha*v);
  const bodyTorque=inverseRotate(q,worldTorque);
  const [Ix,Iy,Iz]=p.inertia, [wx,wy,wz]=omega;
  const gyroscopic=[(Iz-Iy)*wy*wz,(Ix-Iz)*wz*wx,(Iy-Ix)*wx*wy];
  const damping=p.piece.angularDamping??0.01;
  const dw=omega.map((v,i)=>(bodyTorque[i]-gyroscopic[i])/p.inertia[i]-damping*v);
  const qdot=quaternionMultiply(q,[0,...omega]).map(v=>v*0.5);
  return [...velocity,...accel,...qdot,...dw];
}

function step(y,p,dt) {
  const k1=derivative(y,p);
  const add=(a,b,m)=>a.map((v,i)=>v+m*b[i]);
  const k2=derivative(add(y,k1,dt/2),p);
  const k3=derivative(add(y,k2,dt/2),p);
  const k4=derivative(add(y,k3,dt),p);
  const result=y.map((v,i)=>v+dt/6*(k1[i]+2*k2[i]+2*k3[i]+k4[i]));
  result.splice(6,4,...quaternionNormalize(result.slice(6,10)));
  if (!result.every(Number.isFinite)) throw new RangeError('Rigid-body integration diverged');
  return result;
}

function sample(time,y) {
  const q=quaternionNormalize(y.slice(6,10));
  const angularVelocity=rotateByQuaternion(q,y.slice(10,13));
  return {time, state:[...y.slice(0,6),...angularVelocity],
    orientation:q, normal:rotateByQuaternion(q,[0,0,1]), angularVelocity};
}

// Initial translational state matches the sphere integrator's 9-vector. The
// axial rotation comes from spinRPM; pitched or rolled disc spins about its normal.
export function integrateRigidBodyFlight(initial, params, options={}) {
  const piece=params.gamePiece, p=physicalModel(piece,params);
  const dt=options.dt??0.001, maxTime=options.maxTime??5;
  if (!Number.isFinite(dt)||dt<=0||!Number.isFinite(maxTime)||maxTime<0||maxTime>30) {
    throw new RangeError('Invalid rigid-body integrator duration or time step');
  }
  if (dt<1e-5 || dt>0.05 || maxTime/dt>30000) throw new RangeError('Rigid-body step must be in [1e-5,0.05] with at most 30000 steps');
  const q=initialAttitude(piece.pitchDeg??0,piece.rollDeg??0);
  const spin=(params.spinRPM??0)*Math.PI/30;
  let y=[...initial.slice(0,6),...q,0,0,spin], time=0;
  const samples=[sample(0,y)];
  while(time<maxTime-1e-12) {
    const h=Math.min(dt,maxTime-time);
    const next=step(y,p,h);
    if(y[2]>0 && next[2]<=0 && next[5]<0) {
      const f=Math.max(0,Math.min(1,y[2]/(y[2]-next[2])));
      const last=y.map((value,i)=>value+f*(next[i]-value));
      last[2]=0;
      samples.push(sample(time+f*h,last));
      break;
    }
    time+=h; y=next;
    samples.push(sample(time,y));
  }
  return samples;
}

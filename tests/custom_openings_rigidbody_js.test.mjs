import test from 'node:test';
import assert from 'node:assert/strict';
import {polygonArea,polygonClearance,validatePolygonVertices} from '../src/polygonGeometry.js';
import {validateTarget, validatePiece, parseLibrary, emptyLibrary, upsertLibraryItem} from '../src/gameCatalog.js';
import {createTargetGeometry, classifyTargetInteraction, targetWireframes} from '../src/scoringTargets.js';
import {integrateRigidBodyFlight, initialAttitude, rotateByQuaternion} from '../src/rigidBodyFlight.js';
import {gamePieceWireframe} from '../src/gamePieceGeometry.js';
import {simulateShot} from '../src/trajectory2d.js';

const square=[[-0.5,-0.4],[0.5,-0.4],[0.5,0.4],[-0.5,0.4]];
const custom={id:'custom-poly',name:'Drawn goal',kind:'polygon',plane:'vertical',
  x:0,lateralY:0,z:2,points:3,vertices:square};
function segment(a,b) {
  const velocity=a.map((value,i)=>b[i]-value);
  return [{time:0,state:[...a,...velocity,0,0,0]},
    {time:1,state:[...b,...velocity,0,0,0]}];
}
test('polygon accepts concave contours; rejects self-intersections and zero area',()=>{
  const concave=[[0,0],[2,0],[2,2],[1,1],[0,2]];
  assert.ok(polygonArea(validatePolygonVertices(concave))>0);
  assert.ok(polygonClearance([0.5,0.5],concave).inside);
  assert.equal(polygonClearance([1,1.5],concave).inside,false);
  assert.throws(()=>validatePolygonVertices([[0,0],[1,1],[0,1],[1,0]]));
  assert.throws(()=>validatePolygonVertices([[0,0],[1,0],[2,0]]));
  assert.throws(()=>validatePolygonVertices([[0,0],[0,0],[1,1]]));
});

test('drawn polygons validate, save and reload with all vertices and points',()=>{
  const target=validateTarget(custom);
  let lib=upsertLibraryItem(emptyLibrary(),'targets',target);
  const restored=parseLibrary({...lib,version:1});
  assert.deepEqual(restored.targets[0].vertices,square);
  assert.equal(restored.targets[0].points,3);
  assert.deepEqual(targetWireframes(createTargetGeometry(restored.targets[0]),0.1).frames[0],
    square.map(([u,v])=>[0,u,2+v]));
});

test('vertical and horizontal drawn goals use their correct crossing direction',()=>{
  const g=createTargetGeometry(custom);
  assert.equal(classifyTargetInteraction(segment([-1,0,2],[1,0,2]),g,0.1).classification,'clean-entry');
  assert.equal(classifyTargetInteraction(segment([-1,0.48,2],[1,0.48,2]),g,0.1).classification,'rim-collision');
  assert.equal(classifyTargetInteraction(segment([-1,1,2],[1,1,2]),g,0.1).classification,'miss');
  const h=createTargetGeometry({...custom,plane:'horizontal'});
  assert.equal(classifyTargetInteraction(segment([0,0,3],[0,0,1]),h,0.1).classification,'clean-entry');
  assert.equal(classifyTargetInteraction(segment([-1,0,2],[1,0,2]),h,0.1).classification,'miss');
});

test('rigid-body model conserves quaternion length and tracks nonzero axial spin',()=>{
  const piece=validatePiece({id:'custom-disc',name:'Disc',shape:'disc',
    mass:0.175,diameter:0.24,thickness:0.025,dragCoeff:0.15,liftCoeff:0.1,
    pitchDeg:10,rollDeg:5,clAlpha:1.5,cdAlpha:1,cmAlpha:0.02,angularDamping:0.05});
  const init=[-2,0,1,9,0,3,0,0,0];
  const samples=integrateRigidBodyFlight(init,{gamePiece:piece,spinRPM:1000,wind:[0,0,0],
    gravity:9.81,airDensity:1.204,enableDrag:true,enableMagnus:true},{dt:0.01,maxTime:1});
  assert.ok(samples.length>10);
  assert.ok(samples.every(s=>s.state.every(Number.isFinite)));
  assert.ok(samples.every(s=>Math.abs(Math.hypot(...s.orientation)-1)<1e-9));
  assert.ok(Math.abs(samples[5].angularVelocity[2])>1);
  assert.notDeepEqual(samples[0].normal,samples.at(-1).normal);
  const q=initialAttitude(0,0);
  assert.deepEqual(rotateByQuaternion(q,[0,0,1]),[0,0,1]);
});

test('game-piece mesh respects real radius and supports ring inner opening',()=>{
  const ring={shape:'ring',diameter:0.36,innerDiameter:0.21,thickness:0.04};
  const mesh=gamePieceWireframe(ring,[1,2,3]);
  assert.equal(mesh.lines.length,4);
  const r=Math.hypot(mesh.lines[0][0][0]-1,mesh.lines[0][0][1]-2);
  assert.ok(Math.abs(r-0.18)<1e-10);
  assert.equal(gamePieceWireframe({...ring,shape:'sphere'},[1,2,3]).lines.length,3);
});

test('disc attitude reduces vertical slot clearance demand compared to a sphere',()=>{
  const target=createTargetGeometry({id:'custom-thin',name:'Thin slot',kind:'slot',
    x:0,lateralY:0,z:2,width:0.6,height:0.08});
  const path=segment([-1,0,2],[1,0,2]).map(sample=>({...sample,normal:[0,0,1]}));
  const disc={shape:'disc',diameter:0.2,thickness:0.02};
  assert.equal(classifyTargetInteraction(path,target,0.1,disc).classification,'clean-entry');
  assert.equal(classifyTargetInteraction(path,target,0.1).classification,'rim-collision');
});

test('simulateShot selects rigid-body solver for saved discs without changing ball solver',()=>{
  const base={launchX:-2,launchY:1,velocity:11,angleDeg:25,azimuthDeg:0,
    spinRPM:600,mass:0.175,radius:0.12,dragCoeff:0.2,liftCoeff:0.1,
    airDensity:1.204,gravity:9.81,enableDrag:true,enableMagnus:true};
  const disc={shape:'disc',mass:0.175,diameter:0.24,thickness:0.03,
    dragCoeff:0.2,liftCoeff:0.1,pitchDeg:10,rollDeg:0,clAlpha:1.2,cdAlpha:1,cmAlpha:0.02,angularDamping:0.05};
  const d=simulateShot({...base,gamePiece:disc},{dt:0.01,maxTime:2});
  const b=simulateShot(base,{dt:0.01,maxTime:2});
  assert.ok(d.samples3d[0].orientation?.length===4);
  assert.equal(b.samples3d[0].orientation,undefined);
  assert.ok(Math.abs(d.range-b.range)>0.01);
});

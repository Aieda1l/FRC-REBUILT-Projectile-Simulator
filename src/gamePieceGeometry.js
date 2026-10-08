import {initialAttitude, rotateByQuaternion} from './rigidBodyFlight.js';

// Physical-size wireframe mesh in world coordinates. Reused by both charts.
export function gamePieceWireframe(piece, position, orientation = null) {
  const q = orientation ?? initialAttitude(piece.pitchDeg??0, piece.rollDeg??0);
  const R = piece.diameter / 2;
  const h = piece.thickness ?? piece.diameter * 0.1;
  const transform = (p) => rotateByQuaternion(q,p).map((x,i) => x+position[i]);
  const circle = (radius, z, n=40) =>
    Array.from({length:n}, (_,i)=> {
      const a=i*2*Math.PI/n;
      return transform([radius*Math.cos(a),radius*Math.sin(a),z]);
    });
  if (piece.shape === 'sphere') {
    const n=40;
    return {
      lines: [Array.from({length:n},(_,i)=>{
        const a=i*2*Math.PI/n;
        return transform([R*Math.cos(a),R*Math.sin(a),0]);
      }),Array.from({length:n},(_,i)=>{
        const a=i*2*Math.PI/n;
        return transform([R*Math.cos(a),0,R*Math.sin(a)]);
      }),Array.from({length:n},(_,i)=>{
        const a=i*2*Math.PI/n;
        return transform([0,R*Math.cos(a),R*Math.sin(a)]);
      })],
      points:[],
    };
  }
  const lines=[circle(R,h/2),circle(R,-h/2)];
  if(piece.shape === 'ring') {
    const inner=(piece.innerDiameter??piece.diameter*0.5)/2;
    if(inner>0) lines.push(circle(inner,h/2),circle(inner,-h/2));
  }
  const points=[
    [transform([R,0,-h/2]),transform([R,0,h/2])],
    [transform([0,R,-h/2]),transform([0,R,h/2])],
    [transform([0,0,0]),transform([R,0,0])], // spin-phase marker
  ];
  return {lines,points};
}

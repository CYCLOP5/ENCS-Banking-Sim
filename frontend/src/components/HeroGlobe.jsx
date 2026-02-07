import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

/**
 * A slowly rotating wireframe icosphere — used as the Landing page hero background.
 */

function WireGlobe() {
  const meshRef = useRef();
  const points = useMemo(() => {
    // Create random connection lines on a sphere surface
    const pts = [];
    const count = 200;
    for (let i = 0; i < count; i++) {
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = 2 * Math.PI * Math.random();
      const r = 3;
      pts.push(
        new THREE.Vector3(
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.sin(phi) * Math.sin(theta),
          r * Math.cos(phi)
        )
      );
    }
    return pts;
  }, []);

  // Build edges between nearby points
  const lineGeometry = useMemo(() => {
    const positions = [];
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        if (points[i].distanceTo(points[j]) < 1.3) {
          positions.push(
            points[i].x, points[i].y, points[i].z,
            points[j].x, points[j].y, points[j].z
          );
        }
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(positions, 3)
    );
    return geo;
  }, [points]);

  const dotGeometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = points.flatMap((p) => [p.x, p.y, p.z]);
    geo.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(positions, 3)
    );
    return geo;
  }, [points]);

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.06;
      meshRef.current.rotation.x += delta * 0.015;
    }
  });

  return (
    <group ref={meshRef}>
      {/* Wireframe connections */}
      <lineSegments geometry={lineGeometry}>
        <lineBasicMaterial
          color="#32e0c4"
          transparent
          opacity={0.12}
          depthWrite={false}
        />
      </lineSegments>

      {/* Dots at each node */}
      <points geometry={dotGeometry}>
        <pointsMaterial
          color="#05d5fa"
          size={0.04}
          transparent
          opacity={0.6}
          sizeAttenuation
        />
      </points>

      {/* Core wireframe sphere */}
      <mesh>
        <icosahedronGeometry args={[2.95, 2]} />
        <meshBasicMaterial
          wireframe
          color="#ff2a6d"
          transparent
          opacity={0.06}
        />
      </mesh>
    </group>
  );
}

export default function HeroGlobe({ className }) {
  return (
    <div className={className}>
      <Canvas
        camera={{ position: [0, 0, 7], fov: 50 }}
        gl={{ alpha: true, antialias: true }}
        style={{ background: "transparent" }}
      >
        <ambientLight intensity={0.15} />
        <WireGlobe />
      </Canvas>
    </div>
  );
}

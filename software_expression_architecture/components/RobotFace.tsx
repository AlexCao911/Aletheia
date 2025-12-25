import React, { useRef, useMemo, useLayoutEffect, useEffect, Suspense } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Float, Environment, ContactShadows, RoundedBox } from '@react-three/drei';
import * as THREE from 'three';
import { FacsState } from '../types';

interface RobotFaceProps {
  state: FacsState;
  videoStream?: MediaStream | null;
}

// --- Reusable Materials ---
const TitaniumMaterial = new THREE.MeshStandardMaterial({
  color: "#2c2c2e",
  roughness: 0.3,
  metalness: 0.8,
});

const SkinMaterial = new THREE.MeshStandardMaterial({
    color: "#3a3a3c",
    roughness: 0.4,
    metalness: 0.6,
});

const BlackGlassMaterial = new THREE.MeshPhysicalMaterial({
  color: "#000000",
  roughness: 0.1,
  metalness: 0.9,
  clearcoat: 1,
  clearcoatRoughness: 0.1,
});

const EyeWhiteMaterial = new THREE.MeshStandardMaterial({
    color: "#E5E5EA",
    roughness: 0.2,
    metalness: 0.1,
});

const LipMaterial = new THREE.MeshStandardMaterial({
    color: "#FF2D55",
    roughness: 0.4,
    metalness: 0.3,
});

// --- Helper Components ---

// Smooth Curved Lip Component
const LipMesh = ({ points, radius }: { points: THREE.Vector3[], radius: number }) => {
    const meshRef = useRef<THREE.Mesh>(null);
    
    const curve = useMemo(() => {
        return new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.5);
    }, [points]);

    return (
        <mesh ref={meshRef}>
            <tubeGeometry args={[curve, 24, radius, 8, false]} />
            <primitive object={LipMaterial} />
        </mesh>
    );
};

// The Eye Assembly (Socket, Ball, Pupil, Lids)
const Eye = ({ 
    position, 
    lookH, 
    lookV, 
    lidTop, 
    lidBot, 
    pupilColor,
    reflectionTexture
}: { 
    position: [number, number, number], 
    lookH: number, 
    lookV: number, 
    lidTop: number, 
    lidBot: number,
    pupilColor: string,
    reflectionTexture?: THREE.VideoTexture | null
}) => {
    const groupRef = useRef<THREE.Group>(null);
    const eyeballRef = useRef<THREE.Group>(null);
    const topLidRef = useRef<THREE.Mesh>(null);
    const botLidRef = useRef<THREE.Mesh>(null);

    useFrame(() => {
        // Look At Rotation
        if (eyeballRef.current) {
            // Map -100..100 to Radians
            const xRot = THREE.MathUtils.degToRad(-lookV * 0.3); // Up/Down
            const yRot = THREE.MathUtils.degToRad(-lookH * 0.3); // Left/Right
            eyeballRef.current.rotation.x = xRot;
            eyeballRef.current.rotation.y = yRot;
        }

        // Lids Rotation
        if (topLidRef.current) {
            const closedAngle = 0;
            const openAngle = -Math.PI / 2.5;
            const t = lidTop / 100;
            topLidRef.current.rotation.x = THREE.MathUtils.lerp(openAngle, closedAngle, t);
        }
        if (botLidRef.current) {
            const closedAngle = 0;
            const openAngle = Math.PI / 2.5;
            const t = lidBot / 100;
            botLidRef.current.rotation.x = THREE.MathUtils.lerp(openAngle, closedAngle, t);
        }
    });

    return (
        <group position={position} ref={groupRef}>
            {/* Socket (Background) */}
            <mesh position={[0, 0, -0.5]}>
                <sphereGeometry args={[1.1, 32, 32]} />
                <primitive object={BlackGlassMaterial} />
            </mesh>

            {/* Eyeball Group (Rotates) */}
            <group ref={eyeballRef}>
                {/* Sclera */}
                <mesh>
                    <sphereGeometry args={[1, 32, 32]} />
                    <primitive object={EyeWhiteMaterial} />
                </mesh>
                
                {/* Pupil (Emissive or Screen) */}
                <mesh position={[0, 0, 0.85]} scale={[0.4, 0.4, 0.2]}>
                    <sphereGeometry args={[1, 32, 16]} />
                    <meshBasicMaterial 
                        color={reflectionTexture ? "#ffffff" : pupilColor} 
                        map={reflectionTexture || null}
                    />
                    <pointLight 
                        distance={3} 
                        intensity={reflectionTexture ? 1 : 2} 
                        color={reflectionTexture ? "#ffffff" : pupilColor} 
                        decay={2} 
                    />
                </mesh>
                
                {/* Cornea (Glass Layer) */}
                <mesh position={[0, 0, 0]} scale={[1.05, 1.05, 1.05]}>
                    <sphereGeometry args={[1, 32, 32]} />
                    <meshPhysicalMaterial 
                        transmission={0.9} 
                        roughness={0} 
                        thickness={0.1}
                        transparent 
                        opacity={0.3}
                    />
                </mesh>
            </group>

            {/* Eyelids (Fixed to head, don't rotate with eye) */}
            {/* Top Lid */}
            <mesh ref={topLidRef} position={[0, 0, 0]}>
                 <sphereGeometry args={[1.08, 32, 16, 0, Math.PI * 2, 0, Math.PI/2]} />
                 <primitive object={SkinMaterial} />
            </mesh>

            {/* Bottom Lid */}
            <mesh ref={botLidRef} position={[0, 0, 0]}>
                 <sphereGeometry args={[1.08, 32, 16, 0, Math.PI * 2, Math.PI/2, Math.PI/2]} />
                 <primitive object={SkinMaterial} />
            </mesh>
        </group>
    );
};

// Brow Capsule
const Brow = ({ position, inner, outer, side }: { position: [number, number, number], inner: number, outer: number, side: 'left' | 'right' }) => {
    const meshRef = useRef<THREE.Mesh>(null);
    const sign = side === 'left' ? -1 : 1;

    useFrame(() => {
        if (meshRef.current) {
            const baseY = position[1];
            const innerY = inner / 100 * 0.6; 
            const outerY = outer / 100 * 0.6;
            const heightDiff = outerY - innerY;
            let rotZ = heightDiff * 0.5 * -sign; 
            
            meshRef.current.rotation.z = rotZ + (side === 'left' ? -0.1 : 0.1); 
            meshRef.current.position.y = baseY + (innerY + outerY) / 2;
        }
    });

    return (
        <mesh ref={meshRef} position={position}>
            <RoundedBox args={[1.8, 0.25, 0.4]} radius={0.1} smoothness={4}>
                <primitive object={SkinMaterial} />
            </RoundedBox>
        </mesh>
    );
}

const Mouth = ({ state }: { state: FacsState }) => {
    const {
        cornerUpperLeft, cornerUpperRight,
        cornerLowerLeft, cornerLowerRight,
        lipUpperLeft, lipUpperRight,
        lipLowerLeft, lipLowerRight,
        chinHorizontal, tongue
    } = state;

    // --- Configuration Constants ---
    const mouthWidthBase = 1.6;
    const mouthYBase = -1.5;
    const surfaceZ = 2.15; // Slightly pushed out from face
    
    const chinOffset = (chinHorizontal / 100) * 0.5;
    const widthModL = (cornerUpperLeft / 100) * 0.3;
    const widthModR = (cornerUpperRight / 100) * 0.3;

    // Corner L
    const cL_X = -mouthWidthBase / 2 + chinOffset - widthModL;
    const cL_Y = mouthYBase + (cornerUpperLeft/100 * 0.6) - (cornerLowerLeft/100 * 0.5);
    const cL_Z = surfaceZ - (cornerUpperLeft/100 * 0.2) - (cornerLowerLeft/100 * 0.1);
    
    // Corner R
    const cR_X = mouthWidthBase / 2 + chinOffset + widthModR;
    const cR_Y = mouthYBase + (cornerUpperRight/100 * 0.6) - (cornerLowerRight/100 * 0.5);
    const cR_Z = surfaceZ - (cornerUpperRight/100 * 0.2) - (cornerLowerRight/100 * 0.1);

    const cornerL = new THREE.Vector3(cL_X, cL_Y, cL_Z);
    const cornerR = new THREE.Vector3(cR_X, cR_Y, cR_Z);

    const lipTopBaseY = mouthYBase + 0.15;
    const sneerL = lipUpperLeft / 100;
    const sneerR = lipUpperRight / 100;

    const topCenter = new THREE.Vector3(chinOffset, lipTopBaseY + (sneerL + sneerR) * 0.2, surfaceZ + 0.1);
    const peakL = new THREE.Vector3(chinOffset - 0.4, lipTopBaseY + 0.15 + (sneerL * 0.5), surfaceZ + 0.12);
    const peakR = new THREE.Vector3(chinOffset + 0.4, lipTopBaseY + 0.15 + (sneerR * 0.5), surfaceZ + 0.12);

    const midTopL = new THREE.Vector3().lerpVectors(cornerL, peakL, 0.5); midTopL.y += 0.05;
    const midTopR = new THREE.Vector3().lerpVectors(cornerR, peakR, 0.5); midTopR.y += 0.05;

    const upperLipPoints = [cornerL, midTopL, peakL, topCenter, peakR, midTopR, cornerR];

    const lipBotBaseY = mouthYBase - 0.15;
    const openL = lipLowerLeft / 100;
    const openR = lipLowerRight / 100;
    const openAvg = (openL + openR) / 2;

    const botCenter = new THREE.Vector3(chinOffset, lipBotBaseY - (openAvg * 0.8), surfaceZ + 0.1);
    const midBotL = new THREE.Vector3().lerpVectors(cornerL, botCenter, 0.5); midBotL.y -= 0.05;
    const midBotR = new THREE.Vector3().lerpVectors(cornerR, botCenter, 0.5); midBotR.y -= 0.05;

    const lowerLipPoints = [cornerL, midBotL, botCenter, midBotR, cornerR];

    const tongueVisible = tongue > 0;
    const tongueY = botCenter.y + 0.3;
    const tongueZ = surfaceZ - 0.2;
    const tongueExt = tongue / 100;

    return (
        <group>
            {tongueVisible && (
                <mesh position={[chinOffset, tongueY, tongueZ + (tongueExt * 0.5)]} rotation={[0.2 - (tongueExt * 0.2), 0, 0]}>
                    <boxGeometry args={[0.8, 0.2, 1 + tongueExt]} />
                    <meshStandardMaterial color="#FF3B30" roughness={0.3} />
                    <mesh position={[0, 0, (1+tongueExt)/2]}>
                        <sphereGeometry args={[0.4, 16, 16]} />
                        <meshStandardMaterial color="#FF3B30" roughness={0.3} />
                    </mesh>
                </mesh>
            )}
            <LipMesh points={upperLipPoints} radius={0.14} />
            <LipMesh points={lowerLipPoints} radius={0.16} />
            <mesh position={cornerL} scale={[0.9, 0.9, 0.9]}><sphereGeometry args={[0.16]} /><primitive object={LipMaterial} /></mesh>
            <mesh position={cornerR} scale={[0.9, 0.9, 0.9]}><sphereGeometry args={[0.16]} /><primitive object={LipMaterial} /></mesh>
        </group>
    );
};

const RobotHead = ({ state, videoTexture }: { state: FacsState, videoTexture?: THREE.VideoTexture | null }) => {
    // Dynamic Eye Color
    const eyeHue = 210 + (state.reserved * 1.1); // Blue to Pink
    const pupilColor = `hsl(${eyeHue}, 100%, 60%)`;

    return (
        <group>
            {/* Main Face Shape */}
            <RoundedBox args={[4.5, 5.5, 3]} radius={0.8} smoothness={8} position={[0, 0, 0.5]}>
                <primitive object={TitaniumMaterial} />
            </RoundedBox>
            
            {/* Back of head */}
            <RoundedBox args={[4.2, 5.2, 2]} radius={1} smoothness={8} position={[0, 0, -1]}>
                <primitive object={TitaniumMaterial} />
            </RoundedBox>

            {/* Eyes */}
            <Eye 
                position={[-1.2, 0.5, 2]} 
                lookH={state.eyeLookHorizontal} 
                lookV={state.eyeLookVertical}
                lidTop={state.lidLeftTop}
                lidBot={state.lidLeftBottom}
                pupilColor={pupilColor}
                reflectionTexture={videoTexture}
            />
            <Eye 
                position={[1.2, 0.5, 2]} 
                lookH={state.eyeLookHorizontal} 
                lookV={state.eyeLookVertical}
                lidTop={state.lidRightTop}
                lidBot={state.lidRightBottom}
                pupilColor={pupilColor}
                reflectionTexture={videoTexture}
            />

            {/* Brows */}
            <Brow position={[-1.2, 1.8, 2.2]} inner={state.browLeftInner} outer={state.browLeftOuter} side="left" />
            <Brow position={[1.2, 1.8, 2.2]} inner={state.browRightInner} outer={state.browRightOuter} side="right" />

            {/* Mouth */}
            <Mouth state={state} />

            {/* Cheeks (Light emission based on smile) */}
            <mesh position={[-1.8, -0.5, 1.8]}>
                 <sphereGeometry args={[0.5]} />
                 <meshBasicMaterial color="#FF2D55" transparent opacity={(state.cornerUpperLeft / 200)} />
            </mesh>
            <mesh position={[1.8, -0.5, 1.8]}>
                 <sphereGeometry args={[0.5]} />
                 <meshBasicMaterial color="#FF2D55" transparent opacity={(state.cornerUpperRight / 200)} />
            </mesh>
        </group>
    );
};

// Internal component to handle camera responsiveness inside Canvas context
const ResponsiveCamera = () => {
    const { camera, size } = useThree();

    useLayoutEffect(() => {
        // If the viewport aspect ratio is portrait (narrow), move camera back to fit the face
        const aspect = size.width / size.height;
        if (aspect < 1) {
            // Formula: Zoom out as it gets narrower. 
            // Base Z is 10. For aspect 0.5 (mobile), maybe Z=15.
            camera.position.z = 10 + ((1 - aspect) * 8);
        } else {
            camera.position.z = 10;
        }
        camera.updateProjectionMatrix();
    }, [camera, size]);

    return null;
}

export const RobotFace: React.FC<RobotFaceProps> = ({ state, videoStream }) => {
  const [videoTexture, setVideoTexture] = React.useState<THREE.VideoTexture | null>(null);

  useEffect(() => {
    if (videoStream) {
        const video = document.createElement('video');
        video.srcObject = videoStream;
        video.playsInline = true;
        video.autoplay = true;
        video.muted = true;
        video.play().catch(console.error);
        
        const texture = new THREE.VideoTexture(video);
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.colorSpace = THREE.SRGBColorSpace;
        
        setVideoTexture(texture);

        return () => {
            video.pause();
            video.srcObject = null;
            texture.dispose();
        }
    } else {
        setVideoTexture(null);
    }
  }, [videoStream]);

  return (
    <div className="w-full h-full">
        <Canvas shadows camera={{ position: [0, 0, 10], fov: 45 }}>
            <Suspense fallback={null}>
                <ResponsiveCamera />
                <color attach="background" args={['#000000']} />
                
                {/* Lighting Studio */}
                <ambientLight intensity={0.5} />
                <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} castShadow />
                <pointLight position={[-10, -10, -10]} intensity={0.5} color="blue" />
                
                {/* Environment Reflections */}
                <Environment files="https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/potsdamer_platz_1k.hdr" />

                {/* The Robot */}
                <Float floatIntensity={0.5} rotationIntensity={0.2} speed={2}>
                    <RobotHead state={state} videoTexture={videoTexture} />
                </Float>

                <ContactShadows position={[0, -4, 0]} opacity={0.5} scale={10} blur={2.5} far={4} />
            </Suspense>
        </Canvas>
    </div>
  );
};
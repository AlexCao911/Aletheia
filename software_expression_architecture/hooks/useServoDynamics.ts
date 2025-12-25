import { useState, useEffect, useRef } from 'react';
import { FacsState, INITIAL_STATE } from '../types';

export const useServoDynamics = (targetState: FacsState) => {
  const [displayState, setDisplayState] = useState<FacsState>(INITIAL_STATE);
  
  // Refs to hold mutable state for the animation loop
  const currentStateRef = useRef<FacsState>(INITIAL_STATE);
  const velocityRef = useRef<Record<string, number>>({});
  const targetStateRef = useRef<FacsState>(targetState);
  const rafRef = useRef<number>(0);

  // Initialize velocity map
  useEffect(() => {
    Object.keys(INITIAL_STATE).forEach(key => {
        if (typeof velocityRef.current[key] === 'undefined') {
            velocityRef.current[key] = 0;
        }
    });
  }, []);

  // Update target ref when prop changes
  useEffect(() => {
    targetStateRef.current = targetState;
  }, [targetState]);

  useEffect(() => {
    const loop = () => {
      let hasChanges = false;
      const nextState = { ...currentStateRef.current };
      const velocities = velocityRef.current;
      const targets = targetStateRef.current;

      Object.keys(targets).forEach((k) => {
        const key = k as keyof FacsState;
        const target = targets[key];
        const current = nextState[key];
        let vel = velocities[key] || 0;

        // --- PHYSICS PARAMETERS ---
        // Simulating different material properties for different facial features
        
        let k_stiffness = 0.15; // Default (Balanced)
        let k_damping = 0.75;

        // 1. Fleshy / Skin (Lips, Corners, Cheeks)
        // More bounce (lower damping), softer spring (lower stiffness) to simulate elasticity
        if (key.includes('lip') || key.includes('corner')) {
             k_stiffness = 0.08;
             k_damping = 0.65; // Bouncy
        } 
        // 2. Heavy Skin (Brows)
        // Slower acceleration, moderate damping
        else if (key.includes('brow')) {
             k_stiffness = 0.1;
             k_damping = 0.7;
        } 
        // 3. Eyes (Gaze) - Needs to be very smooth to avoid jitter/shaking
        else if (key.includes('eye')) {
             k_stiffness = 0.05; // Low stiffness for heavy/smooth feel
             k_damping = 0.85;   // High damping to kill high-freq noise
        }
        // 4. Mechanical Fast (Lids, Jaw)
        // Snappy response for blinks
        else if (key.includes('lid') || key.includes('chin')) {
             k_stiffness = 0.25;
             k_damping = 0.72; 
        }

        // Spring Force: F = -k * x
        const force = (target - current) * k_stiffness;
        
        // Damping: F_d = -d * v
        vel = (vel + force) * k_damping;
        
        const nextVal = current + vel;
        
        // Snap to target if close enough to stop infinite micro-oscillations
        // Use a stricter threshold for eyes to prevent micro-jitter at rest
        const stopThreshold = key.includes('eye') ? 0.2 : 0.1;

        if (Math.abs(target - nextVal) < stopThreshold && Math.abs(vel) < stopThreshold) {
            nextState[key] = target;
            velocities[key] = 0;
        } else {
            nextState[key] = nextVal;
            velocities[key] = vel;
            hasChanges = true;
        }
      });

      if (hasChanges) {
        currentStateRef.current = nextState;
        setDisplayState(nextState);
        rafRef.current = requestAnimationFrame(loop);
      } else {
         // Even if physics stopped, ensure we display the latest computed state
         // (Handled by the snap logic above, but good to keep loop ready if target changes)
         rafRef.current = requestAnimationFrame(loop);
      }
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, []); // Empty dependency array: Loop runs forever, reading from refs

  return displayState;
};
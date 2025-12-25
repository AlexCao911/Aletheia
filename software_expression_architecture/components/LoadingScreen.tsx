import React, { useEffect, useState } from 'react';

interface LoadingScreenProps {
  onComplete: () => void;
}

export const LoadingScreen: React.FC<LoadingScreenProps> = ({ onComplete }) => {
  const [progress, setProgress] = useState(0);
  const [fading, setFading] = useState(false);

  // Configuration for the graphic equalizer style
  const SEGMENT_COUNT = 12;

  useEffect(() => {
    // Simulation of a complex system initialization
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        // Organic, "thinking" loading speed (stalls and jumps)
        const jump = Math.random();
        let increment = 0;
        if (jump > 0.8) increment = 15; // Big jump
        else if (jump > 0.5) increment = 5; // Small step
        else increment = 0.5; // Stall/Thinking
        
        return Math.min(prev + increment, 100);
      });
    }, 120);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (progress === 100) {
      const timeout = setTimeout(() => {
        setFading(true);
        setTimeout(onComplete, 800);
      }, 600);
      return () => clearTimeout(timeout);
    }
  }, [progress, onComplete]);

  return (
    <div 
        className={`fixed inset-0 z-[100] bg-black flex items-center justify-center transition-opacity duration-1000 ease-in-out ${fading ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}
    >
      {/* 
         Graphic Design Concept: "The Rhythmic Grating"
         A sequence of vertical forms that creates a sense of digital structure.
         Not a thin line, but a substantial, engineered object.
      */}
      <div className={`flex items-center gap-[6px] transition-transform duration-1000 ${fading ? 'scale-110 opacity-0' : 'scale-100 opacity-100'}`}>
        {Array.from({ length: SEGMENT_COUNT }).map((_, i) => {
            // Calculate if this segment should be lit based on progress
            const threshold = (i + 1) * (100 / SEGMENT_COUNT);
            const isActive = progress >= threshold - (100 / SEGMENT_COUNT / 2); // Light up slightly early for fluidity
            
            return (
                <div 
                    key={i}
                    className="w-1.5 h-6 rounded-full transition-all duration-300 ease-out"
                    style={{
                        backgroundColor: isActive ? '#FFFFFF' : '#27272a', // White vs Zinc-800
                        boxShadow: isActive ? '0 0 12px rgba(255, 255, 255, 0.4)' : 'none',
                        transform: isActive ? 'scaleY(1.0)' : 'scaleY(0.8)', // Subtle "pop" when active
                        opacity: isActive ? 1 : 0.3
                    }}
                />
            );
        })}
      </div>
    </div>
  );
};
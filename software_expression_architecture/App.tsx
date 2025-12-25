import React, { useState, useRef, useEffect } from 'react';
import { ControlPanel } from './components/ControlPanel';
import { RobotFace } from './components/RobotFace';
import { LoadingScreen } from './components/LoadingScreen';
import { FacsState, INITIAL_STATE } from './types';
import { useServoDynamics } from './hooks/useServoDynamics';
import { initializeFaceLandmarker, detectExpression } from './services/mediaPipeService';

export default function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [logicalState, setLogicalState] = useState<FacsState>(INITIAL_STATE);
  const displayState = useServoDynamics(logicalState);
  
  // Camera State
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false); // New: Ensures video is actually playing
  const [videoStream, setVideoStream] = useState<MediaStream | null>(null);
  const [modelReady, setModelReady] = useState(false);
  
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const requestRef = useRef<number>(0);
  const lastVideoTimeRef = useRef<number>(-1);

  // Initialize MediaPipe on Mount
  useEffect(() => {
    initializeFaceLandmarker().then((result) => {
        if (result) {
            console.log("MediaPipe Face Landmarker Ready");
            setModelReady(true);
        } else {
            console.error("Failed to initialize MediaPipe");
        }
    });
  }, []);

  const handleChange = (key: keyof FacsState, value: number) => {
    setLogicalState(prev => ({ ...prev, [key]: value }));
  };

  const handleApplyState = (newState: FacsState) => {
    setLogicalState(newState);
  };

  const handleReset = () => {
    setLogicalState(INITIAL_STATE);
    stopCamera();
  };

  const startCamera = async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 }
            } 
        });
        
        setVideoStream(stream);
        setIsCameraActive(true);
        
        // Create a hidden video element for MediaPipe consumption
        let vid = videoRef.current;
        if (!vid) {
            vid = document.createElement('video');
            vid.playsInline = true;
            vid.autoplay = true;
            vid.muted = true;
            videoRef.current = vid;
        }

        // Setup listener for when video data is actually ready
        vid.onloadeddata = () => {
            console.log("Video data loaded, enabling tracking...");
            setIsVideoReady(true);
        };

        vid.srcObject = stream;
        await vid.play();

    } catch (err) {
        console.error("Failed to access camera:", err);
        alert("Could not access camera. Please allow permissions.");
        setIsCameraActive(false);
        setIsVideoReady(false);
    }
  };

  const stopCamera = () => {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
    }
    setVideoStream(null);
    setIsCameraActive(false);
    setIsVideoReady(false);
    
    if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
    }

    if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.srcObject = null;
        // Don't nullify videoRef.current, we can reuse the element
    }
  };

  const toggleCamera = () => {
    if (isCameraActive) {
        stopCamera();
    } else {
        startCamera();
    }
  }

  // Real-time Detection Loop
  useEffect(() => {
    // Strictly wait for ALL conditions: Camera Active, Model Loaded, AND Video Playing
    if (!isCameraActive || !modelReady || !isVideoReady || !videoRef.current) return;

    console.log("Starting detection loop");
    let lastTime = 0;

    const animate = (time: number) => {
        if (videoRef.current && !videoRef.current.paused && !videoRef.current.ended) {
            // throttle slightly if needed, but RAF is usually fine
            
            // Detect expressions
            const detectedState = detectExpression(videoRef.current, time);
            
            if (detectedState) {
                setLogicalState(detectedState);
            }
        }
        requestRef.current = requestAnimationFrame(animate);
    };

    requestRef.current = requestAnimationFrame(animate);

    return () => {
        if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isCameraActive, modelReady, isVideoReady]); // Re-run when video becomes ready


  return (
    <div className="relative h-[100dvh] w-screen overflow-hidden bg-black font-sans selection:bg-apple-blue/30 flex flex-col md:flex-row">
      
      {/* Loading Screen Overlay */}
      {isLoading && <LoadingScreen onComplete={() => setIsLoading(false)} />}

      {/* 1. Global Background Environment (Behind everything) */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] rounded-full bg-blue-900/10 blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-purple-900/10 blur-[120px]" />
      </div>

      {/* 2. Sidebar / Controls Panel */}
      <div className="order-2 md:order-1 w-full md:w-[360px] h-[55%] md:h-full p-4 md:p-6 z-20 flex flex-col shrink-0 transition-all duration-300">
        <div className="flex-1 bg-apple-glass backdrop-blur-xl border border-apple-glassBorder rounded-t-2xl md:rounded-2xl shadow-2xl overflow-hidden flex flex-col transition-all hover:border-white/10">
          
          {/* Header */}
          <div className="h-12 md:h-14 px-4 md:px-6 flex items-center justify-between border-b border-white/5 shrink-0">
            <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500/80" />
                <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                <div className="w-3 h-3 rounded-full bg-green-500/80" />
            </div>
            
            <span className="md:hidden text-xs font-semibold text-white/40 tracking-widest">CONTROLS</span>

            <button 
                onClick={handleReset}
                className="text-[11px] font-medium text-apple-subtext hover:text-white transition-colors uppercase tracking-wider"
            >
                Reset
            </button>
          </div>

          {/* Scrollable Controls */}
          <ControlPanel 
            state={logicalState} 
            onChange={handleChange} 
            onApplyState={handleApplyState} 
            onReset={handleReset}
            isCameraActive={isCameraActive}
            onToggleCamera={toggleCamera}
          />
        </div>
      </div>

      {/* 3. Main Content Area (Face) */}
      <div className="order-1 md:order-2 flex-1 relative h-[45%] md:h-full flex flex-col z-10 min-h-0">
        
        {/* Top Right Status */}
        <div className="absolute top-4 right-4 md:top-6 md:right-6 z-20 pointer-events-none">
            <div className="flex items-center space-x-2 px-3 py-1.5 md:px-4 md:py-2 bg-white/5 backdrop-blur-md border border-white/10 rounded-full">
                <div className={`w-1.5 h-1.5 rounded-full ${isCameraActive ? (isVideoReady ? 'bg-red-500' : 'bg-yellow-500') : (modelReady ? 'bg-green-500' : 'bg-yellow-500')} animate-pulse`}></div>
                <span className="text-[10px] font-medium text-white/60 tracking-wider">
                    {isCameraActive ? (isVideoReady ? 'LIVE TRACKING' : 'INITIALIZING CAM...') : (modelReady ? 'SYSTEM READY' : 'LOADING AI...')}
                </span>
            </div>
        </div>

        {/* Center Canvas: Face */}
        <div className="absolute inset-0">
            <RobotFace state={displayState} videoStream={videoStream} />
        </div>

      </div>

    </div>
  );
}
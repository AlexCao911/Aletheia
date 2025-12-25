import { FilesetResolver, FaceLandmarker } from "@mediapipe/tasks-vision";
import { FacsState, INITIAL_STATE } from "../types";

let faceLandmarker: FaceLandmarker | null = null;
let hasLoggedSuccess = false;

// Initialize the MediaPipe Face Landmarker
export const initializeFaceLandmarker = async () => {
  if (faceLandmarker) return faceLandmarker;

  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
        delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        runningMode: "VIDEO",
        numFaces: 1
    });
    
    console.log("MediaPipe initialized successfully");
    return faceLandmarker;
  } catch (error) {
    console.error("MediaPipe initialization error:", error);
    return null;
  }
};

// Map MediaPipe Blendshapes (0.0 - 1.0) to our Robot FACS (0-100 or -100 to 100)
// This mirrors the expressions from the human face to the robot face.
const mapBlendshapesToState = (blendshapes: any[]): FacsState => {
  const scores: Record<string, number> = {};
  
  // Create a lookup map for easier access
  blendshapes.forEach(b => {
    scores[b.categoryName] = b.score;
  });

  // Helper to scale 0-1 to 0-100 safely
  const s = (name: string, multiplier: number = 100) => {
    return (scores[name] || 0) * multiplier;
  };

  const newState: FacsState = { ...INITIAL_STATE };

  // --- BROWS ---
  // Note: MediaPipe returns left/right relative to the PERSON. 
  // We mirror this: Person's Left -> Robot's Right (Screen Right).
  
  // Mapping Strategy: Mirroring
  // Person Right Brow -> Robot Right Brow (Screen Right)
  newState.browRightInner = s('browInnerUp', 150); // Single browInner in MP, usually affects both
  newState.browLeftInner = s('browInnerUp', 150);
  
  newState.browRightOuter = s('browOuterUpRight', 150); 
  newState.browLeftOuter = s('browOuterUpLeft', 150);

  // --- EYES ---
  // Gaze
  // Range analysis: eyeLookOutRight (0-1), eyeLookInRight (0-1).
  // Extreme Right look: OutRight=1, InLeft=1. lookRight=1, lookLeft=-1. Diff=2.
  // We use a lower multiplier (50) to keep it within -100 to 100 and reduce sensitivity to noise.
  const lookRight = s('eyeLookOutRight') - s('eyeLookInRight'); 
  const lookLeft = s('eyeLookOutLeft') - s('eyeLookInLeft');
  
  let hGaze = (lookRight - lookLeft) * 50; 
  hGaze = Math.max(-100, Math.min(100, hGaze)); // Clamp
  newState.eyeLookHorizontal = hGaze;

  const lookUp = s('eyeLookUpRight') + s('eyeLookUpLeft');
  const lookDown = s('eyeLookDownRight') + s('eyeLookDownLeft');
  
  let vGaze = (lookUp - lookDown) * 50;
  vGaze = Math.max(-100, Math.min(100, vGaze)); // Clamp
  newState.eyeLookVertical = vGaze;

  // Lids (Blink & Squint)
  newState.lidRightTop = s('eyeBlinkRight', 120);
  newState.lidLeftTop = s('eyeBlinkLeft', 120);
  
  newState.lidRightBottom = s('eyeSquintRight', 100);
  newState.lidLeftBottom = s('eyeSquintLeft', 100);

  // --- JAW / CHIN ---
  const jawOpen = s('jawOpen');
  const jawLeft = s('jawLeft');
  const jawRight = s('jawRight');
  newState.chinHorizontal = (jawRight - jawLeft) * 2; // -100 to 100

  // --- MOUTH LIPS ---
  // Upper Lip Raise (Sneer)
  newState.lipUpperRight = s('mouthUpperUpRight', 140);
  newState.lipUpperLeft = s('mouthUpperUpLeft', 140);

  // Lower Lip Down (Open mouth) - heavily influenced by jawOpen
  newState.lipLowerRight = Math.max(s('mouthLowerDownRight', 150), jawOpen * 0.8);
  newState.lipLowerLeft = Math.max(s('mouthLowerDownLeft', 150), jawOpen * 0.8);

  // --- MOUTH CORNERS ---
  // Smile (Up)
  newState.cornerUpperRight = s('mouthSmileRight', 130);
  newState.cornerUpperLeft = s('mouthSmileLeft', 130);

  // Frown (Down)
  newState.cornerLowerRight = s('mouthFrownRight', 120);
  newState.cornerLowerLeft = s('mouthFrownLeft', 120);
  
  // Stretch (Grimace/Fear) usually pulls corners sideways/down
  const stretch = s('mouthStretchLeft') + s('mouthStretchRight');
  if (stretch > 0.5) {
     newState.cornerLowerLeft += stretch * 20;
     newState.cornerLowerRight += stretch * 20;
  }

  // --- EXTRAS ---
  // Puff -> Reserved
  const puff = s('cheekPuff', 100);
  newState.reserved = puff > 10 ? puff : 50; // Default to 50 (blue eyes), puff changes color

  return newState;
};

export const detectExpression = (video: HTMLVideoElement, timestamp: number): FacsState | null => {
  if (!faceLandmarker) return null;
  
  // Safety check: Ensure video has valid dimensions before processing
  if (video.videoWidth === 0 || video.videoHeight === 0) return null;

  try {
    const result = faceLandmarker.detectForVideo(video, timestamp);
    
    if (result.faceBlendshapes && result.faceBlendshapes.length > 0 && result.faceBlendshapes[0].categories) {
      if (!hasLoggedSuccess) {
        console.log("First face detected successfully!");
        hasLoggedSuccess = true;
      }
      return mapBlendshapesToState(result.faceBlendshapes[0].categories);
    }
  } catch (e) {
    // console.warn("MP Detect Error", e);
  }
  return null;
};
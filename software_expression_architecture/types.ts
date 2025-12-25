export interface FacsState {
  // Brows (4)
  browLeftInner: number; // 0 (Neutral/Down) to 100 (Up - AU1)
  browLeftOuter: number; // 0 (Neutral/Down) to 100 (Up - AU2)
  browRightInner: number; // 0 (Neutral/Down) to 100 (Up - AU1)
  browRightOuter: number; // 0 (Neutral/Down) to 100 (Up - AU2)

  // Eyes (2)
  eyeLookHorizontal: number; // -100 (Left) to 100 (Right)
  eyeLookVertical: number; // -100 (Down) to 100 (Up)

  // Lids (4)
  lidLeftTop: number; // 0 (Open) to 100 (Closed - AU43/45)
  lidLeftBottom: number; // 0 (Open) to 100 (Squint - AU7)
  lidRightTop: number; // 0 (Open) to 100 (Closed - AU43/45)
  lidRightBottom: number; // 0 (Open) to 100 (Squint - AU7)

  // Chin/Jaw (1)
  chinHorizontal: number; // -100 (Left) to 100 (Right)

  // Mouth - Lips (4)
  lipUpperLeft: number;  // LUL: 0 (Neutral) to 100 (Up/Sneer - AU10)
  lipUpperRight: number; // LUR: 0 (Neutral) to 100 (Up/Sneer - AU10)
  lipLowerLeft: number;  // LLL: 0 (Neutral) to 100 (Down/Open - AU16/25)
  lipLowerRight: number; // LLR: 0 (Neutral) to 100 (Down/Open - AU16/25)

  // Mouth - Corners (4)
  cornerUpperLeft: number;  // CUL: 0 (Neutral) to 100 (Up/Smile - AU12)
  cornerUpperRight: number; // CUR: 0 (Neutral) to 100 (Up/Smile - AU12)
  cornerLowerLeft: number;  // CLL: 0 (Neutral) to 100 (Down/Frown - AU15)
  cornerLowerRight: number; // CLR: 0 (Neutral) to 100 (Down/Frown - AU15)

  // Tongue (1)
  tongue: number; // 0 (Hidden) to 100 (Extended - AU19)

  // Reserved (1)
  reserved: number; // 0 to 100 (Generic/Auxiliary - AU33 Puffer)
}

export const INITIAL_STATE: FacsState = {
  browLeftInner: 0,
  browLeftOuter: 0,
  browRightInner: 0,
  browRightOuter: 0,
  eyeLookHorizontal: 0,
  eyeLookVertical: 0,
  lidLeftTop: 0,
  lidLeftBottom: 0,
  lidRightTop: 0,
  lidRightBottom: 0,
  chinHorizontal: 0,
  lipUpperLeft: 0,
  lipUpperRight: 0,
  lipLowerLeft: 0,
  lipLowerRight: 0,
  cornerUpperLeft: 0,
  cornerUpperRight: 0,
  cornerLowerLeft: 0,
  cornerLowerRight: 0,
  tongue: 0,
  reserved: 50,
};

export const DOF_LABELS: Record<keyof FacsState, string> = {
  browLeftInner: "L Brow Inner (AU1)",
  browLeftOuter: "L Brow Outer (AU2)",
  browRightInner: "R Brow Inner (AU1)",
  browRightOuter: "R Brow Outer (AU2)",
  eyeLookHorizontal: "Eyes Pan L/R",
  eyeLookVertical: "Eyes Tilt U/D",
  lidLeftTop: "L Lid Top",
  lidLeftBottom: "L Lid Bottom (AU7)",
  lidRightTop: "R Lid Top",
  lidRightBottom: "R Lid Bottom (AU7)",
  chinHorizontal: "Chin L/R",
  lipUpperLeft: "Lip Upper L (LUL)",
  lipUpperRight: "Lip Upper R (LUR)",
  lipLowerLeft: "Lip Lower L (LLL)",
  lipLowerRight: "Lip Lower R (LLR)",
  cornerUpperLeft: "Corner Up L (CUL)",
  cornerUpperRight: "Corner Up R (CUR)",
  cornerLowerLeft: "Corner Down L (CLL)",
  cornerLowerRight: "Corner Down R (CLR)",
  tongue: "Tongue",
  reserved: "Reserved (Puff/Hue)",
};

export const DOF_GROUPS = [
  { name: "Brows", keys: ['browLeftInner', 'browLeftOuter', 'browRightInner', 'browRightOuter'] },
  { name: "Eyes & Lids", keys: ['eyeLookHorizontal', 'eyeLookVertical', 'lidLeftTop', 'lidLeftBottom', 'lidRightTop', 'lidRightBottom'] },
  { name: "Mouth: Lips", keys: ['lipUpperLeft', 'lipUpperRight', 'lipLowerLeft', 'lipLowerRight'] },
  { name: "Mouth: Corners", keys: ['cornerUpperLeft', 'cornerUpperRight', 'cornerLowerLeft', 'cornerLowerRight'] },
  { name: "Misc", keys: ['chinHorizontal', 'tongue', 'reserved'] },
];

// Static expressions
export const EXPRESSION_LIBRARY: Record<string, Partial<FacsState>> = {
  "Neutral": INITIAL_STATE,
  "Happiness (Joy)": {
    cornerUpperLeft: 100, cornerUpperRight: 100,
    lidLeftBottom: 40, lidRightBottom: 40,
    lipUpperLeft: 20, lipUpperRight: 20,
    browLeftOuter: 10, browRightOuter: 10,
  },
  "Sadness": {
    browLeftInner: 80, browRightInner: 80,
    cornerLowerLeft: 80, cornerLowerRight: 80,
    lidLeftTop: 30, lidRightTop: 30,
    lipUpperLeft: 0, lipUpperRight: 0,
    eyeLookVertical: -20,
  },
  "Surprise": {
    browLeftInner: 100, browLeftOuter: 100, browRightInner: 100, browRightOuter: 100,
    lidLeftTop: 0, lidRightTop: 0,
    lipLowerLeft: 100, lipLowerRight: 100,
    lipUpperLeft: 30, lipUpperRight: 30,
  },
  "Anger": {
    browLeftInner: 0, browRightInner: 0, 
    browLeftOuter: 20, browRightOuter: 20,
    lidLeftTop: 20, lidRightTop: 20,
    lidLeftBottom: 60, lidRightBottom: 60,
    lipUpperLeft: 0, lipUpperRight: 0,
    cornerLowerLeft: 20, cornerLowerRight: 20,
    chinHorizontal: 0,
  },
  "Fear": {
    browLeftInner: 100, browRightInner: 100,
    browLeftOuter: 60, browRightOuter: 60,
    lidLeftTop: 0, lidRightTop: 0,
    cornerUpperLeft: 40, cornerUpperRight: 40,
    cornerLowerLeft: 40, cornerLowerRight: 40,
    lipLowerLeft: 50, lipLowerRight: 50,
  },
  "Disgust": {
    lipUpperLeft: 100, lipUpperRight: 100,
    cornerLowerLeft: 40, cornerLowerRight: 40,
    lidLeftBottom: 60, lidRightBottom: 60,
    browLeftInner: 30, browRightInner: 30,
  },
  "Contempt": {
    cornerUpperLeft: 60, cornerUpperRight: 0,
    lipUpperLeft: 30, lipUpperRight: 0,
    lidLeftBottom: 30, lidRightBottom: 10,
    chinHorizontal: -15,
    eyeLookHorizontal: 15,
  },
  "Skepticism": {
    browLeftOuter: 100, browLeftInner: 80,
    browRightOuter: 0, browRightInner: 20,
    lidLeftTop: 0, lidRightTop: 45,
    cornerLowerRight: 30,
    chinHorizontal: 10,
  },
  "Flirty (Wink)": {
    lidLeftTop: 100, lidLeftBottom: 90,
    lidRightTop: 0, lidRightBottom: 20,
    cornerUpperLeft: 70, cornerUpperRight: 40,
    browRightOuter: 40,
    chinHorizontal: 15,
    eyeLookHorizontal: 20,
  },
  "Boredom": {
    lidLeftTop: 55, lidRightTop: 55,
    browLeftInner: 0, browRightInner: 0,
    cornerLowerLeft: 30, cornerLowerRight: 30,
    eyeLookVertical: 40,
    eyeLookHorizontal: 30,
    lipLowerLeft: 30, lipLowerRight: 30,
  },
  "Awe": {
    browLeftInner: 50, browRightInner: 50,
    browLeftOuter: 50, browRightOuter: 50,
    lidLeftTop: 0, lidRightTop: 0,
    lipLowerLeft: 70, lipLowerRight: 70,
    lipUpperLeft: 10, lipUpperRight: 10,
    eyeLookVertical: 20,
  },
  "Suspicion": {
    lidLeftBottom: 85, lidRightBottom: 85,
    lidLeftTop: 45, lidRightTop: 45,
    browLeftInner: 0, browRightInner: 0,
    eyeLookHorizontal: -35,
    chinHorizontal: 10,
    cornerLowerLeft: 20, cornerLowerRight: 20,
  },
  "Rage": {
    browLeftInner: 0, browRightInner: 0,
    browLeftOuter: 40, browRightOuter: 40,
    lidLeftTop: 10, lidRightTop: 10,
    lidLeftBottom: 80, lidRightBottom: 80,
    lipLowerLeft: 80, lipLowerRight: 80,
    lipUpperLeft: 40, lipUpperRight: 40,
    cornerLowerLeft: 60, cornerLowerRight: 60,
    reserved: 80, // Red eyes/Face
  }
};

// Dynamic Actions - mapped to specific AUs with intensity
export const DYNAMIC_ACTIONS: Record<string, Partial<FacsState>> = {
  // Brows
  "AU1: Inner Brow": { browLeftInner: 100, browRightInner: 100 },
  "AU2: Outer Brow": { browLeftOuter: 100, browRightOuter: 100 },
  "AU4: Brow Lower": { browLeftInner: 0, browRightInner: 0, browLeftOuter: 20, browRightOuter: 20, lidLeftTop: 30, lidRightTop: 30 },
  
  // Lids & Eyes
  "AU5: Upper Lid": { lidLeftTop: 0, lidRightTop: 0, browLeftInner: 50, browRightInner: 50 },
  "AU6: Cheek Raise": { lidLeftBottom: 80, lidRightBottom: 80, cornerUpperLeft: 50, cornerUpperRight: 50 },
  "AU7: Lid Tighten": { lidLeftBottom: 60, lidRightBottom: 60, lidLeftTop: 20, lidRightTop: 20 },
  "AU45: Blink": { lidLeftTop: 100, lidRightTop: 100, lidLeftBottom: 100, lidRightBottom: 100 },
  
  // Nose & Mid-Face
  "AU9: Nose Wrinkle": { lipUpperLeft: 100, lipUpperRight: 100, browLeftInner: 30, browRightInner: 30 },
  "AU10: Upper Lip": { lipUpperLeft: 80, lipUpperRight: 80 },
  
  // Mouth Corners
  "AU12: Smile": { cornerUpperLeft: 100, cornerUpperRight: 100 },
  "AU14: Dimple": { cornerUpperLeft: 30, cornerUpperRight: 30, cornerLowerLeft: 30, cornerLowerRight: 30 },
  "AU15: Frown": { cornerLowerLeft: 100, cornerLowerRight: 100 },
  
  // Lips
  "AU16: Lower Lip": { lipLowerLeft: 100, lipLowerRight: 100 },
  "AU18: Pucker": { lipUpperLeft: 40, lipUpperRight: 40, lipLowerLeft: 20, lipLowerRight: 20, cornerLowerLeft: 40, cornerLowerRight: 40 },
  "AU20: Stretch": { cornerUpperLeft: 50, cornerUpperRight: 50, cornerLowerLeft: 50, cornerLowerRight: 50 },
  "AU25: Part Lips": { lipLowerLeft: 40, lipLowerRight: 40 },
  
  // Auxiliary
  "AU33: Puff": { reserved: 100, lidLeftBottom: 40, lidRightBottom: 40 },
  "AU19: Tongue": { tongue: 80, lipLowerLeft: 60, lipLowerRight: 60 },
};
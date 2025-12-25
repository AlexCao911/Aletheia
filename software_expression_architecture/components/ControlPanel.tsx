import React, { useState } from 'react';
import { FacsState, DOF_LABELS, DOF_GROUPS, EXPRESSION_LIBRARY, DYNAMIC_ACTIONS, INITIAL_STATE } from '../types';
import { ChevronDown, ChevronRight, Zap, Play, Eye } from 'lucide-react';

interface ControlPanelProps {
  state: FacsState;
  onChange: (key: keyof FacsState, value: number) => void;
  onApplyState: (newState: FacsState) => void;
  onReset: () => void;
  isCameraActive?: boolean;
  onToggleCamera?: () => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({ 
    state, 
    onChange, 
    onApplyState, 
    onReset,
    isCameraActive = false,
    onToggleCamera 
}) => {
  // Use a Record to track open states for multiple sections
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    "Brows": true,
    "Eyes & Lids": true,
    "Mouth: Lips": false,
    "Mouth: Corners": false,
    "Misc": false
  });

  const toggleSection = (name: string) => {
    setOpenSections(prev => ({
        ...prev,
        [name]: !prev[name]
    }));
  };

  return (
    <div className="flex-1 overflow-y-auto no-scrollbar pb-6">
      <div className="p-4 space-y-6">
        
        {/* Vision Mode Toggle */}
        {onToggleCamera && (
             <div className="bg-white/5 border border-white/10 rounded-xl p-1 flex items-center justify-between">
                <div className="flex items-center gap-3 px-3">
                    <div className={`p-1.5 rounded-full ${isCameraActive ? 'bg-red-500/20 text-red-400' : 'bg-white/10 text-white/40'}`}>
                        <Eye size={16} />
                    </div>
                    <div>
                        <div className="text-[13px] font-medium text-white/90">Vision Mode</div>
                        <div className="text-[10px] text-white/40">Mirror emotion & reflections</div>
                    </div>
                </div>
                <button 
                    onClick={onToggleCamera}
                    className={`px-4 py-1.5 rounded-lg text-[11px] font-semibold transition-all ${isCameraActive ? 'bg-red-500 text-white shadow-lg shadow-red-500/20' : 'bg-white/10 text-white/60 hover:bg-white/20'}`}
                >
                    {isCameraActive ? 'Active' : 'Enable'}
                </button>
            </div>
        )}

        {/* Quick Actions (Grid) */}
        <div>
            <h3 className="text-[10px] font-semibold text-apple-subtext uppercase tracking-widest mb-3 px-2">Presets</h3>
            <div className="grid grid-cols-2 gap-2">
                {Object.entries(EXPRESSION_LIBRARY).map(([name, preset]) => (
                    <button
                        key={name}
                        onClick={() => onApplyState({ ...INITIAL_STATE, ...preset })}
                        className="relative overflow-hidden group px-3 py-3 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 transition-all active:scale-95 text-left"
                    >
                        <span className="relative z-10 text-[12px] font-medium text-white/80 group-hover:text-white">{name}</span>
                        {/* Subtle gradient flash on hover */}
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:animate-[shimmer_1s_infinite]" />
                    </button>
                ))}
            </div>
        </div>

        {/* Accordion Controls */}
        <div>
            <h3 className="text-[10px] font-semibold text-apple-subtext uppercase tracking-widest mb-3 px-2">Manual Control</h3>
            <div className="space-y-2">
                {DOF_GROUPS.map((group) => {
                    const isOpen = openSections[group.name];
                    return (
                        <div key={group.name} className={`rounded-xl transition-all duration-300 ${isOpen ? 'bg-white/5 border border-white/10' : 'bg-transparent hover:bg-white/5'}`}>
                            <button 
                                onClick={() => toggleSection(group.name)}
                                className="w-full flex items-center justify-between px-4 py-3"
                            >
                                <span className={`text-[13px] font-medium transition-colors ${isOpen ? 'text-white' : 'text-apple-subtext'}`}>
                                    {group.name}
                                </span>
                                {isOpen ? <ChevronDown size={14} className="text-white/50" /> : <ChevronRight size={14} className="text-white/30" />}
                            </button>
                            
                            {/* Smooth Collapse */}
                            {isOpen && (
                                <div className="px-4 pb-4 space-y-5 animate-in slide-in-from-top-2 fade-in duration-200">
                                    {group.keys.map((key) => {
                                        const k = key as keyof FacsState;
                                        const isSigned = key.includes("Look") || key.includes("Horizontal") || key.includes("Vertical");
                                        const min = isSigned ? -100 : 0;
                                        const max = 100;
                                        const val = state[k];

                                        return (
                                            <div key={k} className="space-y-1.5">
                                                <div className="flex justify-between items-center text-[11px]">
                                                    <span className="text-white/70 font-medium">{DOF_LABELS[k]}</span>
                                                    <span className="font-mono text-white/40">{val}</span>
                                                </div>
                                                <div className="relative h-5 flex items-center group/slider">
                                                    {/* Custom Track */}
                                                    <div className="absolute left-0 right-0 h-1 bg-white/10 rounded-full overflow-hidden">
                                                        <div 
                                                            className="h-full bg-gradient-to-r from-blue-500 to-purple-500 shadow-[0_0_10px_rgba(0,122,255,0.5)]"
                                                            style={{ width: `${((val - min) / (max - min)) * 100}%` }} 
                                                        />
                                                    </div>
                                                    <input
                                                        type="range"
                                                        min={min}
                                                        max={max}
                                                        value={val}
                                                        onChange={(e) => onChange(k, parseInt(e.target.value))}
                                                        className="z-10 opacity-0 hover:opacity-100 group-hover/slider:opacity-100 transition-opacity"
                                                    />
                                                    {/* Visible Thumb (Pseudo) */}
                                                    <div 
                                                        className="absolute h-3.5 w-3.5 bg-white rounded-full shadow-lg pointer-events-none transition-transform duration-75"
                                                        style={{ 
                                                            left: `calc(${((val - min) / (max - min)) * 100}% - 7px)`
                                                        }}
                                                    />
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>

        {/* Dynamic Actions List (Micro-interactions) */}
         <div>
            <h3 className="text-[10px] font-semibold text-apple-subtext uppercase tracking-widest mb-3 px-2 flex items-center gap-1">
                <Zap size={10} /> Gestures
            </h3>
            <div className="flex flex-wrap gap-2">
                {Object.entries(DYNAMIC_ACTIONS).map(([name, preset]) => (
                    <button
                        key={name}
                        onClick={() => onApplyState({ ...state, ...preset })}
                        className="px-3 py-1.5 rounded-full bg-white/5 hover:bg-white/10 border border-white/5 text-[11px] text-white/70 hover:text-white transition-all active:scale-95 flex items-center gap-1.5"
                    >
                        <Play size={8} fill="currentColor" />
                        {name.split(":")[0]}
                    </button>
                ))}
            </div>
        </div>

      </div>
    </div>
  );
};
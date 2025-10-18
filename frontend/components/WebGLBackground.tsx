import React from 'react';
import { Theme } from '../types';

interface WebGLBackgroundProps {
  theme: Theme;
}

const WebGLBackground: React.FC<WebGLBackgroundProps> = ({ theme }) => {
  const baseBgColor = theme === Theme.Dark ? "bg-zinc-950" : "bg-zinc-100";
  
  const vignetteStyle = theme === Theme.Dark 
    ? { backgroundImage: 'radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,0.5) 100%)' }
    : { backgroundImage: 'radial-gradient(ellipse at center, transparent 60%, rgba(255,255,255,0.5) 100%)' };

  return (
    <>
      {/* Сплошной цвет фона (нижний слой) */}
      <div
        className={`fixed top-0 left-0 w-full h-full -z-20 transition-colors duration-500 ease-in-out ${baseBgColor}`}
      />
    </>
  );
};

export default WebGLBackground;
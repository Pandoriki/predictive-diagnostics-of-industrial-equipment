import React from 'react';
import { Theme } from '../types';

interface WebGLCardBackgroundProps {
  theme: Theme;
}

const WebGLCardBackground: React.FC<WebGLCardBackgroundProps> = ({ theme }) => {
  const backgroundClasses = theme === Theme.Dark
    ? "bg-gray-950"
    : "bg-white";

  return (
    <div
      className={`
        absolute inset-0 w-full h-full rounded-3xl -z-10
        transition-colors duration-500 ease-in-out
        ${backgroundClasses}
      `}
    />
  );
};

export default WebGLCardBackground;
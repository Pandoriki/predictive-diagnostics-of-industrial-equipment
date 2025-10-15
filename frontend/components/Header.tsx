
import React from 'react';
import { Theme, View } from '../types';
import { GearIcon, SunIcon, MoonIcon } from './icons';

interface HeaderProps {
  view: View;
  setView: (view: View) => void;
  theme: Theme;
  setTheme: (theme: Theme) => void;
  onLogoClick: () => void;
}

const Header: React.FC<HeaderProps> = ({ view, setView, theme, setTheme, onLogoClick }) => {
  const toggleTheme = () => {
    setTheme(theme === Theme.Light ? Theme.Dark : Theme.Light);
  };
  
  const accentColor = "cyan"; // Could be dynamic based on theme

  const GlassCapsule: React.FC<{children: React.ReactNode, onClick: () => void, isActive: boolean}> = ({ children, onClick, isActive }) => {
    return (
      <button
        onClick={onClick}
        className={`relative inline-flex items-center justify-center px-6 py-2 rounded-full text-sm font-medium
                   transition-all duration-300 ease-in-out overflow-hidden group
                   border border-white/20 dark:border-white/10
                   ${isActive
                     ? `text-slate-900 dark:text-white shadow-lg shadow-${accentColor}-500/40`
                     : 'text-slate-700 dark:text-gray-400 hover:text-slate-900 dark:hover:text-white'
                   }`}
      >
        <span className={`absolute inset-0 bg-gradient-to-br from-white/5 to-transparent backdrop-blur-lg
                          transition-all duration-300 group-hover:from-white/20
                          ${isActive ? `bg-${accentColor}-500/30` : ''}`}>
        </span>
        <span className="relative z-10">{children}</span>
        {isActive && (
          <span className={`absolute bottom-0 left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-${accentColor}-400 to-transparent animate-pulse`}></span>
        )}
      </button>
    );
  };

  return (
    <header className="relative z-10 flex items-center justify-between p-4 rounded-3xl bg-white/20 dark:bg-black/20 backdrop-blur-2xl border border-white/20 dark:border-white/10 shadow-2xl shadow-black/30">
      <div className="flex items-center space-x-4 cursor-pointer" onClick={onLogoClick}>
        <GearIcon className="h-8 w-8" />
        <h1 className="text-xl md:text-2xl font-bold text-slate-800 dark:text-white tracking-wider hidden sm:block">
            Предиктивная Диагностика: {view === View.List ? 'Обзор Устройств' : 'Панель Инженера'}
        </h1>
      </div>

      <div className="flex items-center space-x-2 md:space-x-4">
        <div className="flex items-center p-1 rounded-full bg-white/5 border border-white/10">
          <GlassCapsule onClick={() => setView(View.List)} isActive={view === View.List}>Список</GlassCapsule>
          <GlassCapsule onClick={() => {}} isActive={view === View.Engineer}>Инженер</GlassCapsule>
          <GlassCapsule onClick={() => setView(View.Guard)} isActive={view === View.Guard}>Охранник</GlassCapsule>
        </div>
        
        <button
            onClick={toggleTheme}
            className={`relative flex items-center justify-center w-12 h-12 rounded-full
                       transition-all duration-300 ease-in-out group
                       bg-gradient-to-br from-white/20 to-transparent backdrop-blur-lg
                       border border-white/20 dark:border-white/10 hover:shadow-lg hover:shadow-cyan-500/30
                       ${theme === Theme.Dark ? 'text-yellow-300' : 'text-blue-400'}`}
        >
          {theme === Theme.Dark ? <SunIcon className="w-6 h-6" /> : <MoonIcon className="w-6 h-6" />}
        </button>
      </div>
    </header>
  );
};

export default Header;

import React, { forwardRef } from 'react';
import { Theme, View } from '../types';
import { SunIcon, MoonIcon } from './icons';

interface HeaderProps {
  view: View;
  setView: (view: View) => void;
  theme: Theme;
  setTheme: (theme: Theme) => void;
  onLogoClick: () => void;
}

// <-- Оборачиваем компонент в forwardRef
const Header = forwardRef<HTMLDivElement, HeaderProps>(({ view, setView, theme, setTheme, onLogoClick }, ref) => {
  const toggleTheme = () => {
    setTheme(theme === Theme.Light ? Theme.Dark : Theme.Light);
  };

  const NavButton: React.FC<{children: React.ReactNode, onClick: () => void, isActive: boolean}> = ({ children, onClick, isActive }) => {
    const baseClasses = "px-5 py-2 rounded-full text-sm font-medium transition-colors duration-300 ease-in-out";
    let themeClasses;
    if (isActive) {
      themeClasses = theme === Theme.Dark ? 'bg-emerald-500 text-white' : 'bg-zinc-800 text-white';
    } else {
      themeClasses = theme === Theme.Dark ? 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700' : 'text-zinc-500 hover:bg-zinc-200';
    }
    return <button onClick={onClick} className={`${baseClasses} ${themeClasses}`}>{children}</button>;
  };

  return (
    // <-- Применяем полученный ref к основному элементу
    <header ref={ref} className={`flex items-center justify-between p-4 backdrop-blur-xl rounded-3xl border shadow-sm transition-colors duration-300 ease-in-out
                       ${theme === Theme.Dark ? 'bg-zinc-900/80 border-zinc-800' : 'bg-zinc-50/80 border-zinc-200'}`}
    >
      <div className="flex items-center space-x-4 cursor-pointer" onClick={onLogoClick}>
        <h1 className={`pl-3 text-lg md:text-xl font-bold hidden sm:block transition-colors duration-300
                       ${theme === Theme.Dark ? 'text-zinc-200' : 'text-zinc-800'}`}>
          Пандорики.Контроль
        </h1>
      </div>
      <div className="flex items-center space-x-2 md:space-x-4">
        <div className={`flex items-center p-1 rounded-full transition-colors duration-300
                        ${theme === Theme.Dark ? 'bg-zinc-950/50' : 'bg-zinc-200/70'}`}>
          <NavButton onClick={() => setView(View.List)} isActive={view === View.List}>Список</NavButton>
          <NavButton onClick={() => {}} isActive={view === View.Engineer}>Инженер</NavButton>
          <NavButton onClick={() => setView(View.Guard)} isActive={view === View.Guard}>Охранник</NavButton>
        </div>
        <button
            onClick={toggleTheme}
            className={`flex items-center justify-center w-11 h-11 rounded-full transition-colors duration-300
                       ${theme === Theme.Dark ? 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700' : 'bg-zinc-200/70 text-zinc-500 hover:bg-zinc-300'}`}>
          {theme === Theme.Dark ? <SunIcon className="w-5 h-5" /> : <MoonIcon className="w-5 h-5" />}
        </button>
      </div>
    </header>
  );
});

export default Header;
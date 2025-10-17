// src/App.tsx (или ваш главный файл)

import React, { useState, useEffect, useRef } from 'react';
import { Theme, View } from './types';

// Импортируем все компоненты
import Header from './components/Header';
import EquipmentListView from './components/EquipmentListView';
import GuardDashboardView from './components/GuardDashboardView'; // <-- Не забудьте импортировать
import WebGLBackground from './components/WebGLBackground';

function App() {
  const [theme, setTheme] = useState<Theme>(Theme.Dark);
  const [view, setView] = useState<View>(View.List);
  
  const headerRef = useRef<HTMLDivElement>(null);
  const [isScrollButtonVisible, setScrollButtonVisible] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      if (headerRef.current) {
        const isHeaderHidden = headerRef.current.getBoundingClientRect().bottom < 0;
        setScrollButtonVisible(isHeaderHidden);
      }
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Функция для обработки клика по оборудованию (может быть разной для разных панелей)
  const handleSelectEquipment = (equipment: any) => {
    console.log("Выбрано оборудование:", equipment.id);
    // Здесь может быть логика открытия модального окна и т.д.
  };

  return (
    <>
      <WebGLBackground theme={theme} />
      <div className="container mx-auto p-4 md:p-6 space-y-6">
        <Header 
          ref={headerRef}
          view={view} 
          setView={setView} 
          theme={theme} 
          setTheme={setTheme} 
          onLogoClick={() => setView(View.List)} 
        />
        <main>
          {/* --- ВОТ КЛЮЧЕВАЯ ЛОГИКА ПЕРЕКЛЮЧЕНИЯ --- */}
          {view === View.List && (
            <EquipmentListView 
              theme={theme}
              onSelectEquipment={handleSelectEquipment}
              isScrollButtonVisible={isScrollButtonVisible}
            />
          )}

          {view === View.Guard && (
            <GuardDashboardView 
              theme={theme}
              onSelectEquipment={handleSelectEquipment}
            />
          )}

          {/* Можно добавить и для инженера, когда он будет готов */}
          {/* {view === View.Engineer && <EngineerDashboardView />} */}
        </main>
      </div>
    </>
  );
}

export default App;
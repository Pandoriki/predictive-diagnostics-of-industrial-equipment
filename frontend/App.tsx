import React, { useState, useEffect } from 'react';
import { Theme, View, Equipment } from './types';
import { mockEquipmentData } from './constants';
import Header from './components/Header';
import EquipmentListView from './components/EquipmentListView';
import EngineerDashboardView from './components/EngineerDashboardView';
import GuardDashboardView from './components/GuardDashboardView';
import WebGLBackground from './components/WebGLBackground';

const App: React.FC = () => {
  const [theme, setTheme] = useState<Theme>(Theme.Dark);
  const [view, setView] = useState<View>(View.List);
  const [selectedEquipment, setSelectedEquipment] = useState<Equipment | null>(null);

  useEffect(() => {
    if (theme === Theme.Dark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  const handleSelectEquipment = (equipment: Equipment) => {
    setSelectedEquipment(equipment);
    setView(View.Engineer);
  };

  const handleReturnToList = () => {
    setSelectedEquipment(null);
    setView(View.List);
  };
    
  return (
    <div className="min-h-screen w-full transition-colors duration-500">
      <WebGLBackground theme={theme} />
      <div className="min-h-screen w-full bg-gradient-to-br dark:from-black/60 dark:to-black/80 from-white/10 to-white/30 backdrop-blur-sm">
        <main className="container mx-auto px-4 py-6 text-slate-800 dark:text-gray-100">
          {view !== View.Guard && (
            <Header 
              view={view} 
              setView={setView} 
              theme={theme} 
              setTheme={setTheme}
              onLogoClick={handleReturnToList}
            />
          )}

          <div className={view !== View.Guard ? "mt-8" : ""}>
            {view === View.List && (
              <EquipmentListView 
                equipmentData={mockEquipmentData} 
                onSelectEquipment={handleSelectEquipment} 
                theme={theme}
              />
            )}
            {view === View.Engineer && selectedEquipment && <EngineerDashboardView equipment={selectedEquipment} theme={theme} />}
            {view === View.Guard && <GuardDashboardView equipmentData={mockEquipmentData} onSelectEquipment={handleSelectEquipment} />}
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;
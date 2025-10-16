import React, { useState, useMemo, useEffect, useRef } from 'react';
import { Equipment, Status, Theme } from '../types';
import { CheckCircleIcon, AlertTriangleIcon, BellIcon, SortAscIcon, SortDescIcon } from './icons';
import WebGLCardBackground from './WebGLCardBackground';

interface EquipmentListViewProps {
  equipmentData: Equipment[];
  onSelectEquipment: (equipment: Equipment) => void;
  theme: Theme;
}

const statusConfig = {
  [Status.Normal]: { icon: CheckCircleIcon, color: 'text-green-400', ring: 'ring-green-500/50' },
  [Status.Warning]: { icon: BellIcon, color: 'text-yellow-400', ring: 'ring-yellow-500/50' },
  [Status.Critical]: { icon: AlertTriangleIcon, color: 'text-red-500', ring: 'ring-red-500/50' },
};

const EquipmentCard: React.FC<{ equipment: Equipment; onSelect: () => void; theme: Theme; isVisible: boolean; delay: number }> = ({ equipment, onSelect, theme, isVisible, delay }) => {
  const { icon: Icon, color } = statusConfig[equipment.status];
  const cardRef = useRef<HTMLDivElement>(null);
  
  return (
    <div 
      ref={cardRef}
      onClick={onSelect}
      className="relative group p-5 rounded-3xl overflow-hidden cursor-pointer transition-all duration-500 ease-in-out"
      style={{
        transitionDelay: `${delay}ms`,
        transform: isVisible ? 'translateY(0)' : 'translateY(20px)',
        opacity: isVisible ? 1 : 0,
      }}
    >
      <WebGLCardBackground theme={theme} parentRef={cardRef} />
      
      <div className="absolute inset-0 bg-transparent border border-black/10 dark:border-white/10 rounded-3xl transition-all duration-300 pointer-events-none"></div>
      
      <div className="relative z-10 flex flex-col h-full">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <Icon className={`w-8 h-8 ${color}`} />
            <h3 className="text-xl font-bold text-slate-800 dark:text-white">{equipment.id} <span className="font-normal text-slate-600 dark:text-gray-300">{equipment.name}</span></h3>
          </div>
          <div className={`px-3 py-1 text-xs font-semibold rounded-full bg-opacity-20 ${color.replace('text-', 'bg-')} ${color}`}>
            {equipment.status}
          </div>
        </div>
        <div className="mt-6 flex-grow space-y-3 text-lg">
          <p className="text-slate-700 dark:text-gray-200">
            Прогноз RUL: <span className="font-bold text-slate-800 dark:text-white">{equipment.rul} {equipment.rulUnit}</span>
          </p>
          <p className="text-slate-500 dark:text-gray-400 text-sm">
            Тип: <span className="font-medium text-slate-600 dark:text-gray-300">{equipment.type}</span>
          </p>
        </div>
      </div>
    </div>
  );
};

const GlassPanel: React.FC<{children: React.ReactNode, className?: string}> = ({ children, className }) => (
    <div className={`relative p-4 rounded-3xl bg-white/20 dark:bg-black/20 backdrop-blur-lg border border-black/10 dark:border-white/10 shadow-2xl shadow-black/30 ${className}`}>
        {children}
    </div>
);

const FilterButton: React.FC<{ children: React.ReactNode; onClick: () => void; isActive: boolean }> = ({ children, onClick, isActive }) => {
    return (
        <button 
            onClick={onClick}
            className={`px-4 py-2 text-sm font-medium rounded-full transition-all duration-300 ease-in-out border
                        ${isActive 
                            ? 'bg-cyan-500/20 border-cyan-400 text-cyan-800 dark:text-cyan-300 shadow-md shadow-cyan-500/20'
                            : 'bg-black/5 dark:bg-white/5 border-black/10 dark:border-white/10 text-slate-700 dark:text-gray-300 hover:bg-black/10 dark:hover:bg-white/10 hover:border-black/20 dark:hover:border-white/20'
                        }`}
        >
            {children}
        </button>
    );
};


const EquipmentListView: React.FC<EquipmentListViewProps> = ({ equipmentData, onSelectEquipment, theme }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<Status | 'Все'>('Все');
  const [sortConfig, setSortConfig] = useState<{ key: 'rul' | 'id'; direction: 'asc' | 'desc' }>({ key: 'rul', direction: 'asc' });
  const [isListVisible, setIsListVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsListVisible(true), 100);
    return () => clearTimeout(timer);
  }, []);

  const filteredAndSortedData = useMemo(() => {
    let data = [...equipmentData];
    
    if (statusFilter !== 'Все') {
      data = data.filter(e => e.status === statusFilter);
    }

    if (searchTerm) {
      data = data.filter(e => 
        e.id.toLowerCase().includes(searchTerm.toLowerCase()) || 
        e.name.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    data.sort((a, b) => {
      let comparison = 0;
      if (sortConfig.key === 'id') {
        comparison = a.id.localeCompare(b.id);
      } else { // RUL
        comparison = a.rul - b.rul;
      }
      return sortConfig.direction === 'asc' ? comparison : -comparison;
    });

    return data;
  }, [equipmentData, searchTerm, statusFilter, sortConfig]);

  const handleSort = (key: 'rul' | 'id') => {
    setSortConfig(prev => {
        if (prev.key === key) {
            return { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
        }
        return { key, direction: 'asc' };
    });
  };

  return (
    <div className="space-y-6">
        <GlassPanel>
            <div className="flex flex-wrap items-center gap-4">
                <input
                    type="text"
                    placeholder="Поиск по ID или Названию..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="flex-grow w-full md:w-auto bg-transparent px-4 py-2 rounded-full border border-black/20 dark:border-white/20 focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500 outline-none transition-all duration-300 text-slate-800 dark:text-white placeholder-slate-500 dark:placeholder-gray-400"
                />
                <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm text-slate-600 dark:text-gray-400 mr-2">Фильтр:</span>
                    <FilterButton onClick={() => setStatusFilter('Все')} isActive={statusFilter === 'Все'}>Все</FilterButton>
                    <FilterButton onClick={() => setStatusFilter(Status.Normal)} isActive={statusFilter === Status.Normal}>Нормально</FilterButton>
                    <FilterButton onClick={() => setStatusFilter(Status.Warning)} isActive={statusFilter === Status.Warning}>Предупреждение</FilterButton>
                    <FilterButton onClick={() => setStatusFilter(Status.Critical)} isActive={statusFilter === Status.Critical}>Критическое</FilterButton>
                </div>
                 <div className="flex items-center gap-2">
                    <span className="text-sm text-slate-600 dark:text-gray-400 mr-2">Сортировка:</span>
                    <FilterButton onClick={() => handleSort('rul')} isActive={sortConfig.key === 'rul'}>
                        <span className="flex items-center gap-1">RUL {sortConfig.key === 'rul' && (sortConfig.direction === 'asc' ? <SortAscIcon /> : <SortDescIcon />)}</span>
                    </FilterButton>
                    <FilterButton onClick={() => handleSort('id')} isActive={sortConfig.key === 'id'}>
                        <span className="flex items-center gap-1">ID {sortConfig.key === 'id' && (sortConfig.direction === 'asc' ? <SortAscIcon /> : <SortDescIcon />)}</span>
                    </FilterButton>
                </div>
            </div>
        </GlassPanel>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {filteredAndSortedData.map((eq, index) => (
                <EquipmentCard 
                    key={eq.id} 
                    equipment={eq} 
                    onSelect={() => onSelectEquipment(eq)} 
                    theme={theme}
                    isVisible={isListVisible}
                    delay={index * 50}
                />
            ))}
        </div>
    </div>
  );
};

export default EquipmentListView;
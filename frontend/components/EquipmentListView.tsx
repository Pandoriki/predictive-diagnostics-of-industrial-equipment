import React, { useState, useMemo, useEffect, useRef } from 'react';
import { Equipment, Status, Theme, BackendStatusSummary } from '../types'; 
import { fetchStatusSummary } from '../api/apiService';
import { CheckCircleIcon, AlertTriangleIcon, BellIcon, SortAscIcon, SortDescIcon, ArrowUpIcon } from './icons';

// --- КОНФИГУРАЦИЯ И ХЕЛПЕРЫ ---

const statusConfig = {
  [Status.Normal]: { icon: CheckCircleIcon, color: 'text-green-400', bg: 'bg-green-500' },
  [Status.Warning]: { icon: BellIcon, color: 'text-yellow-400', bg: 'bg-yellow-500' },
  [Status.Critical]: { icon: AlertTriangleIcon, color: 'text-red-400', bg: 'bg-red-500' },
};

const adaptApiDataToEquipment = (apiData: BackendStatusSummary[]): Equipment[] => {
  const statusCodeToEnum: Record<string, Status> = { 'normal': Status.Normal, 'warning': Status.Warning, 'critical': Status.Critical };
  return apiData.map(item => ({ id: `#${item.unit_id}`, status: statusCodeToEnum[item.status_code] ?? Status.Normal, rul: item.current_rul, last_updated: item.last_updated, name: `Установка`, rulUnit: 'циклов', type: 'Газотурбинный двигатель', model: 'N/A', lastWarning: item.status_code !== 'normal' ? 'Требуется внимание' : 'Нет', degradationReason: 'Подробная информация доступна при выборе оборудования.', rulHistory: [], sensors: {}, }));
};


// --- ДОЧЕРНИЕ UI КОМПОНЕНТЫ ---

const EquipmentCard: React.FC<{ equipment: Equipment; onSelect: () => void; theme: Theme; isVisible: boolean; delay: number }> = ({ equipment, onSelect, theme, isVisible, delay }) => {
  const { icon: Icon, color, bg } = statusConfig[equipment.status];
  return (
    <div 
      onClick={onSelect}
      className={`relative group p-6 rounded-3xl cursor-pointer transition-[transform,box-shadow] duration-300 ease-out hover:shadow-xl hover:-translate-y-1.5 border overflow-hidden
                  ${theme === Theme.Dark ? 'bg-gradient-to-br from-zinc-900 to-zinc-800 border-zinc-800' : 'bg-gradient-to-br from-zinc-50 to-zinc-100 border-zinc-200/80'}`}
      style={{ transitionDelay: `${delay}ms`, transform: isVisible ? 'translateY(0)' : 'translateY(20px)', opacity: isVisible ? 1 : 0 }}
    >
      <div className={`absolute inset-0 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-300
                     ${theme === Theme.Dark ? 'bg-[radial-gradient(ellipse_at_top_left,_rgba(255,255,255,0.05)_0%,_rgba(255,255,255,0)_50%)]' : 'bg-[radial-gradient(ellipse_at_top_left,_rgba(0,0,0,0.03)_0%,_rgba(0,0,0,0)_50%)]'}`}
      />
      <div className="relative z-10 flex flex-col h-full"><div className="flex items-start justify-between"><div className="flex items-center space-x-4"><Icon className={`w-8 h-8 ${color}`} /><div><h3 className={`text-xl font-bold transition-colors duration-300 ${theme === Theme.Dark ? 'text-zinc-200' : 'text-zinc-900'}`}>{equipment.id}</h3><p className={`text-sm transition-colors duration-300 ${theme === Theme.Dark ? 'text-zinc-400' : 'text-zinc-500'}`}>{equipment.name}</p></div></div><div className={`flex items-center space-x-2 text-xs font-semibold capitalize transition-colors duration-300 ${theme === Theme.Dark ? 'text-zinc-300' : 'text-zinc-600'}`}><span className={`w-2.5 h-2.5 rounded-full ${bg}`}></span><span>{equipment.status}</span></div></div><div className="mt-8 flex-grow space-y-4"><p className={`transition-colors duration-300 ${theme === Theme.Dark ? 'text-zinc-300' : 'text-zinc-600'}`}>Прогноз RUL: <span className={`font-bold text-2xl transition-colors duration-300 ${theme === Theme.Dark ? 'text-zinc-100' : 'text-zinc-900'}`}>{Math.floor(equipment.rul)}</span><span className={`ml-1.5 text-sm transition-colors duration-300 ${theme === Theme.Dark ? 'text-zinc-400' : 'text-zinc-500'}`}>{equipment.rulUnit}</span></p><p className={`text-sm transition-colors duration-300 ${theme === Theme.Dark ? 'text-zinc-400' : 'text-zinc-500'}`}>Тип: <span className={`font-medium transition-colors duration-300 ${theme === Theme.Dark ? 'text-zinc-200' : 'text-zinc-700'}`}>{equipment.type}</span></p></div></div>
    </div>
  );
};
const ControlPanel: React.FC<{children: React.ReactNode, className?: string, theme: Theme}> = ({ children, className, theme }) => (
    <div className={`p-4 rounded-3xl backdrop-blur-xl border shadow-sm transition-colors duration-300 ease-in-out ${className} 
                   ${theme === Theme.Dark ? 'bg-zinc-900/80 border-zinc-800' : 'bg-zinc-50/80 border-zinc-200'}`}>
        {children}
    </div>
);

type SmartFilterValue = 'Все' | 'Проблемы' | Status.Warning | Status.Critical;

const SmartFilter: React.FC<{
  theme: Theme;
  activeFilter: SmartFilterValue;
  setFilter: (filter: SmartFilterValue) => void;
  counts: { [key: string]: number };
}> = ({ theme, activeFilter, setFilter, counts }) => {
  const issuesCount = counts[Status.Warning] + counts[Status.Critical];
  const isIssuesActive = ['Проблемы', Status.Warning, Status.Critical].includes(activeFilter);
  const mainButtonStyle = (isActive: boolean) => `px-4 py-2 text-sm font-semibold rounded-full transition-all duration-300 flex items-center gap-2 ${isActive ? (theme === Theme.Dark ? 'bg-emerald-500 text-white' : 'bg-zinc-800 text-white') : (theme === Theme.Dark ? 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700' : 'bg-zinc-200 text-zinc-600 hover:bg-zinc-300')}`;
  const secondaryButtonStyle = (isActive: boolean) => `px-3 py-1 text-xs font-medium rounded-full transition-colors duration-200 ${isActive ? (theme === Theme.Dark ? 'bg-zinc-600 text-white' : 'bg-zinc-500 text-white') : (theme === Theme.Dark ? 'text-zinc-400 hover:text-white' : 'text-zinc-500 hover:text-black')}`;

  return (
    <div className="flex flex-col items-start gap-2">
      <div className="flex items-center gap-2">
        <button className={mainButtonStyle(activeFilter === 'Все')} onClick={() => setFilter('Все')}>
          Все <span className="text-xs opacity-70">{counts['Все']}</span>
        </button>
        <button 
          className={`${mainButtonStyle(isIssuesActive)} ${issuesCount === 0 && !isIssuesActive ? 'opacity-50 !bg-transparent border border-zinc-700' : ''}`}
          onClick={() => setFilter('Проблемы')}
        >
          <AlertTriangleIcon className={`w-4 h-4 ${issuesCount > 0 ? 'text-yellow-400' : ''}`} />
          Требуют внимания <span className="text-xs opacity-70">{issuesCount}</span>
        </button>
      </div>
      {isIssuesActive && (
        <div className="flex items-center gap-3 pl-4 pt-2 border-l-2 border-zinc-700 ml-4">
          <button className={secondaryButtonStyle(activeFilter === 'Проблемы')} onClick={() => setFilter('Проблемы')}>
            Все проблемы <span className="opacity-70">{issuesCount}</span>
          </button>
          <button className={secondaryButtonStyle(activeFilter === Status.Warning)} onClick={() => setFilter(Status.Warning)}>
            Предупреждение <span className="opacity-70">{counts[Status.Warning]}</span>
          </button>
          <button className={secondaryButtonStyle(activeFilter === Status.Critical)} onClick={() => setFilter(Status.Critical)}>
            Критическое <span className="opacity-70">{counts[Status.Critical]}</span>
          </button>
        </div>
      )}
    </div>
  );
};

const SortButton: React.FC<{ children: React.ReactNode; onClick: () => void; theme: Theme }> = ({ children, onClick, theme }) => {
  const baseClasses = "px-4 py-2 text-sm font-medium rounded-full transition-colors duration-300 ease-in-out flex items-center gap-2";
  const themeClasses = theme === Theme.Dark ? 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700' : 'bg-zinc-200/70 text-zinc-600 hover:bg-zinc-300';
  return <button onClick={onClick} className={`${baseClasses} ${themeClasses}`}>{children}</button>;
};

const ScrollToTopButton: React.FC<{ isVisible: boolean; onClick: () => void; theme: Theme }> = ({ isVisible, onClick, theme }) => {
  return (
    <button onClick={onClick} className={`fixed bottom-6 right-6 z-50 w-12 h-12 rounded-full flex items-center justify-center shadow-lg transition-all duration-300 ease-in-out
                  ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-5 pointer-events-none'}
                  ${theme === Theme.Dark ? 'bg-zinc-800 text-zinc-300 hover:bg-emerald-500 hover:text-white' : 'bg-zinc-800 text-zinc-200 hover:bg-emerald-600'}`}>
      <ArrowUpIcon className="w-6 h-6" />
    </button>
  );
};

// --- ОСНОВНОВНОЙ КОМПОНЕНТ ---
const EquipmentListView: React.FC<EquipmentListViewProps> = ({ onSelectEquipment, theme, isScrollButtonVisible }) => {
  const [equipmentData, setEquipmentData] = useState<Equipment[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<SmartFilterValue>('Все');
  const [sortConfig, setSortConfig] = useState<{ key: 'rul' | 'id'; direction: 'asc' | 'desc' }>({ key: 'rul', direction: 'asc' });
  const [isListVisible, setIsListVisible] = useState(false);
    
  const handleScrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };
  
  useEffect(() => {
    const loadData = async () => {
      try { setLoading(true); setError(null); const apiData = await fetchStatusSummary(); const adaptedData = adaptApiDataToEquipment(apiData); setEquipmentData(adaptedData);
      } catch (err) { console.error(err); setError('Не удалось загрузить данные с сервера.'); } finally { setLoading(false); }
    };
    loadData();
  }, []);

  useEffect(() => { if (!loading) { setTimeout(() => setIsListVisible(true), 100); } }, [loading]);

  const statusCounts = useMemo(() => {
    const counts = { 'Все': equipmentData.length, [Status.Normal]: 0, [Status.Warning]: 0, [Status.Critical]: 0 };
    equipmentData.forEach(eq => { counts[eq.status]++; });
    return counts;
  }, [equipmentData]);

  const filteredAndSortedData = useMemo(() => {
    let data = [...equipmentData];
    if (statusFilter === 'Проблемы') {
      data = data.filter(e => e.status === Status.Warning || e.status === Status.Critical);
    } else if (statusFilter !== 'Все') {
      data = data.filter(e => e.status === statusFilter);
    }
    if (searchTerm) { data = data.filter(e => e.id.toLowerCase().includes(searchTerm.toLowerCase()) || e.name.toLowerCase().includes(searchTerm.toLowerCase())); }
    data.sort((a, b) => {
      let comparison = sortConfig.key === 'id' ? a.id.localeCompare(b.id) : a.rul - b.rul;
      return sortConfig.direction === 'asc' ? comparison : -comparison;
    });
    return data;
  }, [equipmentData, searchTerm, statusFilter, sortConfig]);

  const handleSort = (key: 'rul' | 'id') => { setSortConfig(prev => ({ key, direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc' })); };

  if (loading) { return <div className={`text-center p-10 text-lg transition-colors duration-300 ${theme === Theme.Dark ? 'text-zinc-400' : 'text-zinc-500'}`}>Загрузка данных...</div>; }
  if (error) { return <div className="text-center p-10 text-lg text-red-500">{error}</div>; }

  return (
    <div className="space-y-6">
        <ControlPanel theme={theme}>
            <div className="flex flex-col gap-4">
                <input
                    type="text" placeholder="Поиск по ID или названию..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)}
                    className={`w-full px-5 py-2.5 rounded-full border border-transparent focus:ring-2 outline-none transition-all duration-300 placeholder-zinc-400
                                ${theme === Theme.Dark ? 'bg-zinc-800 text-zinc-200 focus:ring-emerald-500' : 'bg-zinc-100 text-zinc-800 focus:ring-emerald-600'}`}
                />
                <div className="flex items-start justify-between">
                    <SmartFilter theme={theme} activeFilter={statusFilter} setFilter={setStatusFilter} counts={statusCounts} />
                    <div className="flex items-center gap-2">
                        <SortButton theme={theme} onClick={() => handleSort('rul')}>
                            RUL
                            {sortConfig.key === 'rul' && (sortConfig.direction === 'asc' ? <SortAscIcon /> : <SortDescIcon />)}
                        </SortButton>
                        <SortButton theme={theme} onClick={() => handleSort('id')}>
                            ID
                            {sortConfig.key === 'id' && (sortConfig.direction === 'asc' ? <SortAscIcon /> : <SortDescIcon />)}
                        </SortButton>
                    </div>
                </div>
            </div>
        </ControlPanel>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {filteredAndSortedData.map((eq, index) => (
                <EquipmentCard key={eq.id} equipment={eq} onSelect={() => onSelectEquipment(eq)} theme={theme} isVisible={isListVisible} delay={index * 50}/>
            ))}
        </div>
        
        <ScrollToTopButton 
          isVisible={isScrollButtonVisible}
          onClick={handleScrollToTop}
          theme={theme}
        />
    </div>
  );
};

export default EquipmentListView;
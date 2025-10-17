// frontend/components/GuardDashboardView.tsx

import React, { useState, useEffect, useMemo } from 'react';
import { Equipment, Status, Theme, BackendStatusSummary } from '../types';
import { fetchStatusSummary } from '../api/apiService';
import { CheckCircleIcon, AlertTriangleIcon, WrenchIcon } from './icons';

interface GuardDashboardViewProps {
    onSelectEquipment: (equipment: Equipment) => void;
    theme: Theme;
}

// --- КОНФИГУРАЦИЯ И ХЕЛПЕРЫ ---

const statusConfig = {
  [Status.Critical]: { 
    icon: AlertTriangleIcon, label: 'КРИТИЧЕСКОЕ', barColor: 'bg-red-500', textColor: 'text-red-400', priority: 3
  },
  [Status.Warning]: { 
    icon: WrenchIcon, label: 'ПРЕДУПРЕЖДЕНИЕ', barColor: 'bg-yellow-500', textColor: 'text-yellow-400', priority: 2
  },
  [Status.Normal]: { 
    icon: CheckCircleIcon, label: 'НОРМА', barColor: 'bg-green-500', textColor: 'text-green-400', priority: 1
  },
};

const adaptApiDataToEquipment = (apiData: BackendStatusSummary[]): Equipment[] => {
  const statusCodeToEnum: Record<string, Status> = { 'normal': Status.Normal, 'warning': Status.Warning, 'critical': Status.Critical };
  return apiData.map(item => ({ id: `#${item.unit_id}`, status: statusCodeToEnum[item.status_code] ?? Status.Normal, rul: item.current_rul, last_updated: item.last_updated, name: `Установка`, rulUnit: 'циклов', type: 'Газотурбинный двигатель', model: 'N/A', lastWarning: item.status_code !== 'normal' ? 'Требуется внимание' : 'Нет', degradationReason: 'Подробная информация доступна при выборе оборудования.', rulHistory: [], sensors: {}, }));
};

// 1. ИСПРАВЛЕННАЯ ФУНКЦИЯ ДЛЯ КОРРЕКТНОГО ОТОБРАЖЕНИЯ ВРЕМЕНИ
const formatDateSafe = (timeString: string | undefined): string => {
    if (!timeString || !/^\d{2}:\d{2}:\d{2}$/.test(timeString)) {
        return '——:——:——';
    }
    try {
        // Получаем сегодняшнюю дату в UTC
        const today = new Date();
        const year = today.getUTCFullYear();
        const month = String(today.getUTCMonth() + 1).padStart(2, '0');
        const day = String(today.getUTCDate()).padStart(2, '0');

        // Создаем полную строку даты и времени в формате ISO 8601 UTC
        const utcDateString = `${year}-${month}-${day}T${timeString}Z`;
        
        // Создаем объект Date, который теперь точно знает, что это время по UTC
        const dateInUtc = new Date(utcDateString);

        // Преобразуем в локальное время пользователя
        return dateInUtc.toLocaleTimeString('ru-RU');

    } catch (error) {
        console.error("Ошибка форматирования времени:", error);
        return '——:——:——';
    }
};

const useClock = () => {
    const [time, setTime] = useState(new Date());
    useEffect(() => {
        const timerId = setInterval(() => setTime(new Date()), 1000);
        return () => clearInterval(timerId);
    }, []);
    return time;
};


// --- ДОЧЕРНИЕ UI КОМПОНЕНТЫ ---

const GuardStatusCard: React.FC<{ equipment: Equipment; onSelect: () => void; justUpdated: boolean; theme: Theme }> = ({ equipment, onSelect, justUpdated, theme }) => {
    const { icon: Icon, label, barColor, textColor } = statusConfig[equipment.status];
    const isCritical = equipment.status === Status.Critical;
    const lastUpdatedTime = useMemo(() => formatDateSafe(equipment.last_updated), [equipment.last_updated]);

    return (
        <div 
            onClick={onSelect}
            className={`relative group p-4 rounded-2xl cursor-pointer border overflow-hidden transition-[transform,box-shadow] duration-300 ease-in-out hover:-translate-y-1
                       ${theme === Theme.Dark
                         ? 'bg-zinc-900 border-zinc-800 hover:border-zinc-700'
                         : 'bg-white border-zinc-200 hover:border-zinc-300'
                       }
                       ${justUpdated 
                         ? 'shadow-lg shadow-emerald-500/20'
                         : 'shadow-md shadow-black/10'
                       }
                       ${isCritical ? 'critical-glow' : ''}
            `}
        >
            <div className={`absolute top-0 left-0 h-full w-1.5 ${barColor}`} />
            <div className="ml-4 flex flex-col h-full">
                <div className={`flex justify-between items-start text-xs mb-2 ${theme === Theme.Dark ? 'text-zinc-500' : 'text-zinc-400'}`}>
                    <span className="font-mono">ПОСЛЕДНЕЕ ОБНОВЛЕНИЕ</span>
                    <span className="font-mono">{lastUpdatedTime}</span>
                </div>
                <div className="flex-grow">
                    <h3 className={`text-xl font-bold ${theme === Theme.Dark ? 'text-zinc-300' : 'text-zinc-800'}`}>{`Оборудование ${equipment.id}`}</h3>
                    <p className={`text-sm ${theme === Theme.Dark ? 'text-zinc-400' : 'text-zinc-500'}`}>
                        Прогноз RUL: <span className={`font-semibold ${theme === Theme.Dark ? 'text-zinc-300' : 'text-zinc-700'}`}>{Math.floor(equipment.rul)}</span> циклов
                    </p>
                </div>
                <div className={`flex items-center gap-2 mt-3 pt-3 border-t ${theme === Theme.Dark ? 'border-zinc-800' : 'border-zinc-200'}`}>
                    <Icon className={`w-5 h-5 ${textColor}`} />
                    <p className={`text-base font-bold tracking-wider ${textColor}`}>{label}</p>
                </div>
            </div>
        </div>
    );
};


// --- ОСНОВНОЙ КОМПОНЕНТ ---

const GuardDashboardView: React.FC<GuardDashboardViewProps> = ({ onSelectEquipment, theme }) => {
    const time = useClock();
    const [equipmentData, setEquipmentData] = useState<Equipment[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [updatedIds, setUpdatedIds] = useState<Set<string>>(new Set());

    useEffect(() => {
        const fetchData = async () => {
            try {
                const apiData = await fetchStatusSummary();
                const adaptedData = adaptApiDataToEquipment(apiData);
                
                setEquipmentData(prevData => {
                    const newUpdatedIds = new Set<string>();
                    adaptedData.forEach(newItem => {
                        const oldItem = prevData.find(item => item.id === newItem.id);
                        if (!oldItem || oldItem.last_updated !== newItem.last_updated) {
                            newUpdatedIds.add(newItem.id);
                        }
                    });
                    setUpdatedIds(newUpdatedIds);
                    setTimeout(() => setUpdatedIds(new Set()), 1500);
                    return adaptedData;
                });

            } catch (error) {
                console.error("Ошибка при обновлении данных:", error);
            } finally {
                if (loading) setLoading(false);
            }
        };

        fetchData();
        const intervalId = setInterval(fetchData, 10000);
        return () => clearInterval(intervalId);
    }, [loading]);

    const sortedData = useMemo(() => {
        return [...equipmentData].sort((a, b) => statusConfig[b.status].priority - statusConfig[a.status].priority);
    }, [equipmentData]);

    if (loading) {
        return <div className={`text-center p-10 text-lg ${theme === Theme.Dark ? 'text-zinc-400' : 'text-zinc-500'}`}>Загрузка данных панели охраны...</div>;
    }

    return (
        <div className={`fixed inset-0 w-full h-full flex flex-col p-4 sm:p-6 transition-colors duration-300 
                       ${theme === Theme.Dark ? 'bg-zinc-950 text-zinc-200' : 'bg-zinc-100 text-zinc-800'}`}
        >
            <header className={`flex-shrink-0 items-center px-6 py-3 rounded-2xl backdrop-blur-lg border transition-colors duration-300
                             ${theme === Theme.Dark ? 'bg-zinc-900/80 border-zinc-800' : 'bg-white/80 border-zinc-200'}`}>
                {/* <h1 className={`text-xl sm:text-2xl md:text-3xl font-bold tracking-wider ${theme === Theme.Dark ? 'text-zinc-300' : 'text-zinc-700'}`}>
                    Диагностическая панель
                </h1> */}
                <div className="text-center">
                    <p className={`text-xl sm:text-2xl font-bold ${theme === Theme.Dark ? 'text-zinc-200' : 'text-zinc-800'}`}>{time.toLocaleTimeString('ru-RU')}</p>
                    <p className={`text-xs sm:text-sm ${theme === Theme.Dark ? 'text-zinc-400' : 'text-zinc-500'}`}>{time.toLocaleDateString('ru-RU', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</p>
                </div>
            </header>
            <main className="flex-grow mt-6 overflow-y-auto hide-scrollbar">
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-5">
                    {sortedData.map(eq => (
                        <GuardStatusCard 
                          key={eq.id} 
                          equipment={eq} 
                          onSelect={() => onSelectEquipment(eq)}
                          justUpdated={updatedIds.has(eq.id)}
                          theme={theme}
                        />
                    ))}
                </div>
            </main>
        </div>
    );
};

export default GuardDashboardView;
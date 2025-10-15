
import React, { useState, useEffect, useMemo } from 'react';
import { Equipment, Status } from '../types';
import { GearIcon, CheckCircleIcon, AlertTriangleIcon, WrenchIcon } from './icons';

interface GuardDashboardViewProps {
    equipmentData: Equipment[];
    onSelectEquipment: (equipment: Equipment) => void;
}

const statusConfig = {
  [Status.Critical]: { 
    icon: AlertTriangleIcon, 
    color: 'red-500', 
    glow: 'shadow-red-500/60', 
    pulseClass: 'animate-pulse',
    priority: 3
  },
  [Status.Warning]: { 
    icon: WrenchIcon, 
    color: 'yellow-400', 
    glow: 'shadow-yellow-400/50', 
    pulseClass: 'animate-[pulse_2s_cubic-bezier(0.4,0,0.6,1)_infinite]',
    priority: 2
  },
  [Status.Normal]: { 
    icon: CheckCircleIcon, 
    color: 'green-400', 
    glow: 'shadow-green-500/40', 
    pulseClass: '',
    priority: 1
  },
};

const useClock = () => {
    const [time, setTime] = useState(new Date());

    useEffect(() => {
        const timerId = setInterval(() => setTime(new Date()), 1000);
        return () => clearInterval(timerId);
    }, []);

    return time;
};

const GuardStatusCard: React.FC<{ equipment: Equipment; onSelect: () => void }> = ({ equipment, onSelect }) => {
    const { icon: Icon, color, glow, pulseClass } = statusConfig[equipment.status];
    
    return (
        <div 
            onClick={onSelect}
            className={`relative p-6 rounded-3xl overflow-hidden cursor-pointer
                        bg-white/30 dark:bg-black/40 backdrop-blur-xl border border-black/10 dark:border-white/10
                        transition-all duration-500 ease-in-out hover:border-cyan-400/80
                        flex flex-col items-center justify-center text-center shadow-2xl ${glow} ${pulseClass}`}
        >
            <Icon className={`w-20 h-20 lg:w-24 lg:h-24 text-${color} mb-4`} />
            <h3 className="text-xl lg:text-2xl font-bold text-slate-800 dark:text-white tracking-wide">
                ОБОРУДОВАНИЕ {equipment.id}
            </h3>
            <p className={`text-lg lg:text-xl font-semibold text-${color}`}>
                {equipment.status} (RUL: {equipment.rul} {equipment.rulUnit})
            </p>
            <p className="text-sm text-slate-600 dark:text-gray-400 mt-2">
                ПОСЛЕДНЕЕ ОБНОВЛЕНИЕ: {new Date().toLocaleTimeString('ru-RU')}
            </p>
        </div>
    );
};

const GuardDashboardView: React.FC<GuardDashboardViewProps> = ({ equipmentData, onSelectEquipment }) => {
    const time = useClock();

    const sortedData = useMemo(() => {
        return [...equipmentData].sort((a, b) => {
            return statusConfig[b.status].priority - statusConfig[a.status].priority;
        });
    }, [equipmentData]);

    return (
        <div className="fixed inset-0 w-full h-full flex flex-col p-4 sm:p-6 text-slate-800 dark:text-white">
            {/* Header */}
            <header className="flex-shrink-0 flex items-center justify-between px-4 py-2 rounded-2xl bg-white/20 dark:bg-black/10 backdrop-blur-lg border border-black/10 dark:border-white/10">
                <div className="flex items-center space-x-4">
                    <GearIcon className="h-8 w-8" />
                </div>
                <h1 className="text-xl sm:text-2xl md:text-3xl font-bold tracking-widest text-slate-800 dark:text-white">
                    ПРЕДИКТИВНАЯ ДИАГНОСТИКА: ОБЗОР АГРЕГАТОВ
                </h1>
                <div className="text-right">
                    <p className="text-xl sm:text-2xl font-bold text-slate-800 dark:text-white">{time.toLocaleTimeString('ru-RU')}</p>
                    <p className="text-xs sm:text-sm text-slate-600 dark:text-gray-300">{time.toLocaleDateString('ru-RU', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</p>
                </div>
            </header>

            {/* Main Content Grid */}
            <main className="flex-grow mt-6 overflow-y-auto">
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
                    {sortedData.map(eq => (
                        <GuardStatusCard key={eq.id} equipment={eq} onSelect={() => onSelectEquipment(eq)} />
                    ))}
                </div>
            </main>
        </div>
    );
};

export default GuardDashboardView;
